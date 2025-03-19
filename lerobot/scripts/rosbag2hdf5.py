import os
import h5py
import rclpy
import rosbag2_py
import numpy as np
import cv2
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from rclpy.serialization import deserialize_message
from rclpy.logging import get_logger
from typing import Dict, List, Tuple
import time

class Rosbag2HDF5Converter:
  """Converts rosbag2 (.db3) files to HDF5 format."""

  def __init__(self, config_path: str):
    """
    Initializes the converter with configuration parameters.

    Args:
      config_path (str): Path to the YAML configuration file.
    """
    self.config = self._load_config(config_path)
    self.rosbag_dir = self.config['rosbag_dir']
    self.output_dir = self.config['output_dir']
    self.max_frames = self.config['max_frames']
    self.fps = self.config['fps']
    self.camera_topics = self.config['camera_topics']
    self.follower_right_topic = self.config['follower_right_topic']
    self.follower_left_topic = self.config['follower_left_topic']
    self.left_leader_topic = self.config['left_leader_topic']
    self.right_leader_topic = self.config['right_leader_topic']
    self.language = self.config.get('language', 'en')
    self.qpos_align_method = self.config.get('qpos_align_method', 'linear')
    self.image_align_method = self.config.get('image_align_method', 'nearest')
    self.logger = get_logger(self.__class__.__name__)
    self.bridge = CvBridge()
    os.makedirs(self.output_dir, exist_ok=True)


  @staticmethod
  def _load_config(config_path: str) -> Dict:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
      return yaml.safe_load(file)


  def _find_rosbag_files(self) -> List[str]:
    """Finds and sorts rosbag2 (.db3) files in the specified directory."""
    rosbag_files = []
    rosbag_dir_abs = os.path.abspath(self.rosbag_dir)

    # find all rosbag files in subdirectories
    for subdir in sorted(os.listdir(rosbag_dir_abs)):
      subdir_path = os.path.join(rosbag_dir_abs, subdir)
      if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
          if file.endswith(".db3"):
            full_path = os.path.join(subdir_path, file)
            rosbag_files.append(full_path)

    return sorted(rosbag_files, key=lambda x: os.path.getmtime(x))


  @staticmethod
  def _resample_array(data: np.ndarray, target_length: int) -> np.ndarray:
    """Resamples a 2D array using nearest-neighbor interpolation."""
    n = data.shape[0]
    indices = np.round(np.linspace(0, n - 1, target_length)).astype(int)
    return data[indices]


  @staticmethod
  def _resample_images(images: np.ndarray, target_length: int) -> np.ndarray:
    """Resamples an image sequence using nearest-neighbor interpolation."""
    n = images.shape[0]
    indices = np.round(np.linspace(0, n - 1, target_length)).astype(int)
    return images[indices]


  @staticmethod
  def _linear_interpolate(data: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly interpolates a 2D array to the target length."""
    n = data.shape[0]
    x = np.linspace(0, n - 1, n)
    x_new = np.linspace(0, n - 1, target_length)
    interpolated_data = np.array([np.interp(x_new, x, data[:, i]) for i in range(data.shape[1])]).T
    return interpolated_data


  def _process_rosbag(self, rosbag_path: str, episode_idx: int) -> Tuple[float, float, float]:
    """Processes a single rosbag2 file and saves it as HDF5."""
    # Initialize variables
    start_time = time.time()
    rosbag_filename = os.path.basename(rosbag_path)
    output_hdf5 = os.path.join(self.output_dir, f"episode_{episode_idx}.hdf5")

    self._log(f"Processing {rosbag_filename} → episode_{episode_idx}.hdf5 ...", color="\033[94m")

    # Open rosbag2 reader 
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    # Initialize data containers
    image_data = {key: [] for key in self.camera_topics.keys()}
    qpos_left_list, qpos_right_list = [], []
    leader_left_list, leader_right_list = [], []

    # Read messages from rosbag2 file
    while reader.has_next():
      topic, msg, _ = reader.read_next()
      # Process messages
      if topic in self.camera_topics.values():
        cam_name = [k for k, v in self.camera_topics.items() if v == topic][0]  # Extract camera name from topic
        compressed_image = deserialize_message(msg, CompressedImage)  # Deserialize compressed image message
        # Convert compressed image to OpenCV format
        cv_image = self.bridge.compressed_imgmsg_to_cv2(compressed_image, desired_encoding="bgr8")
        cv_image = cv2.resize(cv_image, (480, 640))
        image_data[cam_name].append(cv_image)
      elif topic == self.follower_left_topic:
        joint_msg = deserialize_message(msg, JointState)  # Deserialize joint state message
        qpos_left_list.append(joint_msg.position[:6])  # Extract joint positions
      elif topic == self.follower_right_topic:
        joint_msg = deserialize_message(msg, JointState)
        qpos_right_list.append(joint_msg.position[:6])
      elif topic == self.left_leader_topic:
        joint_msg = deserialize_message(msg, JointState)
        leader_left_list.append(joint_msg.position[:6])
      elif topic == self.right_leader_topic:
        joint_msg = deserialize_message(msg, JointState)
        leader_right_list.append(joint_msg.position[:6])


    # Resample and save data to HDF5 
    for key in image_data.keys():
      image_data[key] = self._resample_images(image_data[key], self.max_frames)
      image_data[key] = np.transpose(image_data[key], (0, 3, 1, 2))

    # Process joint data
    qpos, action = self._process_joint_data(qpos_left_list, qpos_right_list, leader_left_list, leader_right_list)

    self._save_hdf5(output_hdf5, image_data, qpos, action)
    self._log(f"HDF5 saved: {output_hdf5}", color="\033[92m")

    hdf5_size = os.path.getsize(output_hdf5) / (1024 * 1024)
    processing_time = time.time() - start_time
    self._log(f"Capacity: {hdf5_size:.2f} MB | Processing time: {processing_time:.2f} secs\n", color="\033[33m")

    return hdf5_size, processing_time, os.path.getsize(rosbag_path) / (1024 * 1024)


  def _process_joint_data(self, qpos_left_list: List[np.ndarray], qpos_right_list: List[np.ndarray], leader_left_list: List[np.ndarray], leader_right_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Processes and resamples joint data."""
    # Find the minimum length of joint data
    t_qpos = min(len(qpos_left_list), len(qpos_right_list))
    qpos_left = np.array(qpos_left_list[:t_qpos], dtype=np.float32)
    qpos_right = np.array(qpos_right_list[:t_qpos], dtype=np.float32)
    qpos = np.hstack((qpos_left, qpos_right))

    # Find the minimum length of leader joint data
    t_action = min(len(leader_left_list), len(leader_right_list))
    leader_left = np.array(leader_left_list[:t_action], dtype=np.float32)
    leader_right = np.array(leader_right_list[:t_action], dtype=np.float32)
    action = np.hstack((leader_left, leader_right))

    # Linearly interpolate joint data
    if self.qpos_align_method == 'linear':
      qpos = self._linear_interpolate(qpos, self.max_frames)
      action = self._linear_interpolate(action, self.max_frames)
    elif self.qpos_align_method == 'nearest':
      qpos = self._resample_array(qpos, self.max_frames)
      action = self._resample_array(action, self.max_frames)
    
    return qpos, action


  def _save_hdf5(self, output_hdf5: str, image_data: Dict[str, np.ndarray], qpos: np.ndarray, action: np.ndarray):
    """Saves processed data to an HDF5 file."""
    # Save data to HDF5
    with h5py.File(output_hdf5, 'w') as f:
      f.attrs["sim"] = False # Set to True if the data is from simulation
      obs = f.create_group("observations")
      image_group = obs.create_group("images")
      vlen_dtype = h5py.special_dtype(vlen=np.uint8)

      # Save images
      for cam_name, images in image_data.items():
        compressed_images = []
        for img in images:
          _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
          compressed_images.append(np.array(buffer, dtype=np.uint8))

        # Pad images to the same length
        max_length = max(len(img) for img in compressed_images)
        padded_images = np.zeros((len(compressed_images), max_length), dtype=np.uint8)

        # Fill padded images
        for i, img in enumerate(compressed_images):
          padded_images[i, : len(img)] = img

        # Save images to HDF5
        image_group.create_dataset(cam_name, data=padded_images, dtype="uint8")

      # Save joint data
      obs.create_dataset("qpos", data=qpos, dtype="float32")
      f.create_dataset("action", data=action, dtype="float32")

      # Save metadata
      num_frames = self.max_frames
      f.create_dataset("frame_index", data=np.arange(num_frames), dtype="int64")
      f.create_dataset("timestamp", data=np.arange(num_frames) / self.fps, dtype="float32")
      f.create_dataset("next.done", data=np.zeros(num_frames, dtype=bool), dtype="bool")
      f["next.done"][-1] = True  # Set the last frame as the terminal state


  def _log(self, message: str, color: str = ""):
    """Logs a message with optional color."""
    print(f"{color}{message}\033[0m" if color else message)


  def convert(self):
    """Converts all rosbag2 files to HDF5."""
    rclpy.init()

    # Suppress rosbag2_storage warnings
    rclpy.logging.set_logger_level('rosbag2_storage', rclpy.logging.LoggingSeverity.FATAL)
    rclpy.logging.set_logger_level('', rclpy.logging.LoggingSeverity.FATAL)

    # Find and sort rosbag files
    rosbag_files = self._find_rosbag_files()
    total_processing_time = 0
    total_capacity_ratio_change = 0

    # Process each rosbag file
    for episode_idx, rosbag_path in enumerate(rosbag_files):
      hdf5_size, processing_time, initial_size = self._process_rosbag(rosbag_path, episode_idx)

      # Update statistics
      total_processing_time += processing_time
      total_capacity_ratio_change += hdf5_size / initial_size

    average_capacity_ratio_change = total_capacity_ratio_change / len(rosbag_files)

    self._log(f"Total processing time: {total_processing_time:.2f} seconds")
    self._log(f"Average capacity ratio change: {average_capacity_ratio_change:.2f}")

    rclpy.shutdown()

if __name__ == "__main__":
  config_path = os.path.join(os.path.dirname(__file__), './configs/rosbag2hdf5.yaml')
  converter = Rosbag2HDF5Converter(config_path)
  converter.convert()
