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
import time

# 從 YAML 檔案中載入參數
config_path = os.path.join(os.path.dirname(__file__), '../configs/scripts/rosbag2hdf5.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

rosbag_dir = config['rosbag_dir']
output_dir = config['output_dir']
max_frames = config['max_frames']
DT = config['DT']
camera_topics = config['camera_topics']
follower_left_topic = config['follower_left_topic']
follower_right_topic = config['follower_right_topic']
left_leader_topic = config['left_leader_topic']
right_leader_topic = config['right_leader_topic']
language = config.get('language', 'en')

os.makedirs(output_dir, exist_ok=True)

# 收集所有子目錄中的 .db3 檔案
rosbag_files = []
rosbag_dir_abs = os.path.abspath(rosbag_dir)
for subdir in sorted(os.listdir(rosbag_dir_abs)):
    subdir_path = os.path.join(rosbag_dir_abs, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith(".db3"):
                full_path = os.path.join(subdir_path, file)
                rosbag_files.append(full_path)

# 依 `.db3` 檔案的修改時間排序
rosbag_files = sorted(rosbag_files, key=lambda x: os.path.getmtime(x))

rclpy.init()

logger = get_logger('rosbag2_storage')
logger.set_level(rclpy.logging.LoggingSeverity.WARN)

# 最近鄰插值函式
def resample_array(data, target_length):
    """重採樣 2D 陣列 (N, d) 到 target_length（最近鄰插值）。"""
    N = data.shape[0]
    indices = np.round(np.linspace(0, N - 1, target_length)).astype(int)
    return data[indices]

def resample_images(images, target_length):
    """重採樣影像序列 (N, H, W, C) 到 target_length（最近鄰插值）。"""
    N = images.shape[0]
    indices = np.round(np.linspace(0, N - 1, target_length)).astype(int)
    return images[indices]

total_processing_time = 0
total_capacity_ratio_change = 0

# 開始轉換
for episode_idx, rosbag_path in enumerate(rosbag_files):
    start_time = time.time()
    
    rosbag_filename = os.path.basename(rosbag_path)
    if language == 'ch':
        print(f"\033[94m處理 {rosbag_filename} → episode_{episode_idx}.hdf5 ...\033[0m")
    else:
        print(f"\033[94mProcessing {rosbag_filename} → episode_{episode_idx}.hdf5 ...\033[0m")

    output_hdf5 = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")

    # 初始化 ROS bag 讀取器
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    # 初始化影像與機械臂數據儲存
    bridge = CvBridge()
    image_data = {key: [] for key in camera_topics.keys()}
    qpos_left_list, qpos_right_list = [], []
    leader_left_list, leader_right_list = [], []

    # 讀取 rosbag2 內的影像與 JointState 資料
    while reader.has_next():
        topic, msg, t = reader.read_next()

        if topic in camera_topics.values():
            cam_name = [k for k, v in camera_topics.items() if v == topic][0]
            compressed_image = deserialize_message(msg, CompressedImage)
            cv_image = bridge.compressed_imgmsg_to_cv2(compressed_image, desired_encoding="bgr8")
            cv_image = cv2.resize(cv_image, (480, 640))  
            image_data[cam_name].append(cv_image)

        if topic == follower_left_topic:
            joint_msg = deserialize_message(msg, JointState)
            qpos_left_list.append(joint_msg.position[:6])  
        if topic == follower_right_topic:
            joint_msg = deserialize_message(msg, JointState)
            qpos_right_list.append(joint_msg.position[:6])
        if topic == left_leader_topic:
            joint_msg = deserialize_message(msg, JointState)
            leader_left_list.append(joint_msg.position[:6])
        if topic == right_leader_topic:
            joint_msg = deserialize_message(msg, JointState)
            leader_right_list.append(joint_msg.position[:6])

    # 轉 NumPy 陣列
    for key in image_data.keys():
        image_data[key] = np.array(image_data[key], dtype=np.uint8)

    # 統一時間長度
    T_qpos = min(len(qpos_left_list), len(qpos_right_list))
    qpos_left = np.array(qpos_left_list[:T_qpos], dtype=np.float32)
    qpos_right = np.array(qpos_right_list[:T_qpos], dtype=np.float32)
    qpos = np.hstack((qpos_left, qpos_right))

    T_action = min(len(leader_left_list), len(leader_right_list))
    leader_left = np.array(leader_left_list[:T_action], dtype=np.float32)
    leader_right = np.array(leader_right_list[:T_action], dtype=np.float32)
    action = np.hstack((leader_left, leader_right))

    qpos = resample_array(qpos, max_frames)
    action = resample_array(action, max_frames)

    print(f"Image shape: {image_data['cam_top'].shape} | qpos shape: {qpos.shape} | action shape: {action.shape}")

    for key in image_data.keys():
        image_data[key] = resample_images(image_data[key], max_frames)

    for key in image_data.keys():
        image_data[key] = np.transpose(image_data[key], (0, 2, 1, 3))  # (T, W, H, C) -> (T, H, W, C)

    # 儲存 HDF5
    with h5py.File(output_hdf5, 'w') as f:
        f.attrs["sim"] = False  # 這是機器人數據
        obs = f.create_group("observations")
        image_group = obs.create_group("images")

        # 使用 HDF5 可變長數據類型來存儲不同大小的 JPEG 影像
        vlen_dtype = h5py.special_dtype(vlen=np.uint8)

        for cam_name, images in image_data.items():
            compressed_images = []
            for img in images:
                _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                compressed_images.append(np.array(buffer, dtype=np.uint8))
            
            # **正確方式：存為 (T, N)，滿足 Hugging Face 格式**
            max_length = max(len(img) for img in compressed_images)  # 找到最長的壓縮影像
            padded_images = np.zeros((len(compressed_images), max_length), dtype=np.uint8)  # 建立填充陣列

            for i, img in enumerate(compressed_images):
                padded_images[i, : len(img)] = img  # 填充影像
            
            # 儲存為 2D (T, N)
            image_group.create_dataset(cam_name, data=padded_images, dtype="uint8")

        obs.create_dataset("qpos", data=qpos, dtype="float32")  
        f.create_dataset("action", data=action, dtype="float32") 

        # 補上時間標記
        num_frames = max_frames
        f.create_dataset("frame_index", data=np.arange(num_frames), dtype="int64")  
        f.create_dataset("timestamp", data=np.arange(num_frames) * DT, dtype="float32")  
        f.create_dataset("next.done", data=np.zeros(num_frames, dtype=bool), dtype="bool")  
        f["next.done"][-1] = True  

        if language == 'ch':
            print(f"\033[92mHDF5 儲存完成: {output_hdf5}\033[0m")
        else:
            print(f"\033[92mSuccessfully saved in {output_hdf5}\033[0m")

    # Print HDF5 file size and processing time in the specified format
    hdf5_size = os.path.getsize(output_hdf5) / (1024 * 1024)  # Convert to MB
    end_time = time.time()
    processing_time = end_time - start_time
    total_processing_time += processing_time

    if language == 'ch':
        print(f"\033[33m容量: {hdf5_size:.2f} MB | 處理時間: {processing_time:.2f} 秒\n\033[0m")
    else:
        print(f"\033[33mCapacity: {hdf5_size:.2f} MB | Processing time: {processing_time:.2f} secs\n\033[0m")

    # Calculate capacity ratio change
    initial_size = os.path.getsize(rosbag_path) / (1024 * 1024)  # Convert to MB
    capacity_ratio_change = hdf5_size / initial_size
    total_capacity_ratio_change += capacity_ratio_change

average_capacity_ratio_change = total_capacity_ratio_change / len(rosbag_files)

if language == 'ch':
    print(f"總處理時間: {total_processing_time:.2f} 秒")
    print(f"平均容量比率變化: {average_capacity_ratio_change:.2f}")
else:
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Average capacity ratio change: {average_capacity_ratio_change:.2f}")

rclpy.shutdown()