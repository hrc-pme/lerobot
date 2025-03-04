import h5py
import rclpy
import time
import numpy as np
import cv2
import yaml
from sensor_msgs.msg import JointState
from rclpy.node import Node

# 從 YAML 檔案讀取參數
with open('/root/lerobot/lerobot/configs/scripts/rosbag_player.yaml', 'r') as file:
    config = yaml.safe_load(file)

hdf5_path = config['hdf5_path']
left_follower_topic = config['left_follower_topic']
right_follower_topic = config['right_follower_topic']
DT = config['DT']

class HDF5FollowerPlayer(Node):
    def __init__(self, hdf5_path):
        super().__init__("hdf5_follower_player")
        self.left_pub = self.create_publisher(JointState, left_follower_topic, 10)
        self.right_pub = self.create_publisher(JointState, right_follower_topic, 10)
        self.load_hdf5(hdf5_path)

    def load_hdf5(self, hdf5_path):
        """ 讀取 HDF5 檔案 """
        with h5py.File(hdf5_path, 'r') as f:
            self.qpos = f["observations/qpos"][:]  # 取得 follower 關節數據 (T, 12)
            self.qpos = np.nan_to_num(self.qpos, nan=0.0)  # 轉換 NaN 為 0.0
            self.total_frames = self.qpos.shape[0]

            if "observations/images" in f:
                self.images = {}
                for cam_name in f["observations/images"].keys():
                    self.images[cam_name] = f[f"observations/images/{cam_name}"][:]
                self.get_logger().info(f"Loaded images from cameras: {list(self.images.keys())}")
            else:
                self.images = None
                self.get_logger().info("No image data found in HDF5.")

        self.get_logger().info(f"Loaded {self.total_frames} frames from {hdf5_path}")

    def play(self):
        """ 播放 JointState 讓 follower 動起來 """
        for i in range(self.total_frames):
            msg_left = JointState()
            msg_right = JointState()

            msg_left.header.stamp = self.get_clock().now().to_msg()
            msg_right.header.stamp = self.get_clock().now().to_msg()

            # 左手臂關節名稱與數據
            msg_left.name = [f"left_joint_{j}" for j in range(6)]
            msg_left.position = self.qpos[i, :6].tolist()  # 取前 6 個關節值

            # 右手臂關節名稱與數據
            msg_right.name = [f"right_joint_{j}" for j in range(6)]
            msg_right.position = self.qpos[i, 6:].tolist()  # 取後 6 個關節值

            print(f"Type of qpos[i, :6]: {type(self.qpos[i, :6])}")
            print(f"Value of qpos[i, :6]: {self.qpos[i, :6]}")
            print(f"Converted to list: {self.qpos[i, :6].tolist()}")
            print(f"正在播放第 {i+1} 幀，共 {self.total_frames} 幀")

            self.left_pub.publish(msg_left)  # 發送到 /left_follower/joint_states
            self.right_pub.publish(msg_right)  # 發送到 /right_follower/joint_states

            self.get_logger().info(f"Playing frame {i+1}/{self.total_frames}")

            if self.images is not None:
                combined_image = None
                for cam_name, img_data in self.images.items():
                    frame = img_data[i]  # 取得當前影像
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 轉換顏色
                    if combined_image is None:
                        combined_image = frame
                    else:
                        combined_image = np.hstack((combined_image, frame))  # 串接不同相機的影像

                cv2.imshow("Robot Camera View", combined_image)
                cv2.waitKey(1)  # 顯示 1ms

            time.sleep(DT)  # 等待 DT 秒

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    player = HDF5FollowerPlayer(hdf5_path)
    player.play()
    player.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()