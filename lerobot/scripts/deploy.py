import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import torch
import time
import numpy as np

from lerobot.common.policies.act.modeling_act import ACTPolicy

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 或 BEST_EFFORT
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

class RobotController(Node):
    def __init__(self):
        super().__init__("robot_controller")

        # 修改 Publisher 訊息型態為 JointState
        self.left_publisher = self.create_publisher(JointState, "/left_follower/joint_states_control", 10)
        self.right_publisher = self.create_publisher(JointState, "/right_follower/joint_states_control", 10)

        # 設定運算設備
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 讀取 AI 模型 (ACTPolicy)
        ckpt_path = "/media/lester/Transcend/outputs/train/2025-02-27/16-40-11_real_world_act_act_koch_wipe0227/checkpoints/100000/pretrained_model/"
        self.policy = ACTPolicy.from_pretrained(ckpt_path, local_files_only=True).to(self.device)

        # 初始化影像轉換工具
        self.bridge = CvBridge()
        self.latest_image_top = None  # 儲存最新接收到的相機影像
        self.latest_image_left = None 
        self.latest_image_right = None 

        # 訂閱 ROS 2 相機影像 Topic
        self.create_subscription(Image, "/camera/camera_top/color/image_raw", self.image_top_callback, qos_profile)
        self.create_subscription(Image, "/camera/camera_left/color/image_raw", self.image_left_callback, qos_profile)
        self.create_subscription(Image, "/camera/camera_right/color/image_raw", self.image_right_callback, qos_profile)

        # 初始化關節狀態
        self.left_joint_positions = np.zeros(6)
        self.right_joint_positions = np.zeros(6)

        # 訂閱關節狀態 Topic
        self.create_subscription(JointState, "/left_follower/joint_states", self.left_joint_callback, 10)
        self.create_subscription(JointState, "/right_follower/joint_states", self.right_joint_callback, 10)

    def image_top_callback(self, msg):
        try:
            self.latest_image_top = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"影像轉換失敗: {str(e)}")
    
    def image_left_callback(self, msg):
        try:
            self.latest_image_left = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"影像轉換失敗: {str(e)}")
    
    def image_right_callback(self, msg):
        try:
            self.latest_image_right = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"影像轉換失敗: {str(e)}")

    def left_joint_callback(self, msg):
        """讀取左手臂關節狀態"""
        self.left_joint_positions = np.array(msg.position[:6])  # 取前 6 個關節

    def right_joint_callback(self, msg):
        """讀取右手臂關節狀態"""
        self.right_joint_positions = np.array(msg.position[:6])  # 取前 6 個關節

    def capture_observation(self):
        """擷取機械手臂關節狀態 & ROS 2 影像"""
        # 處理 top 相機：若未收到則用黑色影像填充
        if self.latest_image_top is None:
            self.get_logger().warn("未收到 top 相機影像，使用黑色影像填充")
            img_top = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img_top = self.latest_image_top

        # 處理 far 相機
        if self.latest_image_left is None:
            self.get_logger().warn("未收到 front 相機影像，使用黑色影像填充")
            img_left = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img_left = self.latest_image_left

        # 處理 frist 相機
        if self.latest_image_right is None:
            self.get_logger().warn("未收到 front 相機影像，使用黑色影像填充")
            img_right = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img_right = self.latest_image_right

        def process_image(img):
            img = torch.tensor(img, dtype=torch.float32) / 255.0
            return img.permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)

        img_tensor_top = process_image(img_top)
        img_tensor_left = process_image(img_left)
        img_tensor_right = process_image(img_right)

        # 轉換關節狀態為 PyTorch
        left_joint_positions = torch.tensor(self.left_joint_positions, dtype=torch.float32).unsqueeze(0).to(self.device)
        right_joint_positions = torch.tensor(self.right_joint_positions, dtype=torch.float32).unsqueeze(0).to(self.device)

        observation_state = torch.cat([left_joint_positions, right_joint_positions], dim=-1)

        observation = {
            "observation.state": observation_state,
            "observation.images.cam_top": img_tensor_top,  # 使用 ROS 2 提供的影像
            "observation.images.cam_right": img_tensor_right,
            "observation.images.cam_left": img_tensor_left
        }
        return observation

    def send_action(self, left_action, right_action):
        """發送機械手臂指令，使用 JointState 格式"""
        left_msg = JointState()
        right_msg = JointState()

        # 設定 header 與 timestamp
        now = self.get_clock().now().to_msg()
        left_msg.header.stamp = now
        right_msg.header.stamp = now

        # 設定 joint names (根據手臂要求的命名規則)
        left_msg.name = [f"left_joint_{i}" for i in range(6)]
        right_msg.name = [f"right_joint_{i}" for i in range(6)]

        # 設定關節位置
        left_msg.position = left_action.tolist()
        right_msg.position = right_action.tolist()

        self.left_publisher.publish(left_msg)
        self.right_publisher.publish(right_msg)

        self.get_logger().info(f"Sent action -> Left: {left_action}, Right: {right_action}")

    def run(self, inference_time_s=300, fps=50):
        """AI 控制機械手臂運行"""
        period = 1.0 / fps
        for _ in range(int(inference_time_s * fps)):
            start_time = time.perf_counter()

            # 處理 ROS 回呼
            rclpy.spin_once(self, timeout_sec=0.01)

            # 1. 擷取當前觀測值
            observation = self.capture_observation()

            # 2. AI 模型決定下一步行動
            with torch.no_grad():
                action = self.policy.select_action(observation)

            # 3. 解析 AI 動作輸出
            left_action = action[0, :6].cpu()
            right_action = action[0, 6:].cpu()

            # 4. 發送動作指令
            self.send_action(left_action, right_action)

            # 5. 控制 FPS
            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, period - dt_s))

def main():
    rclpy.init()
    controller = RobotController()
    controller.run()
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()