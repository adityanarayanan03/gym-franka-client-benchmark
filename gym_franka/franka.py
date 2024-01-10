import cv2
import gym
import socket
import time
import numpy as np
import pyquaternion as pqt
import threading
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.registration import register
from gym_franka.realsense import RealSense
from gym_franka.rewards import REWARD_FUNCTIONS

import omegaconf
from networked_robotics_benchmarking.networks import ZMQ_Pair

BUFFER_SIZE = 1024

class FrankaEnv(gym.Env):
    def __init__(self, server_address, server_port, task,
                 realsense_sn=None, height=128, width=128,
                 enable_gripper=True):

        # Statespace (internal)
        self.position_origin = np.array([0.3, 0, 0.5])
        self.position_low = np.array([0.3, -0.5, 0])
        self.position_high = np.array([0.8, 0.5, 0.5])
        self.rotation_origin = pqt.Quaternion(0, -1, 0, 0)
        self.position_gain = 0.05
        self.rotation_gain = np.pi / 12
        self.position = self.position_origin.copy()
        self.rotation = pqt.Quaternion(self.rotation_origin)
        self.previous_position = self.position
        self.previous_rotation = self.rotation
        self.gripper_action = 1

        # Connect to gym-franka-server
        #self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.server_socket.connect((server_address, server_port))
        self.net_config = omegaconf.OmegaConf.load("net_config.yaml")
        self.network = ZMQ_Pair("client", **self.net_config)

        # Image observation settings
        self.realsense = RealSense(serial_numbers=realsense_sn)
        self.height = height
        self.width = width

        self.np_random = None
        self.viewer = None
        self.init = True

        self.task = task
        self.reward_function = REWARD_FUNCTIONS[self.task]
        self.enable_gripper = enable_gripper
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=float)
        self.observation_space = spaces.Box(low=0, high=255, shape=self._get_obs().shape, dtype=np.uint8)

        # For async update
        self.step_done = False
        self.blocking = True
        self.step_lock = threading.Lock()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def wait_for_response(self, timestamp):
        data = self.network.recv("server")
        return data.split(' ')[0]
        """
        while True:
            try:
                data = self.server_socket.recv(BUFFER_SIZE)
                if len(data) == 0:
                    return False
                recv_message = data.decode('utf8').split(' ')
                if len(recv_message) < 2:
                    continue
                if int(recv_message[1]) == timestamp:
                    return recv_message[0]
            except socket.timeout:
                pass
            except ConnectionResetError:
                return False
        """

    def reset(self):
        if not self.init:
            if self.task == 'lift-red':
                self.block_random_reset()
            if self.task == 'drawer':
                self.drawer_reset()
        while True:
            timestamp = time.time_ns()
            reset_command = f'<Reset> {timestamp}'

            #self.server_socket.send(reset_command.encode('utf8'))
            self.network.send("server", reset_command)

            self.position = self.position_origin.copy()
            self.rotation = pqt.Quaternion(self.rotation_origin)
            self.previous_position = self.position
            self.previous_rotation = self.rotation
            if self.wait_for_response(timestamp) == '<Success>':
                break
        self.init = False
        return self._get_obs()

    def block_random_reset(self):
        if self.reward_function(self._get_obs(), self.position) > 0:
            self.move_random_low()
            self.open_gripper()

    def drawer_reset(self):
        from gym_franka.rewards import blue_handle_grabbed
        if blue_handle_grabbed(self._get_obs()[3:].transpose((1, 2, 0))):
            print('Closing the drawer...')
            self.open_gripper()
            self.gripper_action = 1
            time.sleep(1)
            self.move_left_high()
            self.close_drawer()
        else:
            print('Left reset...')
            self.open_gripper()
            self.gripper_action = 1
            time.sleep(1)
            self.move_left()

    def move_hard_code(self, new_position, rotate=True):
        delta_position = (new_position - self.position) / 20
        delta_rotation = (self.rotation_origin / self.rotation) ** (1 / 20)
        for i in range(20):
            self.position += delta_position
            if rotate:
                self.rotation = delta_rotation * self.rotation
            self.send_move_command(wait=False)
            time.sleep(0.1)

    def move_random_low(self):
        new_position = np.random.uniform(low=np.array([0.4, -0.3, 0.01]),
                                         high=np.array([0.7, 0.3, 0.03]))
        while ((new_position[:2] - self.position_origin[:2]) ** 2).sum() > 0.16:
            new_position = np.random.uniform(low=np.array([0.4, -0.3, 0.01]),
                                             high=np.array([0.7, 0.3, 0.03]))
        self.move_hard_code(new_position)

    def move_left(self):
        new_position = self.position.copy()
        new_position[1] = max(0.2, new_position[1])
        new_position[2] = 0.3
        self.move_hard_code(new_position)

    def move_left_high(self):
        new_position = np.array([0.55, 0.2, 0.5])
        self.move_hard_code(new_position)

    def close_drawer(self):
        new_position = np.array([0.55, -0.24, 0.5])
        self.move_hard_code(new_position)

    def close_gripper(self):
        #self.server_socket.send(b'<Grasp>')
        self.network.send("server", "<Grasp>")

    def open_gripper(self):
        #self.server_socket.send(b'<Open>')
        self.network.send("server", "<Open>")

    def send_move_command(self, wait=True):
        command_key = '<Step-Wait>' if wait else '<Step>'
        command = f'{command_key} {self.position[0]:.6f} {self.position[1]:.6f} {self.position[2]:.6f} ' \
                  f'{self.rotation.w:.6f} {self.rotation.x:.6f} {self.rotation.y:.6f} {self.rotation.z:.6f} ' \
                  f'{self.gripper_action:.6f}'
        timestamp = time.time_ns()
        timed_command = f'{command} {timestamp}'

        #self.server_socket.send(timed_command.encode('utf8'))
        self.network.send("server", timed_command)

        response = self.wait_for_response(timestamp)
        if response == '<Reflex>':
            print('[gym-franka-FrankaEnv] Reflex corrected.')
            self.roll_back()
        elif not response:
            print('[gym-franka-FrankaEnv] Step failed.')
            return False
        return True

    def set_non_blocking(self):
        self.blocking = False

    def set_blocking(self):
        self.blocking = True

    def roll_back(self):
        self.position = self.previous_position
        self.rotation = self.previous_rotation

    def step(self, action):
        if not self.blocking:
            self.step_lock.acquire()
        action = np.array(action)
        action = np.clip(action, -1, 1)
        if not self.enable_gripper:
            action[-1] = 1

        new_position = self.position + action[:3] * self.position_gain
        self.previous_position = self.position
        self.position = new_position.clip(self.position_low, self.position_high)

        rx = pqt.Quaternion(axis=[1, 0, 0], radians=action[3] * self.rotation_gain)
        ry = pqt.Quaternion(axis=[0, 1, 0], radians=action[4] * self.rotation_gain)
        rz = pqt.Quaternion(axis=[0, 0, 1], radians=action[5] * self.rotation_gain)

        self.previous_rotation = self.rotation
        self.rotation = rx * ry * rz * self.rotation
        self.gripper_action = action[6]

        step_success = False
        while not step_success:
            step_success = self.send_move_command(wait=True)
        if not self.blocking:
            self.step_done = True
            self.step_lock.release()
            return None, None, False, {}
        else:
            return self.get_step_return()

    def get_step_return(self):
        obs = self._get_obs()
        reward = self.reward_function(obs, self.position)
        is_success = reward > 0
        done = is_success
        return obs, reward, done, {"is_success": is_success}

    def _get_obs(self):
        return self.render(mode='obs', height=self.height, width=self.width)

    def render(self, mode='human', height=None, width=None, camera_id=None):
        images = self.realsense.show_rgb()
        if mode == 'obs':
            for i in range(len(images)):
                images[i] = cv2.resize(images[i], (height, width))
            images = np.concatenate(images, axis=2).transpose((2, 0, 1))
            return images
        elif mode == 'rgb_array':
            camera_id = 0 if camera_id is None else camera_id
            resized_image = cv2.resize(images[camera_id], (height, width))
            return resized_image


def register_env():
    from gym_franka.realsense import SIDE_CAMERA_SN, HAND_CAMERA_SN

    register(
        id="Franka-v1",
        entry_point=FrankaEnv,
        max_episode_steps=30,
        reward_threshold=100,
        kwargs={'server_address': '10.42.0.139',
                'server_port': 8888,
                'task': None,
                'realsense_sn': [SIDE_CAMERA_SN, HAND_CAMERA_SN],
                'enable_gripper': True}
    )

    register(
        id="FrankaReachBlue-v1",
        entry_point=FrankaEnv,
        max_episode_steps=30,
        reward_threshold=100,
        kwargs={'server_address': '10.42.0.139',
                'server_port': 8888,
                'task': 'reach-blue',
                'realsense_sn': [SIDE_CAMERA_SN, HAND_CAMERA_SN],
                'enable_gripper': False}
    )

    register(
        id="FrankaLiftRed-v1",
        entry_point=FrankaEnv,
        max_episode_steps=30,
        reward_threshold=100,
        kwargs={'server_address': '10.42.0.139',
                'server_port': 8888,
                'task': 'lift-red',
                'realsense_sn': [SIDE_CAMERA_SN, HAND_CAMERA_SN],
                'enable_gripper': True}
    )

    register(
        id="FrankaDrawer-v1",
        entry_point=FrankaEnv,
        max_episode_steps=30,
        reward_threshold=100,
        kwargs={'server_address': '10.42.0.139',
                'server_port': 8888,
                'task': 'drawer',
                'realsense_sn': [SIDE_CAMERA_SN, HAND_CAMERA_SN],
                'enable_gripper': True}
    )
