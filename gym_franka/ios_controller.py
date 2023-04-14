import socket
import threading
import time
import numpy as np
import pyquaternion as pqt

BUFFER_SIZE = 1024


class Controller:
    def __init__(self, server_address, server_port):
        self.init_position = np.array([0.3, 0, 0.5])
        self.init_rotation = pqt.Quaternion(0, -1, 0, 0)
        self.env_position_gain = 0.05
        self.env_rotation_gain = np.pi / 12

        self.position = None
        self.rotation = None
        self.start_input_position = None
        self.start_input_rotation = None
        self.previous_position = None
        self.previous_rotation = None
        self.gripper_action = 1
        self.action = None
        self.live = False

        self.controller_position_gain = 4
        self.controller_rotation_gain = 4
        self.coordinate_change = np.array([[0, 0, -1],
                                           [-1, 0, 0],
                                           [0, 1, 0]])

        self.action_lock = threading.Lock()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(0.1)
        self.socket.bind((server_address, server_port))
        self.socket.listen(1)
        while True:
            try:
                self.connection, self.address = self.socket.accept()
                print('[gym-franka-iOS-Controller] Connected!')
                return
            except socket.timeout:
                pass

    def reset(self):
        self.action_lock.acquire()
        self.position = None
        self.rotation = None
        self.start_input_position = None
        self.start_input_rotation = None
        self.previous_position = None
        self.previous_rotation = None
        self.gripper_action = 1
        self.action = None
        self.live = False
        self.action_lock.release()

    def process_command(self):
        try:
            data = self.connection.recv(BUFFER_SIZE)
            if len(data) == 0:
                return False
        except ConnectionResetError:
            return False

        data_strings = data.decode('utf8').split(' ')

        command = data_strings[0]
        try:
            params = np.array([float(s) for s in data_strings[1:]])
        except ValueError:
            print(f'Failed to parse command: {data_strings}')
            return

        if command == '<Start>':
            self.live = True
            self.start_input_position = params[:3]
            self.start_input_rotation = pqt.Quaternion(array=params[3:7])
        elif command == '<Track>':
            if not self.live:
                return
            input_position = params[:3]
            input_rotation = pqt.Quaternion(array=params[3:7])

            delta_position = input_position - self.start_input_position
            delta_position = delta_position @ self.coordinate_change.T * self.controller_position_gain
            self.position = self.init_position + delta_position

            delta_rotation = input_rotation / self.start_input_rotation
            translated_rotation_vector = delta_rotation.vector @ self.coordinate_change.T
            delta_rotation = pqt.Quaternion(delta_rotation.w,
                                            *translated_rotation_vector) ** self.controller_rotation_gain
            self.rotation = delta_rotation * self.init_rotation

            if len(params) > 7:
                self.gripper_action = params[7]

        elif command == '<Gripper>':
            self.gripper_action = params[0]

        self.update_action()

    def compute_action(self):
        if self.position is None:
            return
        if self.previous_position is None:
            self.previous_position = self.init_position.copy()
            self.previous_rotation = pqt.Quaternion(self.init_rotation)
        delta_position = self.position - self.previous_position
        position_action = delta_position / self.env_position_gain
        delta_rotation = self.rotation / self.previous_rotation
        qw, qx, qy, qz = delta_rotation
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        rx = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
        ry = -np.pi / 2 + 2 * np.arctan2((1 + 2 * (qw * qy - qx * qz)) ** 0.5, (1 - 2 * (qw * qy - qx * qz)) ** 0.5)
        rz = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        rotation_action = np.array([rx, ry, rz]) / self.env_rotation_gain
        self.action = np.array([*position_action, *rotation_action, self.gripper_action]).clip(-1, 1)

    def update_action(self):
        self.action_lock.acquire()
        self.compute_action()
        self.action_lock.release()

    def get_action(self):
        while self.action is None:
            time.sleep(0.1)
        self.action_lock.acquire()
        current_action = self.action
        actual_delta_position = current_action[:3] * self.env_position_gain
        self.previous_position += actual_delta_position
        rx = pqt.Quaternion(axis=[1, 0, 0], radians=current_action[3] * self.env_rotation_gain)
        ry = pqt.Quaternion(axis=[0, 1, 0], radians=current_action[4] * self.env_rotation_gain)
        rz = pqt.Quaternion(axis=[0, 0, 1], radians=current_action[5] * self.env_rotation_gain)
        self.previous_rotation = rx * ry * rz * self.previous_rotation
        self.action = None
        self.action_lock.release()
        return current_action

    def spin(self):
        while True:
            self.process_command()


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    from _thread import start_new_thread
    e = gym.make('FrankaEngageRed-v1')
    done = True
    controller = Controller('192.168.0.1', 6789)
    start_new_thread(controller.spin, ())
    index = 0
    while True:
        if done:
            obs = e.reset()
            index = 0
            # plt.imsave(f'/home/rzhao/obs/{index:03d}.png', obs[3:].transpose((1, 2, 0)))
            controller.reset()
        index += 1
        obs, reward, done, info = e.step(controller.get_action())
        # plt.imsave(f'/home/rzhao/obs/{index:03d}.png', obs[3:].transpose((1, 2, 0)))
