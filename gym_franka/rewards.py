import numpy as np
import cv2


def sparse_reward(success):
    if success:
        return 100
    else:
        return -1


def red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_a = np.array([0, 80, 60], dtype='uint8')
    upper_a = np.array([5, 255, 255], dtype='uint8')
    lower_b = np.array([160, 80, 60], dtype='uint8')
    upper_b = np.array([180, 255, 255], dtype='uint8')
    mask_a = cv2.inRange(hsv, lower_a, upper_a) > 0
    mask_b = cv2.inRange(hsv, lower_b, upper_b) > 0
    mask = np.logical_or(mask_a, mask_b)
    return mask


def blue_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([80, 120, 100], dtype='uint8')
    upper = np.array([120, 255, 255], dtype='uint8')
    mask = cv2.inRange(hsv, lower, upper) > 0
    return mask


def blue_handle_grabbed(image):
    mask = blue_mask(image)
    height, width = mask.shape
    drawer_column = mask[:, width // 2:]
    coverage = drawer_column.sum() / (height * width / 2)
    return coverage > 0.4


def red_engaged(image):
    mask = red_mask(image)
    height, width = mask.shape
    lower_half = mask[height // 2:]
    coverage = lower_half.sum() / (height * width / 2)
    return coverage > 0.50


def blue_reached(image):
    mask = blue_mask(image)
    height, width = mask.shape
    coverage = mask.sum() / (height * width)
    return coverage > 0.25


def reach_blue_reward(obs):
    hand_image = obs[3:].transpose((1, 2, 0))
    return sparse_reward(blue_reached(hand_image))


def lift_red_reward(obs, position):
    hand_image = obs[3:].transpose((1, 2, 0))
    position_satisfied = position[2] > 0.2
    block_engaged = red_engaged(hand_image)
    return sparse_reward(position_satisfied and block_engaged)


def drawer_reward(obs, position):
    hand_image = obs[3:].transpose((1, 2, 0))
    position_satisfied = position[1] > -0.1
    handle_grabbed = blue_handle_grabbed(hand_image)
    return sparse_reward(position_satisfied and handle_grabbed)


REWARD_FUNCTIONS = {
    None: lambda *args: 0,
    'reach-blue': reach_blue_reward,
    'lift-red': lift_red_reward,
    'drawer': drawer_reward
}

if __name__ == '__main__':
    import time
    from gym_franka.realsense import RealSense, HAND_CAMERA_SN
    import matplotlib.pyplot as plt
    camera = RealSense([HAND_CAMERA_SN])
    figures = None
    while True:
        rs_image = camera.show_rgb()[0]
        if figures is None:
            figures = []
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            figures.append(axes.imshow(red_mask(rs_image)))
            axes.axis('off')
            fig.tight_layout()
        else:
            figures[0].set_data(red_mask(rs_image))
        red_engaged(rs_image)
        plt.draw()
        plt.pause(1e-6)
        time.sleep(0.1)
