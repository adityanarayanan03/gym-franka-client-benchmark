import time
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt


SIDE_CAMERA_SN = '134322071173'
HAND_CAMERA_SN = '134322071594'


class RealSense:
    def __init__(self, serial_numbers=None):
        self.figures = None

        self.ctx = rs.context()
        if type(serial_numbers) is not list:
            serial_numbers = [serial_numbers]
        self.num_cameras = len(serial_numbers)
        self.pipelines = []
        self.configs = []
        for sn in serial_numbers:
            pipeline = rs.pipeline(self.ctx)
            config = rs.config()
            if sn is not None:
                config.enable_device(str(sn))
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
            pipeline.start(config)
            self.pipelines.append(pipeline)
            self.configs.append(config)
        print('[gym-franka-RealSense] Waiting 5s for auto exposure...')
        time.sleep(5)

    def get_rgb(self):
        images = []

        for p in self.pipelines:
            frames = p.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_data = color_frame.as_frame().get_data()
            image = np.asanyarray(color_data)[:, 80: -80]
            images.append(image)
        return images

    def show_rgb(self):
        images = self.get_rgb()
        if self.figures is None:
            self.figures = []
            fig, axes = plt.subplots(1, self.num_cameras, figsize=(5 * self.num_cameras, 5))
            if self.num_cameras > 1:
                for i in range(self.num_cameras):
                    self.figures.append(axes[i].imshow(images[i]))
                    axes[i].axis('off')
            else:
                self.figures.append(axes.imshow(images[0]))
                axes.axis('off')
            fig.tight_layout()
        else:
            for i in range(self.num_cameras):
                self.figures[i].set_data(images[i])
        plt.draw()
        plt.pause(1e-6)
        return images

    def get_depth(self):
        images = []
        for p in self.pipelines:
            frames = p.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_data = depth_frame.as_frame().get_data()
            image = np.asanyarray(depth_data)[:, 80: -80]
            images.append(image)
        return images

    def __del__(self):
        for p in self.pipelines:
            p.stop()


if __name__ == '__main__':
    camera = RealSense([SIDE_CAMERA_SN, HAND_CAMERA_SN])
    while True:
        camera.show_rgb()
        time.sleep(0.1)
