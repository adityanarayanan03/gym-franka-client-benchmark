from setuptools import setup

setup(name='gym_franka',
      version='0.1.0',
      description='Gym environment for Franka Emika Panda robot',
      packages=['gym_franka'],
      install_requires=['gym==0.21.0',
                        'opencv-python',
                        'numpy',
                        'pyrealsense2',
                        'pyquaternion',
                        'matplotlib'])
