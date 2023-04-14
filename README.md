# Gym Franka Client

## Getting started
- Install ```librealsense```.

- Install the pip package with:
    ```
    pip install -e .
    ```

## Additional setups
- Change the IP addresses in ```gym_franka/franka.py``` to point to the ROS machine running ```gym-franka-server```.
- Change the camera SN.

## Example
```
import gym
import gym_franka

e = gym.make('Franka-v1')
e.reset()
e.step(e.action_space.sample())
```