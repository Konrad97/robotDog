from __future__ import print_function
import pybullet as p
import pybullet_data as pd
import gym
import numpy as np
from gym import spaces

class VisionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    robot = None
    plane = None

    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=-1.0, high=2.0, shape=(1, 16), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(1, 23), dtype=np.float32)

        self.reset()

    def step(self, actions):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        for idx, a in enumerate(actions):
            a = a * 300
            p.setJointMotorControl2(self.robot, idx, controlMode=p.VELOCITY_CONTROL, targetVelocity=a,
                                    force=10)

        pos_pre_step = np.array(p.getBasePositionAndOrientation(self.robot)[0])

        p.stepSimulation()

        pos_post_step = np.array(p.getBasePositionAndOrientation(self.robot)[0])

        # If the algorith walks forward -> give it reward
        reward = pos_post_step[0] - pos_pre_step[0]

        state_object, _ = p.getBasePositionAndOrientation(self.robot)

        observation = self.get_current_observations()

        return observation, reward, False, {'x': state_object[0]}

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything

        p.setGravity(0, 0, -10)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("quadruped/vision60.urdf", [0, 0, 0.45], useFixedBase=False)  # True)

        p.setPhysicsEngineParameter(numSolverIterations=10)

        for j in range(p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)

        p.setRealTimeSimulation(0)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return self.get_current_observations()

    def render(self, mode='human'):
        ...

    def close(self):
        p.disconnect()

    def get_current_observations(self):
        base_position_and_orientation = p.getBasePositionAndOrientation(self.robot)
        position = np.array(base_position_and_orientation[0])
        orientation = np.array(base_position_and_orientation[1])
        base_position_and_orientation_array = np.append(position, orientation)

        observation = base_position_and_orientation_array

        for joint in range(p.getNumJoints(self.robot)):
            joint_state = p.getJointState(self.robot, joint)
            joint_position = joint_state[0]
            joint_velocity = joint_state[1]
            # idx [2] would be: Joint reaction forces which are 0 without a torque sensor
            joint_motor_torque = joint_state[3]
            joint_state_array = np.array([joint_position, joint_velocity, joint_motor_torque])

            observation = np.append(observation, joint_state_array)

        return observation
