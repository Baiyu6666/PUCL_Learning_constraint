import gym
from gym import error, spaces, utils
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

DIM_OBS = 9 # no. of dimensions in observation space： 3 for robot position, 3 for robot relative position to the target, 3 for velocity
DIM_ACT = 3 # no. of dimensions in action space: 3 for velocity

class ReachBasic(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        mode = p.DIRECT
        p.connect(mode)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1.]*DIM_ACT), np.array([1.]*DIM_ACT))
        self.observation_space = spaces.Box(np.array([-1]*DIM_OBS), np.array([1]*DIM_OBS))
        self.starting_point_candidates = None
        self.ending_point_candidates = None
        self.task_config_idx = 0
        self.current_state = None
        p.setTimeStep(1/60)
        # p.disconnect()

    def switch_mode(self, ):
        # Disconnect the current connection
        p.disconnect()
        p.connect(p.GUI)

    def step(self, action):
        raise NotImplementedError("The 'step' method must be implemented in the subclass.")

    def reset(self, render_mode=False):
        self.cost_counter = 0
        # p.resetSimulation()
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        # urdfRootPath=pybullet_data.getDataPath()
        # p.setGravity(0, 0, -9.81)
        # self.debug_uid = p.addUserDebugText('', [0,0,0], textSize=1.5)
        #=========================================================================#
        #  Generate plane, panda arm, table and target object                     #
        #=========================================================================#
        # plane
        # planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        
        # panda arm
        # self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        # self.pandaUid = p.loadURDF("gym_panda/franka_panda/panda.urdf",useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)

        # rest_poses = [0,-0.215,0.,-3.,0,2.356,2.356,0.08,0.08]
        # for i in range(7):
        #     p.resetJointState(self.pandaUid,i, rest_poses[i])
        # p.resetJointState(self.pandaUid, 9, 0.08)
        # p.resetJointState(self.pandaUid,10, 0.08)

        if self.starting_point_candidates is not None:
            task_config_idx = round(self.task_config_idx)
            starting_point = self.starting_point_candidates[task_config_idx % self.starting_point_candidates.shape[0]][:3]
        else:
            starting_point = (random.uniform(0.35, 0.48), random.uniform(-0.09, 0.09), random.uniform(0.05, 0.1))
            # starting_point = (random.uniform(0.7, 0.8), random.uniform(-0.15, 0.15), random.uniform(0.05, 0.08))

        # orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        # rest_poses = p.calculateInverseKinematics(self.pandaUid, 11, starting_point, orientation, maxNumIterations=300,
        #                                           residualThreshold=.003)[0:7]
        # for i in range(7):
        #     p.resetJointState(self.pandaUid,i, rest_poses[i])
        #
        # # table
        # tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5,0,-0.65])
        
        # target object
        # ending_point= (random.uniform(0.72,0.77),random.uniform(-0.05, 0.05), 0.04)
        if self.ending_point_candidates is not None:
            task_config_idx = round(self.task_config_idx)
            ending_point = self.ending_point_candidates[task_config_idx % self.ending_point_candidates.shape[0]][:3]
        else:
            # while True:
            #     ending_point= (random.uniform(0.5,0.77),random.uniform(-0.09, 0.09), 0.04)
            #     if (ending_point[0] - 0.55) ** 2 + (ending_point[1]) ** 2 > 0.06 ** 2:
            #         break
            ending_point = (0.68, 0, 0.04)
        # Change to the next task configuration every two reset due to a double reset bug
        self.task_config_idx += 0.5

        # self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "lego/lego.urdf"), basePosition=ending_point, useFixedBase=True)
        # ending_point = np.array(p.getBasePositionAndOrientation(self.objectUid)[0])
        self.ending_point = ending_point

        #=========================================================================#
        #  Generate obstacles                                                     #
        #=========================================================================#
        #state_obstacle1 = (random.uniform(0.3,0.9),random.uniform(-0.3, 0.3),0.05)
        
        # OPTION 1: obstacle in front of the target object
        # state_obstacle1 = (ending_point[0]-0.2, ending_point[1],0.05)
        # self.obstacle1 = p.loadURDF("gym_panda/envs/cylinder.urdf", basePosition=state_obstacle1)
        
        # OPTION 2: generate X number of obstacles in random positions
        # for i in range(3):
        #     state_obstacle = (random.uniform(0.3,0.9),random.uniform(-0.3, 0.3),0.05)
        #     obstacle_name = 'obstacle' + str(i)
        #     setattr(self, obstacle_name, p.loadURDF("gym_panda/envs/cylinder.urdf", basePosition=state_obstacle))

        # state_obstacle1 = (0.55, 0, 0.05)
        # self.obstacle1 = p.loadURDF("gym_panda/envs/cylinder_area.urdf", basePosition=state_obstacle1)
        #=========================================================================#
        #  Observation definition                                                 #
        #=========================================================================#
        # state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])

        state_robot = np.array(starting_point)
        self.current_state = state_robot

        # state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = np.concatenate((state_robot, ending_point - state_robot, np.zeros((3))))
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        return self.observation.astype(np.float32)

    def set_starting_point(self, starting_point, ending_point=None):
        if starting_point is not None and starting_point.ndim == 1:
            starting_point = starting_point.reshape(1, -1)
        self.starting_point_candidates = starting_point
        # self.ending_point_candidates = ending_point

    def true_cost_function(self, obs, acs):
       raise NotImplementedError("The 'true_cost_function' method must be implemented in the subclass.")

    def render(self, mode='rgb_array'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5,0,0.05],
                                                            distance=1.4,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=320,
                                              height=240,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()


class ReachObs(ReachBasic):
    def step(self, action):
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        # last_state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])
        # =========================================================================#
        #  Execute Actions                                                        #
        # =========================================================================#

        # METHOD 1: action = dx,dy,dz in cartesian space (dim_act=4)
        # orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.008  # default: 0.005, how big are the actions
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = 0  # action[3]

        # currentPose = p.getLinkState(self.pandaUid, 11)
        # currentPosition = currentPose[0]

        currentPosition = self.current_state

        newPosition = [np.clip(currentPosition[0] + dx, 0, 1),
                       np.clip(currentPosition[1] + dy, -0.12, 0.12),
                       np.clip(currentPosition[2] + dz, 0, 0.12)]
        # jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)[0:7]
        #
        # # p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])
        #
        # for i in range(7):
        #     p.resetJointState(self.pandaUid,i, jointPoses[i])
        #
        # # for _ in range(10):
        # p.stepSimulation()

        # METHOD 2: action = delta_q (dim_act=7)
        # # Get the current joint angles
        # joint_angles = [p.getJointState(self.pandaUid, i)[0] for i in range(7)]
        #
        # # Apply the delta_q values from the action vector
        # dv = 0.1 # how big are the actions
        # joint_angles = [a + action[i]*dv for i, a in enumerate(joint_angles)]
        # p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL, joint_angles)
        # p.stepSimulation()

        # =========================================================================#
        #  Reward Function and Episode End States                                 #
        # =========================================================================#

        # ending_point = np.array(p.getBasePositionAndOrientation(self.objectUid)[0])
        ending_point = self.ending_point

        # state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])
        # state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        # tip_state = p.getLinkState(self.pandaUid, 10)[0] #p.getJointState(self.pandaUid, 10)

        state_robot = np.array(newPosition)
        last_state_robot = self.current_state
        self.current_state = state_robot

        self.observation = np.concatenate((state_robot, ending_point - state_robot, np.zeros(3)))

        #############
        # Cost function
        ##############

        cost = self.true_cost_function(self.observation, action)
        self.cost_counter += (cost == 1)

        info = {'cost': cost, 'is_success': False}  # {'object_position': ending_point}

        if np.linalg.norm(state_robot - ending_point) < 0.01:
            done_reward = 0
            done = True
            info.update({'is_success': self.cost_counter == 0})
        else:
            done = False
            done_reward = 0

        ########################
        # Reward function:
        ########################
        # Dense reward - displacement
        delta_distance = np.linalg.norm(state_robot - ending_point) - np.linalg.norm(last_state_robot - ending_point)
        dis_reward = delta_distance * - 2
        # Dense reward - distance to target
        # distance = np.linalg.norm(state_robot - ending_point)
        # dis_reward = distance * -1

        # Dense reward - time
        # ctl_reward = - 0.0001 # 0.0002

        # Dense reward - control penalty (minimum distance)
        ctl_reward = - 0.008 * np.linalg.norm(action)
        # ctl_reward = - 0.5 * np.linalg.norm(state_robot - last_state_robot)

        reward = dis_reward + ctl_reward + done_reward
        info.update(
            {'ctl_reward': ctl_reward, 'dis_reward': dis_reward, 'action_norm': np.linalg.norm(action, ord=np.inf)})
        # formatted_list = ["{:.2f}".format(x) for x in state_robot]
        # formatted_str = ", ".join(formatted_list)
        # p.addUserDebugText(formatted_str, state_robot, textSize=1.5, replaceItemUniqueId=self.debug_uid)
        return self.observation.astype(np.float32), reward, done, info

    def true_cost_function(self, obs, acs):
        # Note that here the input areare unnormalized observation and action
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        # Define true obstacle here. For example, here the robot is avoiding a tall cylinder.
        cost = (obs[:, 0] - 0.55) ** 2 + (obs[:, 1]) ** 2 < 0.04 ** 2
        if obs.shape[0] == 1:
            return cost[0]
        else:
            return cost


class ReachVel(ReachBasic):
    # only the velocity is defined in the cost function. The obstacle-avoidance is forced in the dynamics.

    def __init__(self):
        mode = p.DIRECT
        p.connect(mode)
        # Note that here the physical maximum velocity is different from that in ReachObs.
        self.action_space = spaces.Box(np.array([-1.2] * DIM_ACT), np.array([1.2] * DIM_ACT))
        # self.action_space = spaces.Box(np.array([-1, -0.7, -1.3]), np.array([1., 0.7, 1.3]))
        self.observation_space = spaces.Box(np.array([-1] * DIM_OBS), np.array([1] * DIM_OBS))
        self.starting_point_candidates = None
        self.ending_point_candidates = None
        self.task_config_idx = 0
        self.current_state = None
        p.setTimeStep(1 / 60)

    def step(self, action):
        dv = 0.008  # how big are the actions
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        currentPosition = self.current_state
        newPosition = [np.clip(currentPosition[0] + dx, 0, 0.75),
                       np.clip(currentPosition[1] + dy, -0.12, 0.12),
                       np.clip(currentPosition[2] + dz, 0, 0.1)]

        # We include the obstacle avoidance as part of the dynamics for easier policy training, since the obstacle is known in this environment.
        obstacle_center = [0.55, 0]
        obstacle_radius = 0.05

        delta_x = newPosition[0] - obstacle_center[0]
        delta_y = newPosition[1] - obstacle_center[1]
        dist = np.sqrt(delta_x ** 2 + delta_y ** 2)

        if dist < obstacle_radius:
            scaling_factor = obstacle_radius / dist
            newPosition[0] = obstacle_center[0] + delta_x * scaling_factor
            newPosition[1] = obstacle_center[1] + delta_y * scaling_factor

        # =========================================================================#
        #  Reward Function and Episode End States                                 #
        # =========================================================================#
        ending_point = self.ending_point
        state_robot = np.array(newPosition)
        last_state_robot = self.current_state
        self.current_state = state_robot
        self.observation = np.concatenate((state_robot, ending_point - state_robot, action))

        #############
        # Cost function
        ##############
        cost = self.true_cost_function(self.observation, action)
        self.cost_counter += (cost == 1)
        info = {'cost': cost, 'is_success': False}  # {'object_position': ending_point}

        if np.linalg.norm(state_robot - ending_point) < 0.01:
            done_reward = 0
            done = True
            info.update({'is_success': self.cost_counter == 0})
        else:
            done = False
            done_reward = 0

        ########################
        # Reward function:
        ########################
        # Dense reward - displacement
        # delta_distance = np.linalg.norm(state_robot - ending_point) - np.linalg.norm(last_state_robot - ending_point)
        # dis_reward = delta_distance * - 2
        # Dense reward - distance to target
        distance = np.linalg.norm(state_robot - ending_point)
        dis_reward = distance * -1

        # Dense reward - time
        ctl_reward = - 0.00

        # Dense reward - control penalty (minimum distance)
        # ctl_reward = - 0.008 * np.linalg.norm(action)
        # ctl_reward = - 0.5 * np.linalg.norm(state_robot - last_state_robot)

        reward = dis_reward + ctl_reward + done_reward
        info.update({'ctl_reward': ctl_reward, 'dis_reward': dis_reward, 'action_norm_inf': np.linalg.norm(action, ord=np.inf)})
        return self.observation.astype(np.float32), reward, done, info

    def true_cost_function(self, obs, acs):
        # Note that here the input must be unnormalized observation and action
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        # cost = (np.linalg.norm(obs[:, 6:9], ord=np.inf, axis=1) > 1)
        # cost = np.logical_or((np.linalg.norm(obs[:, 6:7], ord=np.inf, axis=1) > 1), (np.linalg.norm(obs[:, 7:8], ord=np.inf, axis=1) > 1))
        cost = np.logical_or((np.linalg.norm(obs[:, 6:7], ord=np.inf, axis=1) > 1), (np.linalg.norm(obs[:, 8:9], ord=np.inf, axis=1) > 0.4))
        # cost = (np.linalg.norm(obs[:, 6:8], ord=2, axis=1) > 1)

        if obs.shape[0] == 1:
            return cost[0]
        else:
            return cost


# This class is exclusively for making videos
class ReachRender(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        mode = p.GUI
        p.connect(mode)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1.] * DIM_ACT), np.array([1.] * DIM_ACT))
        self.observation_space = spaces.Box(np.array([-1] * DIM_OBS), np.array([1] * DIM_OBS))
        self.starting_point_candidates = None
        self.ending_point_candidates = None
        self.task_config_idx = 0
        self.current_state = None
        p.setTimeStep(1 / 60)
        # p.disconnect()

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        # last_state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])

        dv = 0.008  # default: 0.005, how big are the actions
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = 0  # action[3]

        # currentPose = p.getLinkState(self.pandaUid, 11)
        # currentPosition = currentPose[0]

        currentPosition = self.current_state

        newPosition = [np.clip(currentPosition[0] + dx, 0, 1),
                       np.clip(currentPosition[1] + dy, -0.12, 0.12),
                       np.clip(currentPosition[2] + dz, 0, 0.12)]
        # jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)[0:7]
        #
        # # p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])
        #
        # for i in range(7):
        #     p.resetJointState(self.pandaUid,i, jointPoses[i])
        #
        # # for _ in range(10):
        # p.stepSimulation()


        # =========================================================================#
        #  Reward Function and Episode End States                                 #
        # =========================================================================#
        # TODO: here is risky, some people said we should use [4], but I checkek their returns are the same and all other repos use 0
        position_goal = np.array(p.getBasePositionAndOrientation(self.objectUid)[0])
        # state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])
        # state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        # tip_state = p.getLinkState(self.pandaUid, 10)[0] #p.getJointState(self.pandaUid, 10)

        state_robot = np.array(newPosition)
        last_state_robot = self.current_state
        self.current_state = state_robot

        #############
        # Cost function
        ##############

        cost = 0
        # Cylinder position constraint
        if (state_robot[0] - 0.55) ** 2 + (state_robot[1]) ** 2 < 0.045 ** 2:
            # reward -= 0.8
            cost = 1
            self.cost_counter += 1
        info = {'cost': cost, 'is_success': False}  # {'object_position': ending_point}

        if np.linalg.norm(state_robot - position_goal) < 0.01:
            done_reward = 0
            done = True
            info.update({'is_success': self.cost_counter == 0})
        else:
            done = False
            done_reward = 0

        ########################
        # Reward function:
        ########################
        # Dense reward - displacement
        delta_distance = np.linalg.norm(state_robot - position_goal) - np.linalg.norm(last_state_robot - position_goal)
        dis_reward = delta_distance * - 2
        # Dense reward - distance to target
        # distance = np.linalg.norm(state_robot - ending_point)
        # dis_reward = distance * -1

        # Dense reward - time
        # ctl_reward = - 0.0001 # 0.0002

        # Dense reward - control penalty (minimum distance)
        ctl_reward = - 0.008 * np.linalg.norm(action)
        # ctl_reward = - 0.5 * np.linalg.norm(state_robot - last_state_robot)

        reward = dis_reward + ctl_reward + done_reward
        info.update(
            {'ctl_reward': ctl_reward, 'dis_reward': dis_reward, 'action_norm': np.linalg.norm(action, ord=np.inf)})
        self.observation = np.concatenate((state_robot, position_goal - state_robot, np.zeros(3)))
        # formatted_list = ["{:.2f}".format(x) for x in state_robot]
        # formatted_str = ", ".join(formatted_list)
        # p.addUserDebugText(formatted_str, state_robot, textSize=1.5, replaceItemUniqueId=self.debug_uid)
        return self.observation.astype(np.float32), reward, done, info

    def reset(self, render_mode=False):
        self.cost_counter = 0
        p.resetSimulation()
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -9.81)
        # self.debug_uid = p.addUserDebugText('', [0,0,0], textSize=1.5)
        # =========================================================================#
        #  Generate plane, panda arm, table and target object                     #
        # =========================================================================#
        # plane
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        # panda arm
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        self.pandaUid = p.loadURDF("gym_panda/franka_panda/panda.urdf",useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)

        rest_poses = [0,-0.215,0.,-3.,0,2.356,2.356,0.08,0.08]
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)

        if self.starting_point_candidates is not None:
            task_config_idx = round(self.task_config_idx)
            starting_point = self.starting_point_candidates[task_config_idx % self.starting_point_candidates.shape[0]][
                             :3]
        else:
            starting_point = (random.uniform(0.35, 0.4), random.uniform(-0.05, 0.05), random.uniform(0.05, 0.05))
            # starting_point = (random.uniform(0.7, 0.8), random.uniform(-0.15, 0.15), random.uniform(0.05, 0.08))

        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        rest_poses = p.calculateInverseKinematics(self.pandaUid, 11, starting_point, orientation, maxNumIterations=300,
                                                  residualThreshold=.003)[0:7]
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])

        # # table
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5,0,-0.65])

        # target object
        # ending_point= (random.uniform(0.72,0.77),random.uniform(-0.05, 0.05), 0.04)
        if self.ending_point_candidates is not None:
            position_goal = self.ending_point_candidates[task_config_idx % self.ending_point_candidates.shape[0]][:3]
        else:
            # while True:
            #     ending_point= (random.uniform(0.5,0.77),random.uniform(-0.09, 0.09), 0.04)
            #     if (ending_point[0] - 0.55) ** 2 + (ending_point[1]) ** 2 > 0.06 ** 2:
            #         break
            position_goal = (0.68, 0, 0.04)
        self.task_config_idx += 0.5

        collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0001)
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])  # 红色小球

        self.objectUid = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=collision_shape_id,
                                           baseVisualShapeIndex=visual_shape_id,
                                           basePosition=position_goal)
        position_goal = np.array(p.getBasePositionAndOrientation(self.objectUid)[0])

        for obstacle_position in  np.array([[0.55, 0.0, 0.05], [0.526, -0.032, 0.05], [0.527, 0.038, 0.05], [0.498, 0.07, 0.05]]):
            self.obstacle1 = p.loadURDF("gym_panda/envs/cup/cup_small.urdf", basePosition=obstacle_position)

        state_obstacle1 = (0.55, 0, 0.05)

        # =========================================================================#
        #  Observation definition                                                 #
        # =========================================================================#
        # state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])

        state_robot = np.array(starting_point)
        self.current_state = state_robot

        # state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = np.concatenate((state_robot, position_goal - state_robot, np.zeros((3))))
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        return self.observation.astype(np.float32)

    def set_starting_point(self, starting_point, ending_point=None):
        if starting_point.ndim == 1:
            starting_point = starting_point.reshape(1, -1)
        self.starting_point_candidates = starting_point
        # self.ending_point_candidates = ending_point

    def true_cost_function(self, obs, acs):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        # Define true obstacle
        cost = (obs[:, 0] - 0.55) ** 2 + (obs[:, 1]) ** 2 < 0.04 ** 2
        if obs.shape[0] == 1:
            return cost[0]
        else:
            return cost

    def render(self, mode='rgb_array'):
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0.05],
        #                                                   distance=1.4,
        #                                                   yaw=90,
        #                                                   pitch=-70,
        #                                                   roll=0,
        #                                                   upAxisIndex=2)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, -0., 0.03],
                                                          distance=0.8,
                                                          yaw=70,
                                                          pitch=-50,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=320*3,
                                            height=240*3,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    env = ReachObs()
    state_obstacle1 = [0.55, 0]
    # Draw the obstacle as a circle
    circle = plt.Circle((state_obstacle1[0], state_obstacle1[1]), 0.05, color='b', alpha=0.3)
    ax.add_artist(circle)

    for _ in range(30):
        state = env.reset()
        position_goal = state[3:5] + state[:2]
        # Plot the object and obstacle positions

        ax.plot(position_goal[0], position_goal[1], 'ro', label='Target Object')
        ax.plot(state_obstacle1[0], state_obstacle1[1], 'bo', label='Obstacle 1')

        # Draw the object as a circle
        circle_object = plt.Circle((position_goal[0], position_goal[1]), 0.01, color='r', alpha=0.3)
        ax.add_artist(circle_object)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0.4, 0.8)
    ax.set_ylim(-0.2, 0.2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Target Object and Obstacle Positions')
    plt.grid(True)
    plt.show()
