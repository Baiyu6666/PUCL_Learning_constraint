import time

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import gym
import numpy as np
import matplotlib.pyplot as plt

# 定义函数 f(x, y)
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import numpy as np
import os
import math
import random
from icrl.constraint_net import ConstraintNet
#
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
class Node:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=5, max_iter=500):
        self.start = Node(start)
        self.end = Node(goal)
        self.obstacle_list = obstacle_list
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]

    def planning(self):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if self.calc_distance_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if not self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Path not found

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate / 100:
            rnd = [np.random.uniform(self.min_rand, self.max_rand),
                   np.random.uniform(self.min_rand, self.max_rand)]
        else:
            rnd = [self.end.x, self.end.y]
        return Node(rnd)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node([from_node.x, from_node.y])
        d, theta = self.calc_distance_and_angle(from_node, to_node)

        extend_length = min(extend_length, d)

        new_node.x += extend_length * np.cos(theta)
        new_node.y += extend_length * np.sin(theta)
        new_node.parent = from_node

        return new_node

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

    def check_collision(self, node, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy
            if d <= size ** 2:
                return True  # collision
        return False  # safe

    def calc_distance_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return np.hypot(dx, dy)

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

@profile
def main():
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 1)
    ]

    rrt = RRT(start=[0, 0], goal=[6, 10], obstacle_list=obstacle_list, rand_area=[-2, 15])
    path = rrt.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")
        # Draw final path
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], '-r')

        # Draw obstacles
        for (ox, oy, size) in obstacle_list:
            circle = plt.Circle((ox, oy), size, color="blue")
            plt.gca().add_patch(circle)

        plt.plot([node.x for node in rrt.node_list], [node.y for node in rrt.node_list], "bo", markersize=3)
        plt.plot(rrt.start.x, rrt.start.y, "go", markersize=10)  # Start
        plt.plot(rrt.end.x, rrt.end.y, "ro", markersize=10)  # Goal
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()



# def main():
#     p.connect(p.GUI)
#     p.resetSimulation()
#     p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
#     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering during setup
#
#     urdfRootPath = pybullet_data.getDataPath()
#     p.setGravity(0, 0, -9.81)
#
#     # Load plane
#     planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
#
#     # Load panda arm
#     pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
#
#     rest_poses = [0, -0.215, 0., -3., 0, 2.356, 2.356, 0.08, 0.08]
#     for i in range(7):
#         p.resetJointState(pandaUid, i, rest_poses[i])
#     p.resetJointState(pandaUid, 9, 0.08)
#     p.resetJointState(pandaUid, 10, 0.08)
#
#     # Set starting point
#     starting_point = (random.uniform(0.35, 0.35), random.uniform(-0.09, 0.09), random.uniform(0.2, 0.2))
#     orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
#     rest_poses = p.calculateInverseKinematics(pandaUid, 11, starting_point, orientation, maxNumIterations=300, residualThreshold=.003)[0:7]
#     for i in range(7):
#         p.resetJointState(pandaUid, i, rest_poses[i])
#
#     # Load table
#     tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
#
#     # Load target object
#     ending_point = (0.68, 0, 0.04)
#     collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0001)
#     visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0, 0, 1])  # Red small sphere
#
#     objectUid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=ending_point)
#     ending_point = np.array(p.getBasePositionAndOrientation(objectUid)[0])
#
#     # Load obstacle
#
#     for obstacle_position in np.array([[0.55, 0.0, 0], [0.51, -0.0512, 0], [0.515, 0.052, 0], [0.477, 0.103, 0]]):
#         p.loadURDF("gym_panda/envs/cup/cup_small.urdf", basePosition=obstacle_position)
#
#
#     # Set observation
#     state_robot = np.array(starting_point)
#     current_state = state_robot
#     observation = np.concatenate((state_robot, ending_point - state_robot, np.zeros((3))))
#
#     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Enable rendering
#
#     # Plot demonstrations
#     from icrl import utils
#     (obs_e, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data('icrl/expert_data/PandaDSConcave', 20)
#     for i in range(0, obs_e.shape[0]-1, 2):
#         start_point = obs_e[i][:3]
#         end_point = obs_e[i + 1][:3]
#         if np.linalg.norm(start_point - end_point) >0.1:
#             continue
#         p.addUserDebugLine(start_point, end_point, lineColorRGB=[0, 0.8, 0], lineWidth=4.0)
#
#     # Plot learned constraints
#
#
#     while True:
#         p.stepSimulation()
#         time.sleep(1)
#
# if __name__ == "__main__":
#     main()




# # 更新方程以包含w参数
# def updated_equation(x, r, w):
#     return 1/x - r/(1-x) + w*(1+r)
#
# # 生成w和r的值
# w_values = np.linspace(0, 1.5, 50)
# r_values = np.linspace(0.2, 5, 50)
#
# # 初始化解的网格
# X_solutions = np.zeros((len(w_values), len(r_values)))
#
# # 对每对(w, r)求解方程
# for i, w in enumerate(w_values):
#     for j, r in enumerate(r_values):
#         # 使用fsolve求解，提供一个初始猜测值
#         x_solution = fsolve(updated_equation, 0.5, args=(r, w))
#         X_solutions[i, j] = x_solution[0]
#
# W, R = np.meshgrid(r_values, w_values)
#
# # 绘制三维图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(R, W, X_solutions, cmap='viridis')
#
# ax.set_xlabel('r')
# ax.set_ylabel('w')
# ax.set_zlabel('x')
# ax.set_title('Solution of the equation as a function of w and r')
#
# # 定义固定的r值，按照log比例划分
# r_fixed_values = np.logspace(np.log10(0.3), np.log10(100), 5)
#
# # 生成w的值
# w_values = np.linspace(0, 2.5, 100)
#
# # 绘图
# plt.figure(figsize=(10, 6))
#
# # 对每个固定的r值求解方程并绘制曲线
# for r in r_fixed_values:
#     x_solutions = []
#     for w in w_values:
#         x_solution = fsolve(updated_equation, 0.5, args=(r, w))
#         x_solutions.append(x_solution[0])
#     plt.plot(w_values, x_solutions, label=f'r={r:.2f}')
#     w_0_index = np.where(np.isclose(w_values, .0, atol=0.05))[0][0]
#     x_at_r_1 = x_solutions[w_0_index]
#     plt.scatter([0], [x_at_r_1], color='red')
#     plt.text(0, x_at_r_1, f'x={x_at_r_1:.4f}', color='red', verticalalignment='bottom')
#
# plt.xlabel('w')
# plt.ylabel('x')
# plt.title('Solution of the equation as a function of w for different r')
# plt.legend()
# plt.grid(True)
#
#
# w_fixed_values = np.linspace(0, 1.5, 5)
#
# # 生成r的值
# r_values = np.linspace(0.2, 60, 300)
#
# # 绘图
# plt.figure(figsize=(10, 6))
#
# # 对每个固定的w值求解方程并绘制曲线
# for w in w_fixed_values:
#     x_solutions = []
#     for r in r_values:
#         x_solution = fsolve(updated_equation, 0.5, args=(r, w))
#         x_solutions.append(x_solution[0])
#     plt.plot(r_values, x_solutions, label=f'w={w:.2f}')
#     x_at_max_r = x_solutions[-1]  # 最后一个元素对应最大的r值
#     max_r = r_values[-1]
#     plt.annotate(f'({max_r:.2f}, {x_at_max_r:.2f})', xy=(max_r, x_at_max_r), textcoords="offset points", xytext=(5, 5),
#                  ha='center')
#     r_1_index = np.where(np.isclose(r_values, 1.0, atol=0.05))[0][0]
#     x_at_r_1 = x_solutions[r_1_index]
#     plt.scatter([1], [x_at_r_1], color='red')
#     plt.text(1, x_at_r_1, f'x={x_at_r_1:.2f}', color='red', verticalalignment='bottom')
#
# plt.xlabel('r')
# plt.ylabel('x')
# plt.title('Solution of the equation as a function of r for different w')
# plt.legend()
# plt.grid(True)
# plt.show()