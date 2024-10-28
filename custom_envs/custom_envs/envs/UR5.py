import os
import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import const

ABS_PATH = os.path.abspath(os.path.dirname(__file__))


class UR5Env(mujoco_env.MujocoEnv):
    def __init__(
            self,
            xml_file=ABS_PATH + '/xmls/UR5+gripper/UR5gripper_v3.xml',
            *args,
            **kwargs
    ):
        # self.target_pos = np.array([0, -0.68, 1.02])
        self.target_thresh = 0.03
        self.target_name = 'target'
        self.starting_point_idx = 0
        self.starting_point = None
        self.starting_point_candidates = None
        super(UR5Env, self).__init__(xml_file, 5)  # 5 is the frame skipping

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward_ctrl = - 0.01 * np.square(action).sum()

        def quat_to_euler_xyz(quat):
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = np.arctan2(t0, t1)

            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = np.arcsin(t2)

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = np.arctan2(t3, t4)

            return np.array([roll_x, pitch_y, yaw_z])

        end_effector_pos = self.data.get_site_xpos("gripper_endpoint")
        target_pos = self.data.get_site_xpos(self.target_name)
        # end_effector_euler = quat_to_euler_xyz(self.data.get_body_xquat("ee_link"))

        distance = np.linalg.norm(end_effector_pos - target_pos)
        # angle_difference = np.abs(end_effector_euler[2]) + np.abs(end_effector_euler[1] - np.pi)
        reward_dist = -1 * distance  # - 0.05 * angle_difference
        # print(distance)
        bonus = 5
        done = False
        if distance < self.target_thresh:  # and angle_difference < 0.05:
            reward_dist += bonus
            done = True
        reward = reward_ctrl + reward_dist
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        # self.viewer.render()

        # print(self.data.qpos)
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        if self.starting_point is not None:
            init = self.starting_point.copy()
            self.starting_point = None
            self.set_state(init[:self.model.nq], init[self.model.nq:])
        elif self.starting_point_candidates is not None:
            init = self.starting_point_candidates[
                round(self.starting_point_idx) % self.starting_point_candidates.shape[0]]
            self.starting_point_idx += 0.5
            self.set_state(init[:self.model.nq], init[self.model.nq:])
        else:
            # init_qpos = np.array([1.9516, 0.000763007, -2.28516])
            init_qpos = np.array([1.9516, 0.000763007, -2.28516, -2.46196, 1.8283])
            while True:
                qpos = init_qpos + self.np_random.uniform(low=-1.5, high=1.5, size=self.model.nq)
                qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .03
                self.set_state(qpos, qvel)
                end_effector_pos = self.sim.data.get_site_xpos("gripper_endpoint")
                # Make sure the end-effector is above the table and there is no self collision.
                if end_effector_pos[2] >= .9 and end_effector_pos[2] <= 1.5 and end_effector_pos[1] <= 0. and self.sim.data.ncon == 0:
                    break
        self.step_count = 0
        return self._get_obs()

    def set_starting_point(self, starting_point):
        self.starting_point = starting_point

    def set_starting_point_candidates(self, starting_point_candidates):
        self.starting_point_candidates = starting_point_candidates

    def viewer_setup(self):
        from mujoco_py.generated import const
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.fixedcamid = 0


# =========================================================================== #
#                   UR5 With Global Postion Coordinates                   #
# =========================================================================== #

class UR5WithPos(UR5Env):
    pass


class UR5WithPosTest(UR5Env):

    def __init__(self, ):
        super(UR5WithPosTest, self).__init__(xml_file=ABS_PATH + '/xmls/UR5+gripper/UR5gripper_v3_eval.xml')

    def step(self, action):
        next_obs, reward, done, infos = super().step(action)
        end_effector_pos = self.data.get_site_xpos("gripper_endpoint")

        cost = 0
        # collision detection
        for contact in self.data.contact[:self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.body_id2name(self.model.geom_bodyid[g]) for g in geom_ids])
            if any('link' or 'inner' in n for n in geom_names):
                if any('workpiece' in n for n in geom_names):
                    cost = np.zeros_like(reward) + 1
                    # print('collision')

        # Draw the trajectory of the end-effector
        if hasattr(self, 'step_count'):
            self.step_count += 1
            if self.step_count % 5 == 0 and hasattr(self.viewer, '_markers'):
                self.viewer.add_marker(pos=end_effector_pos.copy(),
                                       size=np.ones(3) * 0.008,
                                       mat=np.identity(3),
                                       rgba=[1, 0, 0, 1],
                                       type=const.GEOM_SPHERE)
                # print(self.step_count)

        # print(self.data.qpos)
        infos['cost'] = cost
        return next_obs, reward, done, infos

    #
    # def unwrapped(self):
    #     return self.sim

    # def reset_model(self):
    #     self.init_qpos = np.array([1.9516 - 0.5, 0.000763007, -2.28516, -2.46196, 1.8283])
    #     # [1.40748, -0.594667, -1.8853, -0.008293, 0.135118, 0.00921288, -0.0669425, -0.0668504]
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .03
    #     self.set_state(qpos, qvel)
    #     self.step_count = 0
    #     # if hasattr(self.viewer, '_markers'):
    #     #     del self.viewer._markers[:]
    #     return self._get_obs()


class UR5WithPosTrans(UR5WithPos):
    def __init__(
            self,
            xml_file=ABS_PATH + '/xmls/UR5+gripper/UR5gripper_v3.xml',
            *args,
            **kwargs
    ):
        super(UR5WithPosTrans, self).__init__(xml_file, 5)
        self.target_name = 'target_rightup'


class UR5WithPosTransTest(UR5WithPosTest):
    def __init__(
            self,
            xml_file=ABS_PATH + '/xmls/UR5+gripper/UR5gripper_v3.xml',
            *args,
            **kwargs
    ):
        super(UR5WithPosTransTest, self).__init__(xml_file, 5)
        self.target_name = 'target_rightup'


if __name__=='__main__':
    env = UR5WithPos()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        print("Sampled action:", action)
        observation, reward, done, info = env.step(action)
        print("Observation:", observation)
