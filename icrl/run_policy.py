"""Load and run policy"""

import argparse
import os
import shutil
from icrl.constraint_net import ConstraintNet, plot_for_UR5_envs_3D, plot_constraints
import numpy as np
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common.vec_env import VecNormalize
from icrl.true_constraint_net import get_true_cost_function
import icrl.utils as utils
import wandb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_config(d):
    config = utils.load_dict_from_json(d, "config")
    config = utils.dict_to_namespace(config)
    return config

def run_policy(args):
    # Find which file to load
    if args.is_icrl:
        if args.load_itr is not None:
            f = f"models/icrl_{args.load_itr}_itrs/nominal_agent"
        else:
            f = "best_nominal_model"
    else:
        if args.load_itr is not None:
            f = f"models/rl_model_{args.load_itr}_steps"
        else:
            f = "best_model"

    # Configure paths (restore from W&B server if needed)
    if args.remote:
        # Save everything in wandb/remote/<run_id>
        load_dir = os.path.join("icrl/wandb/remote/", args.load_dir.split('/')[-1])
        utils.del_and_make(load_dir)
        # Restore form W&B
        wandb.init(dir=load_dir)
        USER = None
        run_path = os.path.join(USER, args.load_dir)
        wandb.restore("config.json", run_path=run_path, root=load_dir)
        config = load_config(load_dir)
        if not config.dont_normalize_obs:
            wandb.restore("train_env_stats.pkl", run_path=run_path, root=load_dir)
        wandb.restore(f+".zip", run_path=run_path, root=load_dir)
    else:
        load_dir = os.path.join(args.load_dir, "files")
        config = load_config(load_dir)

    save_dir = os.path.join(load_dir, args.save_dir)
    utils.del_and_make(save_dir)
    model_path = os.path.join(load_dir, f)

    # Load model
    model = PPOLagrangian.load(model_path)

    # Create env, model
    def make_env():
        env_id = args.env_id or config.eval_env_id
        if "Test" in env_id and args.use_training_env:
            env_id = env_id.replace("Test", "")
        # env_id = 'pandavis-v0'
        env = utils.make_eval_env(env_id, use_cost_wrapper=False, normalize_obs=False)
        true_cost_function = get_true_cost_function(env_id)
        # Restore environment stats
        if not config.dont_normalize_obs:
            env = VecNormalize.load(os.path.join(load_dir, f"models/icrl_{args.load_itr}_itrs/{args.load_itr}_train_env_stats.pkl"), env)
            env.norm_reward = False
            env.training = False
        return env, true_cost_function

    # Evaluate and make video
    if not args.dont_make_video:
        env, _ = make_env()
        utils.eval_and_make_video(env, model, save_dir, "video1", args.n_rollouts)

    sampling_func = utils.sample_from_agent
    # Save trajectories
    if not args.dont_save_trajs:
        env, true_cost_function = make_env()
        # Make saving dir
        rollouts_dir = os.path.join(save_dir, "rollouts")
        utils.del_and_make(rollouts_dir)

        obs_all, len_all, rew_all, acs_all = [], [], [], []

        if args.same_starting_as_expert:
            (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(args.expert_path, args.expert_rollouts)
            start_indices = np.cumsum(np.insert(expert_lengths[:-1], 0, 0))
            starting_points = expert_obs[start_indices]
            end_indices = np.cumsum(expert_lengths) - 1
            ending_points = expert_obs[end_indices]
            env.env_method('set_starting_point', starting_points, ending_points)

        idx = 0
        while True:
            saving_dict = sampling_func(model, env, 1, deterministic=True)
            observations, _, actions, rewards, lengths = saving_dict
            saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
            obs_all.append(observations)
            len_all.append(int(lengths))
            rew_all.append(rewards)
            acs_all.append(actions)
            saving_dict['save_scheme'] = 'not_airl'

            if (args.reward_threshold is None or np.mean(saving_dict['rewards']) >= args.reward_threshold) and\
               (args.length_threshold is None or np.mean(saving_dict['lengths']) <= args.length_threshold):
                print(f"{idx}. Mean reward: {np.mean(saving_dict['rewards'])} | Mean length: {np.mean(saving_dict['lengths'])}")
                utils.save_dict_as_pkl(saving_dict,
                                       rollouts_dir, str(idx))
                idx += 1
                if idx == args.n_rollouts:
                    break
            else:
                print('One episode not saved due to reward/length threshold')

        fc = f"models/icrl_{args.load_itr}_itrs/cn.pt"
        cn_path = os.path.join(load_dir, fc)
        constraint_net = ConstraintNet.load(cn_path)

        plot_constraints(constraint_net.cost_function_non_binary, 0.5, 13, None, config.eval_env_id,
                         constraint_net.select_dim, observations.shape[1], actions.shape[1],
                         os.path.join(save_dir, "%d.png" % args.load_itr),
                         'Constraint Function of Iteration ' + str(args.load_itr), obs_expert=np.vstack(obs_all),
                         rew_expert=np.vstack(rew_all), len_expert=len_all, obs_nominal=np.vstack(obs_all), obs_rn=None, show=True)

        if args.generate_evaluation_states:
            # Generate evaluation states from expert data and policy data that are later used to evaluate the performance of constraint accuracy
            (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(
                args.expert_path, args.expert_rollouts)
            nominal_obs = np.vstack(obs_all)
            nominal_acs = np.vstack(acs_all)
            evaluation_obs = np.vstack((expert_obs, nominal_obs))
            evaluation_acs = np.vstack((expert_acs, nominal_acs))
            evaluation_true_cost = true_cost_function(evaluation_obs, evaluation_acs).reshape(-1, 1)

            evaluation_states = np.hstack((evaluation_obs, evaluation_acs, evaluation_true_cost))
            np.save(args.expert_path + '/files/evaluation_states.npy', evaluation_states)
            print('Generate and save evaluation states successfully!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_to_run", type=str)
    parser.add_argument("--load_dir", "-l", type=str, default="icrl/wandb/latest-run/")
    parser.add_argument("--is_icrl", "-ii", action='store_true')
    parser.add_argument("--remote", "-r", action="store_true")
    parser.add_argument("--save_dir", "-s", type=str, default="run_policy")
    parser.add_argument("--env_id", "-e", type=str, default=None)
    parser.add_argument("--load_itr", "-li", type=int, default=None)
    parser.add_argument("--n_rollouts", "-nr", type=int, default=3)
    parser.add_argument("--dont_make_video", "-dmv", action="store_true")
    parser.add_argument("--dont_save_trajs", "-dst", action="store_true")
    parser.add_argument("--use_training_env", "-ute", action="store_true")
    parser.add_argument("--same_starting_as_expert", "-ssae", action="store_true")
    parser.add_argument("--reward_threshold", "-rt", type=float, default=None)
    parser.add_argument("--length_threshold", "-lt", type=int, default=None)
    parser.add_argument("--generate_evaluation_states", "-ges", action="store_true", help='Generate evaluation states for constraint accuracy evaluation')
    parser.add_argument('--expert_path', '-ep', type=str, default='icrl/expert_data/HCWithPos-New')
    parser.add_argument('--expert_rollouts', '-er', type=int, default=30)   # max
    args = parser.parse_args()

    run_policy(args)
