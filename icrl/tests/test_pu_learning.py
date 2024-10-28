#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
import os, pickle, tqdm, gym, wandb
import torch as th
import numpy as np
from icrl.run_policy import load_config
from matplotlib.patches import Ellipse
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common.vec_env import VecNormalize
import argparse
from icrl.constraint_net import ConstraintNet, evaluate_constraint_accuracy, plot_for_PointObs, evaluate_policy_accuracy
from icrl.true_constraint_net import get_true_cost_function, null_cost
import icrl.utils as utils
from icrl.ds_policy import DS_Policy
from stable_baselines3.common.utils import set_random_seed

def plot_gmm_ellipse(gmm, ax, **kwargs):
    for means, covariances in zip(gmm.means_, gmm.covariances_):
        vals, vecs = np.linalg.eigh(covariances)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals * gmm.confidence)
        ellipse = Ellipse(xy=means, width=width, height=height, angle=theta, fc='None', lw=3, **kwargs)
        ellipse_handle = ax.add_patch(ellipse)
    return ellipse_handle

def plot_for_pu_learning(cost_function, manual_threshold, obs_dim, acs_dim, save_name, obs_n, obs_e, gmm=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    joint1_lim = [-13, 13]
    joint2_lim = [-13, 13]

    r1 = np.linspace(joint1_lim[0], joint1_lim[1], num=200)
    r2 = np.linspace(joint2_lim[0], joint2_lim[1], num=200)
    X, Y = np.meshgrid(r1, r2)
    obs_all = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
    obs_all = np.concatenate((obs_all, np.zeros((np.size(X), obs_dim-2))), axis=-1)

    action = np.zeros((np.size(X), acs_dim))
    outs = 1 - cost_function(obs_all, action)
    im = ax.imshow(outs.reshape(X.shape), extent=[joint1_lim[0], joint1_lim[1], joint2_lim[0], joint2_lim[1]],
                   cmap='jet_r', vmin=0, vmax=1, origin='lower')
    contour = ax.contour(X, Y, outs.reshape(X.shape), [1-manual_threshold], colors='k', linewidths=2.0)
    plt.clabel(contour, fontsize=15, colors=('k'))

    contour1 = ax.contour(X, Y, outs.reshape(X.shape), [0.5], colors='k', linewidths=2.0, linestyles='--')
    plt.clabel(contour1, fontsize=15, colors=('k'))

    contour2 = ax.contour(X, Y, outs.reshape(X.shape), [0.01], colors='k', linewidths=2.0, linestyles='--' )
    plt.clabel(contour2, fontsize=15, colors=('k'))

    contour3 = ax.contour(X, Y, outs.reshape(X.shape), [0.1], colors='k', linewidths=2.0, linestyles='--')
    plt.clabel(contour3, fontsize=15, colors=('k'))


    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=20)
    ax.scatter(obs_n[..., 0], obs_n[..., 1], clip_on=False, c='r', s=10)
    ax.scatter(obs_e[..., 0], obs_e[..., 1], clip_on=False, c='g', s=10)

    if gmm is not None:
        gmm.confidence = 5
        plot_gmm_ellipse(gmm, ax)

    ax.set_ylim([joint2_lim[0], joint2_lim[1]])
    ax.set_xlim([joint1_lim[0], joint1_lim[1]])
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid('on')
    ax.set_title('PU_learning_PointCircle', fontsize=30)
    fig.savefig(save_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig=fig)

def generate_policy_data(args, env):
    policy_obs = []
    policy_act = []
    policy_len = []
    for load_itr in args.load_itr:
        f = f"models/icrl_{load_itr}_itrs/nominal_agent"
        load_dir = os.path.join(args.load_dir, "files")
        config = load_config(load_dir)
        model_path = os.path.join(load_dir, f)
        model = PPOLagrangian.load(model_path)

        # Check if we want to save using airl scheme
        sampling_func = utils.sample_from_agent
        # Make saving dir
        idx = 0
        while True:
            saving_dict = sampling_func(model, env, 1)
            observations, _, actions, rewards, lengths = saving_dict
            saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
            # print(f"{idx}. Mean reward: {np.mean(saving_dict['rewards'])} | Mean length: {np.mean(saving_dict['lengths'])}")
            policy_obs.append(observations)
            policy_act.append(actions)
            policy_len.append(lengths)
            idx += 1
            if idx == args.n_rollouts:
                break
    return np.vstack(policy_obs), np.vstack(policy_act), np.hstack(policy_len), model.predict

def generate_policy_DS(args, env, expert_obs, expert_lengths):
    A = np.diag([-1, -1])
    linear_ds = lambda x: A @ x
    nominal_agent = DS_Policy(2, linear_ds)
    nominal_obs, _, nominal_acs, nominal_rew, nominal_len = utils.sample_from_same_starting_points_as_demonstrations(
        nominal_agent, env, expert_obs, expert_lengths, 1, deterministic=True)
    return nominal_obs, nominal_acs, nominal_len, nominal_agent

def pu_learning_PointDS():
    args = argparse.Namespace()
    args.env_id = 'PointDSTest-v0'
    args.eval_env_id = 'PointDSTest-v0'
    args.expert_dir = 'icrl/expert_data/PointDS'
    args.load_dir = 'icrl/wandb/run-20240208_112143-74jdxx4w'
    pu_config = {'refine_iterations':1,
                 'rn_decision_method': 'kNN',
                 'kNN_k': 5,
                 'kNN_thresh': 0.22,
                 'CPU_n_gmm': 30,
                 'CPU_ratio_thresh': 0.95,
                 'GPU_n_gmm': 30,
                 'GPU_likelihood_thresh': -5,
                 'add_rn_each_traj': True,
                 'add_boundary_data': False,
                 'angle_diff_thresh': 20
                 }
    pu_config = utils.Dict2Class(pu_config)
    load_dir = os.path.join(args.load_dir, "files")
    config = load_config(load_dir)
    # config.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    args.expert_n_rollouts = 10
    # Main parameters
    config.cn_layers = [16, 16]
    config.cn_learning_rate = 0.005
    config.cn_reg_coeff = 0

    backward_iters = 3000
    backward_outter_iters = 1
    repeat_run = 1
    config.cn_manual_threshold = 0.5
    args.loss_type = 'bce'

    config.device = 'cpu'
    # config.device = 'cuda:0'

    del (config.save_dir)
    wandb.init(project='PU_learning', config=config, dir='./icrl')
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config
    print(utils.colorize('configured folder %s for saving' % config.save_dir, color='green', bold=True))
    print(utils.colorize('name: %s' % config.name, color='green', bold=True))

    # set_random_seed(config.seed)

    # Load model, data
    env = utils.make_eval_env(args.env_id, use_cost_wrapper=False, normalize_obs=False)

    obs_dim = env.observation_space.shape[0]
    acs_dim = env.action_space.shape[0]
    action_low, action_high = None, None
    is_discrete = False
    if isinstance(env.action_space, gym.spaces.Box):
        action_low, action_high = env.action_space.low, env.action_space.high
    cn_lr_schedule = lambda x: config.cn_learning_rate
    (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(args.expert_dir,
                                                                                                   args.expert_n_rollouts)
    for j in range(repeat_run):
        constraint_net = ConstraintNet(obs_dim, acs_dim, config.cn_layers, 256, cn_lr_schedule, is_discrete,
                                       config.cn_reg_coeff, config.cn_obs_select_dim, config.cn_acs_select_dim,
                                       no_importance_sampling=config.no_importance_sampling,
                                       per_step_importance_sampling=config.per_step_importance_sampling,
                                       clip_obs=config.clip_obs,
                                       initial_obs_mean=None if not config.cn_normalize else np.zeros(obs_dim),
                                       initial_obs_var=None if not config.cn_normalize else np.ones(obs_dim),
                                       action_low=action_low, action_high=action_high,
                                       target_kl_old_new=config.cn_target_kl_old_new,
                                       target_kl_new_old=config.cn_target_kl_new_old,
                                       train_gail_lambda=config.train_gail_lambda, eps=config.cn_eps,
                                       device=config.device, loss_type=config.cn_loss_type,
                                       manual_threshold=config.cn_manual_threshold, weight_expert_loss=1,
                                       weight_nominal_loss=1)
        # wandb.watch(constraint_net.network, None, log="all", log_freq=1)
        policy_obs, policy_acs, policy_len, nominal_agent = generate_policy_DS(args, env, expert_obs, expert_lengths)
        columns = ['iter', 'accuracy', 'cn_net']
        training_table = wandb.Table(columns)
        images = []
        print(f'N_data_expert:{expert_obs.shape[0]}, N_data_nominal:{policy_obs.shape[0]}')
        if repeat_run == 1:
            plot_for_pu_learning(constraint_net.cost_function_non_binary, config.cn_manual_threshold, obs_dim, acs_dim,
                                 f'icrl/tests/pu_learning/{j}_{0}', policy_obs, expert_obs)
        tic = time.time()
        for i in range(backward_outter_iters):

            expert_gmm, rn_policy_data, _ = constraint_net.train_with_two_step_pu_learning(backward_iters, expert_obs,
                                                                                           expert_acs, policy_obs,
                                                                                           policy_acs, policy_len,
                                                                                           pu_config)
            plot_for_PointObs(constraint_net.cost_function_non_binary, 0.5, obs_dim, acs_dim,
                              f'icrl/tests/pu_learning/{j}_{i + 1}', f'DSCL iter {j + 1}', policy_obs, expert_obs,
                              rn_policy_data, env.get_attr('gmm_true')[0], expert_gmm)
            eval_metric = evaluate_constraint_accuracy(args.env_id,  constraint_net.cost_function,
                                                       get_true_cost_function(args.eval_env_id), obs_dim,
                                                       acs_dim)

            # nominal_agent.modulate_with_NN(constraint_net.gamma_with_grad, 1)
            # test_obs, _, _, _, test_len = utils.sample_from_demonstrations(nominal_agent,
            #                                                                  env,
            #                                                                  expert_obs,
            #                                                                  expert_lengths,
            #                                                                  1,
            #                                                                  deterministic=True)
            # eval_metric.update(evaluate_policy_accuracy(nominal_agent, expert_obs, expert_acs, expert_lengths, test_obs, test_len))

            images.append(wandb.Image(f'icrl/tests/pu_learning/0_{i + 1}.png', caption=f'Ite {i}'))
            wandb.log(eval_metric)
        toc = time.time()
        print("time used:", toc - tic)
        wandb.log({'images': images})

def pu_learning_PointObstacle():
    args = argparse.Namespace()
    args.env_id = 'PointObstacle-v0'
    args.eval_env_id = 'PointObstacleTest-v0'
    args.expert_dir = 'icrl/expert_data/PointObstacle'

    args.load_dir = 'icrl/wandb/run-20240208_112143-74jdxx4w'
    pu_config = {'refine_iterations':1,
                 'rn_decision_method': 'kNN',
                 'kNN_k':5,
                 'kNN_thresh': 2.6,
                 'CPU_n_gmm': 30,
                 'CPU_ratio_thresh': 0.95,
                 'GPU_n_gmm':5,
                 'GPU_likelihood_thresh': -60,
                 'add_rn_each_traj': False,
                 'add_boundary_data': True,
                 'angle_diff_thresh': 20
                 }
    pu_config = utils.Dict2Class(pu_config)

    load_dir = os.path.join(args.load_dir, "files")
    config = load_config(load_dir)
    # config.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    args.expert_n_rollouts = 30
    # Main parameters
    config.cn_layers = [8, 8]
    config.cn_learning_rate = 0.003
    config.cn_reg_coeff = 0  # 1.1
    args.n_rollouts = 5
    args.load_itr = [1,]  # 1,3,12
    backward_iters = 3000
    backward_outter_iters = 1
    repeat_run = 1
    config.cn_manual_threshold = 0.5
    args.loss_type = 'bce'

    config.device = 'cpu'
    # config.device = 'cuda:0'

    del (config.save_dir)
    wandb.init(project='PU_learning', config=config, dir='./icrl')
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config
    print(utils.colorize('configured folder %s for saving' % config.save_dir, color='green', bold=True))
    print(utils.colorize('name: %s' % config.name, color='green', bold=True))

    # Load model, data
    env = utils.make_eval_env(args.env_id, use_cost_wrapper=False, normalize_obs=False)

    obs_dim = env.observation_space.shape[0]
    acs_dim = env.action_space.shape[0]
    action_low, action_high = None, None
    is_discrete = False
    if isinstance(env.action_space, gym.spaces.Box):
        action_low, action_high = env.action_space.low, env.action_space.high
    cn_lr_schedule = lambda x: config.cn_learning_rate
    (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(args.expert_dir,
                                                                                                   args.expert_n_rollouts)
    for j in range(repeat_run):
        constraint_net = ConstraintNet(obs_dim, acs_dim, config.cn_layers, None, cn_lr_schedule, is_discrete,
                                       config.cn_reg_coeff, config.cn_obs_select_dim, config.cn_acs_select_dim,
                                       no_importance_sampling=config.no_importance_sampling,
                                       per_step_importance_sampling=config.per_step_importance_sampling,
                                       clip_obs=config.clip_obs,
                                       initial_obs_mean=None if not config.cn_normalize else np.zeros(obs_dim),
                                       initial_obs_var=None if not config.cn_normalize else np.ones(obs_dim),
                                       action_low=action_low, action_high=action_high,
                                       target_kl_old_new=config.cn_target_kl_old_new,
                                       target_kl_new_old=config.cn_target_kl_new_old,
                                       train_gail_lambda=config.train_gail_lambda, eps=config.cn_eps,
                                       device=config.device, loss_type=config.cn_loss_type,
                                       manual_threshold=config.cn_manual_threshold, weight_expert_loss=1,
                                       weight_nominal_loss=1)
        wandb.watch(constraint_net.network, None, log="all", log_freq=1)
        policy_obs, policy_acs, policy_len, model = generate_policy_data(args, env)
        columns = ['iter', 'accuracy', 'cn_net']
        training_table = wandb.Table(columns)
        images = []
        print(f'N_data_expert:{expert_obs.shape[0]}, N_data_nominal:{policy_obs.shape[0]}')
        if repeat_run == 1:
            plot_for_pu_learning(constraint_net.cost_function_non_binary, config.cn_manual_threshold, obs_dim, acs_dim,
                                 f'icrl/tests/pu_learning/{j}_{0}', policy_obs, expert_obs)
        tic = time.time()
        for i in range(backward_outter_iters):
            expert_gmm, rn_policy_data, _ = constraint_net.train_with_two_step_pu_learning(backward_iters, expert_obs,
                                                                                           expert_acs, policy_obs,
                                                                                           policy_acs, policy_len,
                                                                                           pu_config)
            plot_for_pu_learning(constraint_net.cost_function_non_binary, config.cn_manual_threshold, obs_dim, acs_dim, f'icrl/tests/pu_learning/{j}_{i + 1}', rn_policy_data, expert_obs, expert_gmm)
            eval_metric = evaluate_constraint_accuracy(args.env_id, constraint_net.cost_function,
                                                       get_true_cost_function(args.eval_env_id), obs_dim,
                                                       acs_dim)

            # training_table.add_data(i, 1, wandb.Image(f'icrl/tests/pu_learning/0_{i}.png', caption=f'Ite {i}'))
            images.append(wandb.Image(f'icrl/tests/pu_learning/0_{i + 1}.png', caption=f'Ite {i}'))
            wandb.log({'accuracy': eval_metric['true/jaccard']})
        toc = time.time()
        print("time used:", toc - tic)
        wandb.log({'images': images})
        # wandb.log({'Training': training_table})


def pu_learning_Panda():
    args = argparse.Namespace()
    args.env_id = 'PointObstacle-v0'
    args.eval_env_id = 'PointObstacleTest-v0'
    args.expert_dir = 'icrl/expert_data/PointObstacle'

    args.load_dir = 'icrl/wandb/run-20240208_112143-74jdxx4w'
    pu_config = {'refine_iterations':1,
                 'rn_decision_method': 'kNN',
                 'kNN_k':5,
                 'kNN_thresh': 2.6,
                 'CPU_n_gmm': 30,
                 'CPU_ratio_thresh': 0.95,
                 'GPU_n_gmm':5,
                 'GPU_likelihood_thresh': -60,
                 'add_rn_each_traj': False,
                 'add_boundary_data': True,
                 'angle_diff_thresh': 20
                 }
    pu_config = utils.Dict2Class(pu_config)

    load_dir = os.path.join(args.load_dir, "files")
    config = load_config(load_dir)
    # config.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    args.expert_n_rollouts = 30
    # Main parameters
    config.cn_layers = [8, 8]
    config.cn_learning_rate = 0.003
    config.cn_reg_coeff = 0  # 1.1
    args.n_rollouts = 5
    args.load_itr = [1,]  # 1,3,12
    backward_iters = 3000
    backward_outter_iters = 1
    repeat_run = 1
    config.cn_manual_threshold = 0.5
    args.loss_type = 'bce'

    config.device = 'cpu'
    # config.device = 'cuda:0'

    del (config.save_dir)
    wandb.init(project='PU_learning', config=config, dir='./icrl')
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config
    print(utils.colorize('configured folder %s for saving' % config.save_dir, color='green', bold=True))
    print(utils.colorize('name: %s' % config.name, color='green', bold=True))

    # Load model, data
    env = utils.make_eval_env(args.env_id, use_cost_wrapper=False, normalize_obs=False)

    obs_dim = env.observation_space.shape[0]
    acs_dim = env.action_space.shape[0]
    action_low, action_high = None, None
    is_discrete = False
    if isinstance(env.action_space, gym.spaces.Box):
        action_low, action_high = env.action_space.low, env.action_space.high
    cn_lr_schedule = lambda x: config.cn_learning_rate
    (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(args.expert_dir,
                                                                                                   args.expert_n_rollouts)
    for j in range(repeat_run):
        constraint_net = ConstraintNet(obs_dim, acs_dim, config.cn_layers, None, cn_lr_schedule, is_discrete,
                                       config.cn_reg_coeff, config.cn_obs_select_dim, config.cn_acs_select_dim,
                                       no_importance_sampling=config.no_importance_sampling,
                                       per_step_importance_sampling=config.per_step_importance_sampling,
                                       clip_obs=config.clip_obs,
                                       initial_obs_mean=None if not config.cn_normalize else np.zeros(obs_dim),
                                       initial_obs_var=None if not config.cn_normalize else np.ones(obs_dim),
                                       action_low=action_low, action_high=action_high,
                                       target_kl_old_new=config.cn_target_kl_old_new,
                                       target_kl_new_old=config.cn_target_kl_new_old,
                                       train_gail_lambda=config.train_gail_lambda, eps=config.cn_eps,
                                       device=config.device, loss_type=config.cn_loss_type,
                                       manual_threshold=config.cn_manual_threshold, weight_expert_loss=1,
                                       weight_nominal_loss=1)
        wandb.watch(constraint_net.network, None, log="all", log_freq=1)
        policy_obs, policy_acs, policy_len, model = generate_policy_data(args, env)
        columns = ['iter', 'accuracy', 'cn_net']
        training_table = wandb.Table(columns)
        images = []
        print(f'N_data_expert:{expert_obs.shape[0]}, N_data_nominal:{policy_obs.shape[0]}')
        if repeat_run == 1:
            plot_for_pu_learning(constraint_net.cost_function_non_binary, config.cn_manual_threshold, obs_dim, acs_dim,
                                 f'icrl/tests/pu_learning/{j}_{0}', policy_obs, expert_obs)
        tic = time.time()
        for i in range(backward_outter_iters):
            expert_gmm, rn_policy_data, _ = constraint_net.train_with_two_step_pu_learning(backward_iters, expert_obs,
                                                                                           expert_acs, policy_obs,
                                                                                           policy_acs, policy_len,
                                                                                           pu_config)
            plot_for_pu_learning(constraint_net.cost_function_non_binary, config.cn_manual_threshold, obs_dim, acs_dim, f'icrl/tests/pu_learning/{j}_{i + 1}', rn_policy_data, expert_obs, expert_gmm)
            eval_metric = evaluate_constraint_accuracy(args.env_id, constraint_net.cost_function,
                                                       get_true_cost_function(args.eval_env_id), obs_dim,
                                                       acs_dim)

            # training_table.add_data(i, 1, wandb.Image(f'icrl/tests/pu_learning/0_{i}.png', caption=f'Ite {i}'))
            images.append(wandb.Image(f'icrl/tests/pu_learning/0_{i + 1}.png', caption=f'Ite {i}'))
            wandb.log({'accuracy': eval_metric['true/jaccard']})
        toc = time.time()
        print("time used:", toc - tic)
        wandb.log({'images': images})
        # wandb.log({'Training': training_table})


def main():
    pu_learning_PointDS()

if __name__=='__main__':
    main()

