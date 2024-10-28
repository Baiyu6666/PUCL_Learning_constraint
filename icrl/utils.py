import collections
import json
import math
import os
import pickle
import shutil
import subprocess
import types
import gym, time
import numpy as np
from numpy.ma.extras import average

import stable_baselines3.common.callbacks as callbacks
import stable_baselines3.common.vec_env as vec_env
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import safe_mean, set_random_seed
from stable_baselines3.common.vec_env import (VecCostWrapper,
                                              VecNormalizeWithCost)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from tqdm import tqdm
import matplotlib.pyplot as plt

#==============================================================================
# Functions to handle parser.
#==============================================================================

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def concat_nondefault_arguments(parser, ignore_keys=[], path_keys=[],
                                default_config=None, actual_config=None):
    """
    Given an instance of argparse.ArgumentParser, return a concatenation
    of the names and values of only those arguments that do not have the
    default value (i.e. alternative values were specified via command line).

    So if you run

        python file.py -abc 123 -def 456

    this function will return abc_123_def_456. (In case the value passed for
    'abc' or 'def' is the same as the default value, they will be ignored)

    If a shorter version of the argument name is specified, it will be
    preferred over the longer version.

    Arguments are sorted alphabetically.

    If you want this function to ignore some arguments, you can pass them
    as a list to the ignore_keys argument.

    If some arguments expect paths, you can pass in those as a list to the
    path_keys argument. The values of these will be split at '/' and only the
    last substring will be used.

    If the default config dictionary is specified then the default values in it
    are preferred ovr the default values in parser.

    If the actual_config dictionary is specified then the values in it are preferred
    over the values passed through command line.
    """
    sl_map = get_sl_map(parser)

    def get_default(key):
        if default_config is not None and key in default_config:
            return default_config[key]
        return parser.get_default(key)

    # Determine save dir based on non-default arguments if no
    # save_dir is provided.
    concat = ''
    for key, value in sorted(vars(parser.parse_args()).items()):
        if actual_config is not None:
            value = actual_config[key]

        # Skip these arguments.
        if key in ignore_keys:
            continue

        if type(value) == list:
            b = False
            if get_default(key) is None or len(value) != len(get_default(key)):
                b = True
            else:
                for v, p in zip(value, get_default(key)):
                    if v != p:
                        b = True
                        break
            if b:
                concat += '%s_' % sl_map[key]
                for v in value:
                    if type(v) not in [bool, int] and hasattr(v, "__float__"):
                        if v == 0:
                            valstr = 0
                        else:
                            valstr = round(v, 4-int(math.floor(math.log10(abs(v))))-1)
                    else: valstr = v
                    concat += '%s_' % str(valstr)

        # Add key, value to concat.
        elif value != get_default(key):
            # For paths.
            if value is not None and key in path_keys:
                value = value.split('/')[-1]

            if type(value) not in [bool, int] and hasattr(value, "__float__"):
                if value == 0:
                    valstr = 0
                else:
                    valstr = round(value, 4-int(math.floor(math.log10(abs(value))))-1)
            else: valstr = value
            concat += '%s_%s_' % (sl_map[key], valstr)

    if len(concat) > 0:
        # Remove extra underscore at the end.
        concat = concat[:-1]

    return concat

def get_sl_map(parser):
    """Return a dictionary containing short-long name mapping in parser."""
    sl_map = {}

    # Add arguments with long names defined.
    for key in parser._option_string_actions.keys():
        if key[1] == '-':
            options = parser._option_string_actions[key].option_strings
            if len(options) == 1:   # No short argument.
                sl_map[key[2:]] = key[2:]
            else:
                if options[0][1] == '-':
                    sl_map[key[2:]] = options[1][1:]
                else:
                    sl_map[key[2:]] = options[0][1:]

    # We've now processed all arguments with long names. Now need to process
    # those with only short names specified.
    known_keys = list(sl_map.keys()) + list(sl_map.values())
    for key in parser._option_string_actions.keys():
        if key[1:] not in known_keys and key[2:] not in known_keys:
            sl_map[key[1:]] = key[1:]

    return sl_map

def reverse_dict(x):
    """
    Exchanges keys and values in x i.e. x[k] = v ---> x[v] = k.
    Added Because reversed(x) does not work in python 3.7.
    """
    y = {}
    for k,v in x.items():
        y[v] = k
    return y

def merge_configs(config, parser, sys_argv):
    """
    Merge a dictionary (config) and arguments in parser. Order of priority:
    argument supplied through command line > specified in config > default
    values in parser.
    """

    parser_dict = vars(parser.parse_args())
    config_keys = list(config.keys())
    parser_keys = list(parser_dict.keys())

    sl_map = get_sl_map(parser)
    rev_sl_map = reverse_dict(sl_map)
    def other_name(key):
        if key in sl_map:
            return sl_map[key]
        elif key in rev_sl_map:
            return rev_sl_map[key]
        else:
            return key

    merged_config = {}
    for key in config_keys + parser_keys:
        if key in parser_keys:
            # Was argument supplied through command line?
            if key_was_specified(key, other_name(key), sys_argv):
                merged_config[key] = parser_dict[key]
            else:
                # If key is in config, then use value from there.
                if key in config:
                    merged_config[key] = config[key]
                else:
                    merged_config[key] = parser_dict[key]
        elif key in config:
            # If key was only specified in config, use value from there.
            merged_config[key] = config[key]

    return merged_config

def key_was_specified(key1, key2, sys_argv):
    for arg in sys_argv:
        if arg[0] == '-' and (key1 == arg.strip('-') or key2 == arg.strip('-')):
            return True
    return False

def get_name(parser, default_config, actual_config, mod_name):
    """Returns a name for the experiment based on parameters passed."""
    prefix = lambda x, y: x + '_'*(len(y)>0) + y

    name = actual_config["name"]
    if name is None:
        name = concat_nondefault_arguments(
                parser,
                ignore_keys=["config_file", "train_env_id", "eval_env_id", "seed",
                             "timesteps", "save_every", "e)val_every", "n_iters",
                             "sync_wandb", "file_to_run", "project", "group", 'dont_normalize_reward'
                             ,'dont_normalize_cost', 'cn_obs_select_dim', 'dont_normalize_obs',
                             'policy_layers', 'reward_vf_layers', 'loading_policy_dir', 'loading_policy',
                             'cn_layers', 'expert_path', 'debug_mode'],
                path_keys=["expert_path"],
                default_config=default_config,
                actual_config=actual_config
        )
        if len(mod_name) > 0:
            name = prefix(mod_name.split('.')[-1], name)

        name = prefix(actual_config["eval_env_id"], name)
        name = prefix(actual_config["train_env_id"], name)

    # Append seed and system id regardless of whether the name was passed in
    # or not
    if "wandb_sweep" in actual_config and not actual_config["wandb_sweep"]:
        sid = get_sid()
    else:
        sid = "-1"
    name = name + "_s_" + str(actual_config["seed"]) + "_sid_" + sid

    return name

# =============================================================================
# Gym utilities
# =============================================================================

def make_env(env_id, rank, log_dir, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        if 'ReachVel' in env_id:
            env = Monitor(env, log_dir, info_keywords=('is_success',), track_keywords=('cost', 'dis_reward', 'ctl_reward', 'action_norm_inf'))
        elif 'Reach' in env_id:
            env = Monitor(env, log_dir, info_keywords=('is_success',),
                          track_keywords=('cost', 'dis_reward', 'ctl_reward'))
        else:
            env = Monitor(env, log_dir)

        return env
    set_random_seed(seed)
    return _init

def make_train_env(env_id, save_dir, use_cost_wrapper, base_seed=0, num_threads=1,
                   normalize_obs=True, normalize_reward=True, normalize_cost=True,
                   **kwargs):
    env = [make_env(env_id, i, save_dir, base_seed)
           for i in range(num_threads)]
    env = vec_env.SubprocVecEnv(env)
    if use_cost_wrapper:
        env = vec_env.VecCostWrapper(env)
    if normalize_reward and normalize_cost:
        assert(all(key in kwargs for key in ['cost_info_str','reward_gamma','cost_gamma']))
        env = vec_env.VecNormalizeWithCost(
                env, training=True, norm_obs=normalize_obs, norm_reward=normalize_reward,
                norm_cost=normalize_cost, cost_info_str=kwargs['cost_info_str'],
                reward_gamma=kwargs['reward_gamma'], cost_gamma=kwargs['cost_gamma'])
    elif normalize_reward:
        assert(all(key in kwargs for key in ['reward_gamma']))
        env = vec_env.VecNormalizeWithCost(
                env, training=True, norm_obs=normalize_obs, norm_reward=normalize_reward,
                norm_cost=normalize_cost, reward_gamma=kwargs['reward_gamma'])
    else:
        env = vec_env.VecNormalizeWithCost(
                env, training=True, norm_obs=normalize_obs, norm_reward=normalize_reward,
                norm_cost=normalize_cost)
    return env

def make_eval_env(env_id, use_cost_wrapper, normalize_obs=True):
    env = [lambda: gym.make(env_id)]
    env = vec_env.SubprocVecEnv(env)
    if use_cost_wrapper:
        env = vec_env.VecCostWrapper(env)
    print("Wrapping eval env in a VecNormalize.")
    env = vec_env.VecNormalizeWithCost(env, training=False, norm_obs=normalize_obs,
                                       norm_reward=False, norm_cost=False)

    if is_image_space(env.observation_space) and not isinstance(env, vec_env.VecTransposeImage):
        print("Wrapping eval env in a VecTransposeImage.")
        env = vec_env.VecTransposeImage(env)

    return env

def eval_and_make_video(env, model, folder, name_prefix, n_rollouts=3, deterministic=False):
    """This will also close the environment"""
    video_length = int(n_rollouts * env.get_attr('spec')[0].max_episode_steps)

    # Make a video
    # Record the video starting at the first step
    env = vec_env.VecVideoRecorder(env, folder,
                                   record_video_trigger=lambda x: x == 0,
                                   video_length=video_length,
                                   name_prefix=name_prefix)
    env.metadata["video.frames_per_second"] = 30
    env.env.metadata["video.frames_per_second"] = 30
    mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_rollouts, deterministic=deterministic
    )
    print("Mean reward: %f +/- %f." % (mean_reward, std_reward))

    # Save the video
    env.close()

def sample_from_agent(agent, env, rollouts, deterministic=False, starting_point=None, return_feasibility=False):
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
    orig_observations, observations, actions = [], [], []
    rewards, lengths, feasibles = [], [], []
    for i in range(rollouts):
        if starting_point is not None:
            env.env_method('set_starting_point', starting_point)
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, vec_env.VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_feasibility = 1

        if isinstance(env, vec_env.VecNormalize):
            orig_observations.append(env.get_original_obs())
        else:
            orig_observations.append(obs)
        episode_length += 1

        while not done:
            action, state = agent.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            if return_feasibility:
                episode_feasibility *= 1 - _info[0]['cost']
            observations.append(obs)
            if not done:
                episode_length += 1
                if isinstance(env, vec_env.VecNormalize):
                    orig_observations.append(env.get_original_obs())
                else:
                    orig_observations.append(obs)
            # else:
            #     orig_observations.append(_info[0]['terminal_observation'][np.newaxis,:])
            actions.append(action)
            episode_reward += reward

        rewards.append(episode_reward)
        lengths.append(episode_length)
        feasibles.append(episode_feasibility)
    orig_observations = np.squeeze(np.array(orig_observations), axis=1)
    observations = np.squeeze(np.array(observations), axis=1)
    actions = np.squeeze(np.array(actions), axis=1)
    rewards = np.squeeze(np.array(rewards), axis=1)
    lengths = np.array(lengths)
    feasibles = np.array(feasibles)
    if starting_point is not None:
        env.env_method('set_starting_point', None)
    if return_feasibility:
        return orig_observations, observations, actions, rewards, lengths, feasibles
    else:
        return orig_observations, observations, actions, rewards, lengths

def sample_trajectories_for_reach_env(agent, env, rollouts, cost_function, target, deterministic=False, save_dir=None, save_namegroup=None):
    # This function is only used for Reach env and sample trajectories to reach the target.
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    orig_observations, observations, actions = [], [], []
    rewards, lengths, costs = [], [], []
    # Generating trajectories from starting points of all the demonstrations
    i = 0
    while i < rollouts:
        if not isinstance(env, vec_env.VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_length = 0
        current_episode_reward = np.array([0.])
        current_episode_cost = np.array([0.])
        current_episode_obs = []
        current_episode_acs = []

        if isinstance(env, vec_env.VecNormalize):
            current_episode_obs.append(env.get_original_obs())
        else:
            current_episode_obs.append(obs)
        episode_length += 1

        while not done:
            action, state = agent.predict(obs, state=state, deterministic=deterministic)
            current_episode_cost += cost_function(env.get_original_obs(), action)
            obs, reward, done, _info = env.step(action)
            if not done:
                if isinstance(env, vec_env.VecNormalize):
                    current_episode_obs.append(env.get_original_obs())
                else:
                    current_episode_obs.append(obs)
                episode_length += 1
            # else:
            #     current_episode_obs.append(_info[0]['terminal_observation'][np.newaxis, :])
            current_episode_acs.append(action)
            current_episode_reward += reward
        # print('policy: %01.2f , expert: %01.2f' %(episode_reward, expert_reward[j]))

        # Check if the trajectory is safe and reach the transferred target.
        average_cost = current_episode_cost/episode_length
        if  (average_cost < 0.07 and average_cost > 0 and
                np.linalg.norm(target - current_episode_obs[-2][0, :3]) <= 0.001): # for all training point+ panda, 0.03, may be problemabtica
            rewards.append(current_episode_reward)
            lengths.append(episode_length)
            orig_observations += current_episode_obs
            actions += current_episode_acs

            i += 1
            if save_dir is not None:
                saving_dict = dict(observations=np.concatenate(current_episode_obs), actions=np.concatenate(current_episode_acs)
                                   ,rewards=current_episode_reward, lengths=episode_length)
                saving_dict['save_scheme'] = 'not_airl'
                if save_namegroup is not None:
                    save_name = save_namegroup + '_' + str(i)
                else:
                    save_name = str(i)
                save_dict_as_pkl(saving_dict, save_dir, save_name)

            print('Save %i-th trajectory with nominal reward: %0.3f; average cost %0.3f' % (i, current_episode_reward[0], current_episode_cost[0]/episode_length))
        observations += current_episode_obs


    orig_observations = np.vstack(orig_observations) if orig_observations != [] else np.empty((0, obs.shape[1]))
    observations = np.vstack(observations)
    actions = np.vstack(actions) if actions != [] else np.empty((0, action.shape[1]))
    rewards = np.concatenate(rewards) if rewards != [] else np.empty((0))
    lengths = np.array(lengths) if lengths != [] else np.empty((0))
    env.env_method('set_starting_point', None)
    # Note here we only return orig obs and org reward.
    return orig_observations, observations, actions, rewards, lengths


def sample_from_same_starting_points_as_demonstrations(agent, env, expert_obs, expert_len, rollouts_per_demonstration=1, deterministic=False, policy_excerpts=False,
                               expert_reward=None, cost_function=None, policy_excerpts_weight=1.):
    # This function samples trajectories from the same starting points as in the demonstrations.
    print('------------------------------------------------------------------------------')
    print('Generating trajectories from the same starting points as in the demonstrations.')
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
    assert rollouts_per_demonstration == 1, 'havent check rollouts >1'

    orig_observations, observations, actions = [], [], []
    rewards, lengths, costs = [], [], []
    # Generating trajectories from starting points of all the demonstrations
    idx_of_starting_point = 0
    for j in range(expert_len.shape[0]):
        current_expert_len = expert_len[j]
        starting_point = expert_obs[idx_of_starting_point]
        for i in range(rollouts_per_demonstration):
            # Reset to specified starting point
            env.env_method('set_starting_point', starting_point)
            if not isinstance(env, vec_env.VecEnv) or i == 0:
                obs = env.reset()
            done, state = False, None
            episode_length = 0
            current_episode_reward = np.array([0.])
            current_episode_cost = np.array([0.])
            current_episode_obs = []
            current_episode_acs = []

            if isinstance(env, vec_env.VecNormalize):
                current_episode_obs.append(env.get_original_obs())
            else:
                current_episode_obs.append(obs)
            episode_length += 1

            while not done:
                action, state = agent.predict(obs, state=state, deterministic=deterministic)
                current_episode_cost += cost_function(env.get_original_obs(), action)
                obs, reward, done, _info = env.step(action)
                if not done:
                    if isinstance(env, vec_env.VecNormalize):
                        current_episode_obs.append(env.get_original_obs())
                    else:
                        current_episode_obs.append(obs)
                    episode_length += 1
                # else:
                #     current_episode_obs.append(_info[0]['terminal_observation'][np.newaxis, :])
                current_episode_acs.append(action)
                current_episode_reward += reward
            # print('policy: %01.2f , expert: %01.2f' %(episode_reward, expert_reward[j]))

            # If policy excerpt is on, save trajectories with a reward higher than the expert. If policy excerpt is off, save all trajectories.
            if policy_excerpts == False or (current_episode_reward * policy_excerpts_weight > expert_reward[
                j] and current_episode_cost / episode_length < 0.5):
                rewards.append(current_episode_reward)
                lengths.append(episode_length)
                orig_observations += current_episode_obs
                actions += current_episode_acs
                if expert_reward is not None:
                    print('%i-th starting point: expert reward: %0.3f ; policy reward: %0.3f; policy average cost %0.3f ;'  %
                                  (j, expert_reward[j], current_episode_reward[0],
                                   current_episode_cost[0] / episode_length), 'Valid unlabeled trajectory saved.')
            else:
                # Discard bad trajectories
                print('%i-th starting point: expert reward: %0.3f ; policy reward: %0.3f; policy average cost %0.3f ;' %
                      (j, expert_reward[j], current_episode_reward[0],
                       current_episode_cost[0] / episode_length), 'Invalid unlabeled trajectory.')
            observations += current_episode_obs

        idx_of_starting_point += current_expert_len
    print('------------------------------------------------------------------------------')

    orig_observations = np.vstack(orig_observations) if orig_observations != [] else np.empty((0, obs.shape[1]))
    observations = np.vstack(observations)
    actions = np.vstack(actions) if actions != [] else np.empty((0, action.shape[1]))
    rewards = np.concatenate(rewards) if rewards != [] else np.empty((0))
    lengths = np.array(lengths) if lengths != [] else np.empty((0))
    env.env_method('set_starting_point', None)
    # Note here we only return orig obs and org reward.
    return orig_observations, observations, actions, rewards, lengths


def sample_true_labeled_data(cost_function, env_id):
    # This function is only for ReachConcave env or Reach2regions env, it generates true labeled data from true env.
    if 'Reach' not in env_id:
        raise NotImplementedError('Warning! This function is only for Reach env')
    obs_dim = 9
    acs_dim = 3
    # for concave env
    if 'ReachConcave' in env_id:
        x_lim = (0.42, 0.7)
        y_lim = (-0.1, 0.18)
        z_lim = (0.02, 0.4)
    elif 'Reach2regions' in env_id:
        x_lim = (0.42, 0.6)
        y_lim = (-0.15, 0.18)
        z_lim = (0.02, 0.4)

    num_points = 50
    r1 = np.linspace(x_lim[0], x_lim[1], num=num_points)
    r2 = np.linspace(y_lim[0], y_lim[1], num=num_points)
    r3 = np.linspace(z_lim[0], z_lim[1], num=num_points)
    X, Y, Z = np.meshgrid(r1, r2, r3)

    obs = np.concatenate(
        [X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1]), np.zeros((X.size, obs_dim - 3))], axis=-1)
    action = np.zeros((np.size(X), acs_dim))
    outs = (1 - cost_function(obs, action)).reshape(X.shape)
    feasible_obs = obs[outs.flatten() > 0.5]
    infeasible_obs = obs[outs.flatten() <= 0.5]
    feasible_acs = action[outs.flatten() > 0.5]
    infeasible_acs = action[outs.flatten() <= 0.5]
    return feasible_obs, feasible_acs, infeasible_obs, infeasible_acs

# =============================================================================
# Policy utilities
# =============================================================================

def compute_kl(agent_2, observations, actions, agent_1=None):
    """Compute KL(agent_1 || agent_2). Observatiosn and actions must have been sampled
    from agent 1. If agent_1 is None, then all observations and actions are assumed to
    have equal probability. agent_1 and agent_2 are functions that compute (return)
    p(actions|observations).
    """
    observations = th.tensor(observations, dtype=th.float32)
    actions = th.tensor(actions, dtype=th.float32)
    log_prob = lambda agent: agent.policy.evaluate_actions(observations, actions)[1]

    kl = -log_prob(agent_2)
    if agent_1 is not None:
        kl += log_prob(agent_1)

    kl = kl.sum()/observations.shape[0]

    return kl.item()

# =============================================================================
# Various
# =============================================================================
def load_expert_data(expert_path, num_rollouts, load_type='EXPERT'):
    rollouts_dir = os.path.join(expert_path, 'files/' + load_type + '/rollouts')
    all_files = os.listdir(rollouts_dir)

    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    pkl_files = sorted(pkl_files, key=lambda x: int(x.split('.')[0]))
    if num_rollouts > len(pkl_files):
        print("Warning! Requested number of rollouts exceeds the available files. Using all available files.")
        num_rollouts = len(pkl_files)

    # Use the first num_rollouts data
    # selected_indices = np.arange(num_rollouts)
    # selected_indices = np.array([1,2,4,6,7,12,13,14,18,21,24,25,26,27])
    # selected_indices = np.array([1,2, 3])

    # Use randomly selected num_rollouts data
    set_random_seed(int(time.time()))
    selected_indices = np.random.choice(len(pkl_files), num_rollouts, replace=False)

    selected_files = [pkl_files[idx] for idx in selected_indices]
    expert_length = []
    for idx, file_name in enumerate(selected_files):
        with open(os.path.join(rollouts_dir, file_name), 'rb') as f:
            data = pickle.load(f)
        if idx == 0:
            expert_obs = data['observations']
            expert_acs = data['actions']
            expert_rew = data['rewards']
        else:
            expert_obs = np.concatenate([expert_obs, data['observations']], axis=0)
            expert_acs = np.concatenate([expert_acs, data['actions']], axis=0)
            expert_rew = np.concatenate([expert_rew, data['rewards']], axis=0)
        expert_length.append(int(data['lengths']))
    expert_mean_reward = np.mean(expert_rew)
    expert_length = np.array(expert_length)
    return (expert_obs, expert_acs, expert_rew), expert_length, expert_mean_reward

def load_expert_data_and_plot(expert_path, num_rollouts, load_type='EXPERT'):
    import matplotlib
    matplotlib.use('TkAgg')

    rollouts_dir = os.path.join(expert_path, 'files/' + load_type + '/rollouts')
    all_files = os.listdir(rollouts_dir)

    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    pkl_files = sorted(pkl_files, key=lambda x: int(x.split('.')[0]))
    if num_rollouts > len(pkl_files):
        print("Warning! Requested number of rollouts exceeds the available files. Using all available files.")
        num_rollouts = len(pkl_files)

    selected_indices = np.arange(num_rollouts)
    selected_files = [pkl_files[idx] for idx in selected_indices]
    expert_length = []


    # # 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # for idx, file_name in enumerate(selected_files):
    #     with open(os.path.join(rollouts_dir, file_name), 'rb') as f:
    #         data = pickle.load(f)
    #
    #     expert_length.append(int(data['lengths']))
    #     expert_obs = data['observations'][:expert_length[-1]]
    #
    #     # Plot the expert_obs
    #     ax.plot(expert_obs[:, 0], expert_obs[:, 1], expert_obs[:, 2], label=f'Trajectory {idx}')
    #
    #     ax.text(expert_obs[0, 0], expert_obs[0, 1], expert_obs[0, 2], f'{file_name}', color='red')
    #     if len(expert_obs) > 5:
    #         ax.text(expert_obs[5, 0], expert_obs[5, 1], expert_obs[5, 2], f'{file_name}', color='blue')
    # plt.show()

    # 2D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for idx, file_name in enumerate(selected_files):
        with open(os.path.join(rollouts_dir, file_name), 'rb') as f:
            data = pickle.load(f)

        expert_length.append(int(data['lengths']))
        expert_obs = data['observations'][:expert_length[-1]]

        # Plot the expert_obs
        ax.plot(expert_obs[:, 0], expert_obs[:, 1], label=f'Trajectory {idx}')

        ax.text(expert_obs[0, 0], expert_obs[0, 1], f'{file_name}', color='red')
        if len(expert_obs) > 5:
            ax.text(expert_obs[5, 0], expert_obs[5, 1], f'{file_name}', color='blue')
    plt.show()

class Dict2Class(object):
    def __init__(self, dict_input):
        for key in dict_input:
            setattr(self, key, dict_input[key])


# =============================================================================
# File handlers
# =============================================================================

def save_dict_as_json(dic, save_dir, name=None):
    if name is not None:
        save_dir = os.path.join(save_dir, name+".json")
    with open(save_dir, 'w') as out:
        out.write(json.dumps(dic, separators=(',\n','\t:\t'),
                  sort_keys=True))

def load_dict_from_json(load_from, name=None):
    if name is not None:
        load_from = os.path.join(load_from, name+".json")
    with open(load_from, "rb") as f:
        dic = json.load(f)

    return dic

def save_dict_as_pkl(dic, save_dir, name=None):
    if name is not None:
        save_dir = os.path.join(save_dir, name+".pkl")
    with open(save_dir, 'wb') as out:
        pickle.dump(dic, out, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_from_pkl(load_from, name=None):
    if name is not None:
        load_from = os.path.join(load_from, name+".pkl")
    with open(load_from, "rb") as out:
        dic = pickle.load(out)

    return dic

# =============================================================================
# Custom callbacks
# =============================================================================

class ProgressBarCallback(callbacks.BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = int(self.num_timesteps)
        self._pbar.update(0)

    def _on_rollout_end(self):
        total_reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        average_cost = safe_mean(self.model.rollout_buffer.orig_costs)
        total_cost = np.sum(self.model.rollout_buffer.orig_costs)
        self._pbar.set_postfix(
                tepr='%05.1f' % total_reward,
                ac='%05.3f' % average_cost,
                tc='%05.1f' % total_cost,
                nu='%05.1f' % self.model.dual.nu().item()
        )

# This callback should be used with the 'with' block, to allow for correct
# initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = int(total_timesteps)

    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps, dynamic_ncols=True)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class LogTorqueCallback(callbacks.BaseCallback):
    """
    This callback logs stats about actions.
    """
    def __init__(self, verbose: int=1):
        super(LogTorqueCallback, self).__init__(verbose)

    def _init_callback(self):
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        actions_abs = np.abs(self.model.rollout_buffer.actions.copy())
        greater_than_50 = np.sum(np.any(actions_abs > 0.5, axis=-1))
        greater_than_30 = np.sum(np.any(actions_abs > 0.3, axis=-1))
        greater_than_25 = np.sum(np.any(actions_abs > 0.25, axis=-1))
        mean_torque = np.mean(actions_abs, axis=(0,1))
        #var_torque = np.var(action_abs, axis=0)
        self.logger.record('torque/greater_than_0.5', greater_than_50)
        self.logger.record('torque/greater_than_0.3', greater_than_30)
        self.logger.record('torque/greater_than_0.25', greater_than_25)
        for i in range(mean_torque.shape[0]):
            self.logger.record('torque/mean_motor'+str(i),mean_torque[i])

class AdjustedRewardCallback(callbacks.BaseCallback):
    """
    This callback computes an estimate of adjusted reward i.e. R + lambda*C.
    """
    def __init__(self, cost_fn, verbose: int=1):
        super(AdjustedRewardCallback, self).__init__(verbose)
        self.history = []        # Use for smoothing if needed
        self.cost_fn = cost_fn

    def _init_callback(self):
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        rewards = self.model.rollout_buffer.rewards.copy()
        costs = self.model.rollout_buffer.costs.copy()
        if isinstance(self.training_env, vec_env.VecNormalize):
            rewards = self.training_env.unnormalize_reward(rewards)
        adjusted_reward = (np.mean(rewards - self.model.dual.nu().item()*costs))
        self.logger.record("rollout/adjusted_reward", float(adjusted_reward))
        if self.cost_fn is not None:
            obs = self.model.rollout_buffer.orig_observations.copy()
            acs = self.model.rollout_buffer.actions.copy()
            cost = np.mean(self.cost_fn(obs, acs))
            self.logger.record("eval/true_cost", float(cost))

class PlotCallback(callbacks.BaseCallback):
    """
    This callback can be used/modified to fetch something from the buffer and make a
    plot using some custom plot function.
    """
    def __init__(
        self,
        plot_fn,
        train_env_id: str,
        plot_freq: int = 10000,
        log_path: str = None,
        plot_save_dir: str = None,
        verbose: int = 1,
    ):
        super(PlotCallback, self).__init__(verbose)
        self.plot_fn = plot_fn
        self.log_path = log_path
        self.plot_save_dir = plot_save_dir

    def _init_callback(self):
        # Make directory to save plots
        del_and_make(os.path.join(self.plot_save_dir, "plots"))
        self.plot_save_dir = os.path.join(self.plot_save_dir, "plots")

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        try:
            obs = self.model.rollout_buffer.orig_observations.copy()
        except: # PPO uses rollout buffer which does not store orig_observations
            obs = self.model.rollout_buffer.observations.copy()
            # unormalize observations
            obs = self.training_env.unnormalize_obs(obs)
        obs = obs.reshape(-1, obs.shape[-1])    # flatten the batch size and num_envs dimensions
        self.plot_fn(obs, os.path.join(self.plot_save_dir, str(self.num_timesteps)+".png"))

class SaveEnvStatsCallback(callbacks.BaseCallback):
    def __init__(
            self,
            env,
            save_path
    ):
        super(SaveEnvStatsCallback, self).__init__()
        self.env = env
        self.save_path = save_path

    def _on_step(self):
        if isinstance(self.env, vec_env.VecNormalize):
            self.env.save(os.path.join(self.save_path, "train_env_stats.pkl"))

# =============================================================================
# Miscellaneous
# =============================================================================

def del_and_make(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)

def dict_to_nametuple(dic):
    return collections.namedtuple("NamedTuple", dic)(**dic)

def dict_to_namespace(dic):
    return types.SimpleNamespace(**dic)

def get_net_arch(config):
    """
    Returns a dictionary with sizes of layers in policy network,
    value network and cost value network.
    """
    try:
        separate_layers = dict(pi=config.policy_layers,    # Policy Layers
                           vf=config.reward_vf_layers, # Value Function Layers
                           cvf=config.cost_vf_layers)  # Cost Value Function Layers
    except:
        print("Could not define layers for policy, value func and "+ \
               "cost_value_function, will attempt to just define "+ \
               "policy and value func")
        separate_layers = dict(pi=config.policy_layers,    # Policy Layers
                               vf=config.reward_vf_layers) # Value Function Layers

    if config.shared_layers is not None:
        return [*config.shared_layers, separate_layers]
    else:
        return [separate_layers]

def get_sid():
    try:
        sid = subprocess.check_output(['/bin/bash', '-i', '-c', "who_am_i"], timeout=2).decode("utf-8").split('\n')[-2]
        sid = sid.lower()
        if "system" in sid:
            sid = sid.strip("system")
        else:
            sid = -1
    except:
        sid = -1
    return str(sid)

def sync_wandb(folder, timeout=None):
    folder = folder.strip("/files")
    print(colorize("\nSyncing %s to wandb" % folder, "green", bold=True))
    run_bash_cmd("wandb sync %s" % folder, timeout)

def run_bash_cmd(cmd, timeout=None):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    try:
        output, error = process.communicate(timeout=timeout)
    except:
        pass
