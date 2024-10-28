import sys
import wandb

if __name__ == "__main__":
    file_to_run = sys.argv[1]

    # choosing the specified file
    # all files must ignore the first argument passed via command line.
    file_to_run = sys.argv[1]

    # Run specified file. All files must ignore the first argument
    # passed via command line
    if file_to_run == "icrl":
        from icrl.icrl_main import main
        sweep_config = {
            'method': 'grid'
        }
        metric = {
            'name': 'true/jaccard',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {
            # 'cn_learning_rate': {
            #     'values': [0.003, 0.01]
            # },
            # 'penalty_initial_value': {
            #     'values':[1, 0.5, 2, 1.5]    # 0.8 thre  0.2 penalty for bce
            # },
            # 'cn_manual_threshold': {
            #     'values': [0.5]
            # },
            # 'cn_reg_coeff': {
            #     'values': [0.4, 0.5, 0.7, 1]  # [0.9, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55]
            # },
            # 'backward_iters':{
            #     'values': [20, 40]
            # },
            'seed': {
                'values': [1, 2, 3, 4,5,6,7,8,9,10, 11,12, 13, 14, 15, 16,17,18,19,20,21,22,23,24,25,26,27,28]  # [0.9, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55]
            },
            # 'ent_coef': {
            #     'values': [0.01, 0.0015, 0.02]
            # },
        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="ICRL-FE2")
        wandb.agent(sweep_id, main, )
    elif file_to_run == "pucl":
        from icrl.pucl import main

        sweep_config = {
            'method': 'grid'
        }
        metric = {
            'name': 'true/violation_steps_portion',
            'goal': 'minimize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {

            'seed': {
                'min':20, 'max':40
                # 'values': [10, 11, 12, 13, 14]
            },
            # 'GPU_likelihood_thresh':{'values':[-10, -7, -4]},
            # 'GPU_n_gmm': {'values': [8, 15]}
            # 'backward_iters': {'values':[50, 100, 200, 500]},
            # 'learning_rate': {
            #     'values': [1e-4, 3e-4]
            # },
            # 'ent_coef': {
            #         'values': [0.01, 0.0]
            #     },
            # 'n_epochs': {
            #     'values': [10, 20]
            # },
            # 'penalty_initial_value': {
            #     'min':0.3, 'max':1.0
            # },
            # 'penalty_initial_value': {
            #     'values': [0, 2, 5, 10]
            # },
            # 'WB_label_frequency': {
            #     'value': 1
            # },
            # 'cn_learning_rate':{'values':[3e-4, 5e-4, 10e-4, 20e-4]},
            # 'kNN_thresh': {
            #    'values': [0.03, 0.027, 0.025]
            # },
            # 'cn_layers': {
            #     'values': [[4], [16, 16]]
            # },
            # 'integral_control_coeff': {
            #     'values': [0.5, 0.1, 0.2]
            # },
            # 'proportional_control_coeff': {
            # 'values': [1, 5, 10, 20]
            # },
            # # 'forward_timesteps': {
            # #     'values':[10e4, 5e4, 1e4]
            # # },
            # 'kNN_thresh': {
            #     'values': [0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3]
            # },
            # 'GPU_likelihood_thresh': {
            #     'min':-10., 'max':-3.
            # },
            # 'GPU_n_gmm': {
            #     'min':5, 'max':15
            # },
            # 'policy_excerpts_weight': {
            #     'values':[1., 2., 4]
            # }
            # 'cn_reg_coeff': {
            #     'min':0., 'max':0.1
            #     # 'values': [0, 0.03, 0.05, 0.1]
            # },
            # 'backward_iters': {
            #     'values': [20, 50, 100, 500]
            # },

        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="ICRL-FE2")
        wandb.agent(sweep_id, main)
    elif file_to_run == "ppo-main":
        from icrl.ppo_main import main

        sweep_config = {
            'method': 'bayes'
        }
        metric = {
            'name': 'rollout/ep_is_success_mean',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {

            #
            # 'learning_rate': {
            #     'values': [2e-4, 10e-4]
            # },
            # 'n_epochs': {
            #     'values': [10, 20]
            # },
            # 'penalty_initial_value': {
            #     'values': [0.5, 1]
            # },
            'integral_control_coeff': {
                'min':0.5, 'max':1.5
            },
            'proportional_control_coeff': {
                'min':20., 'max': 30.
            },
            # 'ent_coef': {
            #     'values': [0.0, 0.005, 0.01]
            # },
            # 'budget':{
            #     'min': 0.001, 'max':0.02
            # }
            # 'seed': {'min':1, 'max':999}

        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="ICRL-FE2")
        wandb.agent(sweep_id, main)
    elif file_to_run == "dscl":
        from icrl.dscl import main

        sweep_config = {
            'method': 'random'
        }
        metric = {
            'name': 'true/jaccard',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {

            # 'cn_reg_coeff': {
            #     # 'min':0., 'max':0.5
            #     'values': [0,]
            # },
            # 'seed': {
            #     'min':0, 'max':11111
            # },
            # 'cn_pu_labeling_frequency': {
            #     'min': 0.1, 'max': 0.2
                # 'values': [0,  1.1, ]
            # },
            'cn_layers': {
                'values': [ [32, 32], [16, 16]]
            },
            'cn_learning_rate': {
                'values': [0.002, 0.005]
            },
            'kNN_thresh': {
               'min':0.010, 'max':0.019
            },
            'ds_modulation_rho': {
                'values': [2, 5, 10]
            },

            # 'cn_batch_size': {
            #     'values': [64, 128, 512]
            # },

        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="ICRL-FE2")
        wandb.agent(sweep_id, main)
    elif file_to_run == "pu_learning":
        from icrl.tests.test_pu_learning import main
        sweep_config = {
            'method': 'random'
        }
        metric = {
            'name': 'true/F1score',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {
            # 'cn_learning_rate': {
            #     'values': [0.005]
            # },
            # 'cn_reg_coeff': {
            #     'values': [0, 0.5, 1.1]  # [0.9, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55]
            # },
            # 'cn_manual_threshold': {
            #     'values': [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.999]
            # },
            'seed': {'values': [1, 2, 3, 4, 5]}

        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="PU_learning2")
        wandb.agent(sweep_id, main, )
    elif file_to_run == "acl":
        from icrl.acl import main

        sweep_config = {
            'method': 'grid'
        }
        metric = {
            'name': 'true/F1score',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {
            'cn_learning_rate': {
                'values': [0.005]
            },

            'seed': {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="acl_active")
        wandb.agent(sweep_id, main, )
    else:
        raise ValueError("File %s not defined" % file_to_run)

    # running



