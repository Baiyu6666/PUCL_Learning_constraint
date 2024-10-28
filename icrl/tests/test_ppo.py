import gym

from stable_baselines3 import PPO, TD3, SAC, DDPG, PPOLagrangian
import wandb
import gym_panda
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
wandb.init(project='icrl')
env = Monitor(gym.make("ReachObs-v0"), 'tests', info_keywords=('cost',), track_keywords=('cost', ))

# model = PPOLagrangian("TwoCriticsMlpPolicy", env, verbose=2, n_epochs=20, penalty_initial_value=0., penalty_learning_rate=0)
model = PPO("MlpPolicy", env, verbose=2)

# Train the model with WandbCallback
model.learn(total_timesteps=3, callback=WandbCallback()) # , cost_function='cost'
env.close()

env = gym.make("ReachObs-v0")
env.switch_mode()
obs = env.reset(render_mode=True)

for i in range(3000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    # print((reward))
    # VecEnv resets automatically
    if done:
      obs = env.reset()

env.close()