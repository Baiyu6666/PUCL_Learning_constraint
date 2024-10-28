import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_checker import check_env

env_id = "UR5WithPos-v0"
video_folder = "tests/videos/"
video_length = 100
check_env(gym.make(env_id), warn=True, skip_render_check=True)
vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array", camera_name='top_down')])

obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent-{env_id}")

vec_env.reset()
for _ in range(video_length + 1):
  action = [vec_env.action_space.sample()]
  obs, _, _, _ = vec_env.step(action)
# Save the video
vec_env.close()