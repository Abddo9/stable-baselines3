import os
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.configs import load_config_data
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym

# for visualization
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show, save


# Dataset
os.environ["L5KIT_DATA_FOLDER"] = '../data'

# get environment config
env_config_path = './drivergym_config/gym_config.yaml'
cfg = load_config_data(env_config_path)


# Train on episodes of length 32 time steps
train_eps_length = 32
train_envs = 4

# Evaluate on entire scene (~248 time steps)
eval_eps_length = None
eval_envs = 1

# make train env
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=train_envs,
                   vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

# make eval env
validation_sim_cfg = SimulationConfigGym()
validation_sim_cfg.num_simulation_steps = None
eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, \
                   'return_info': True, 'train': False, 'sim_cfg': validation_sim_cfg}
eval_env = make_vec_env("L5-CLE-v0", env_kwargs=eval_env_kwargs, n_envs=eval_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})


print("reach step number .....................................1")
# Define backbone feature extractor (To be replaced)
# A simple 2 Layer CNN architecture with group normalization
model_arch = 'simple_gn'
features_dim = 128

# Custom Feature Extractor backbone
policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": features_dim, "model_arch": model_arch},
    "normalize_images": False
}


# We linearly decrease the value of the clipping parameter ϵ as the PPO training progress as it shows improved training stability
# Clipping schedule of PPO epsilon parameter
start_val = 0.1
end_val = 0.01
training_progress_ratio = 1.0
clip_schedule = get_linear_fn(start_val, end_val, training_progress_ratio)

#Hyperparameters for PPO
lr = 3e-4
num_rollout_steps = 256
gamma = 0.8
gae_lambda = 0.9
n_epochs = 10
seed = 42
batch_size = 64


tensorboard_log = 'tb_log'

print("reach step number .....................................2")
# define model  (Here replace with Dreamer)
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=num_rollout_steps,
            learning_rate=lr, gamma=gamma, tensorboard_log=tensorboard_log, n_epochs=n_epochs,
            clip_range=clip_schedule, batch_size=batch_size, seed=seed, gae_lambda=gae_lambda)


#Defining Callbacks
callback_list = []

# Save Model Periodically
save_freq = 1000
save_path = './logs/'
output = 'PPO'
checkpoint_callback = CheckpointCallback(save_freq=(save_freq // train_envs), save_path=save_path, \
                                         name_prefix=output)
callback_list.append(checkpoint_callback)

# Eval Model Periodically
eval_freq = 1000
n_eval_episodes = 1
val_eval_callback = L5KitEvalCallback(eval_env, eval_freq=(eval_freq // train_envs), \
                                      n_eval_episodes=n_eval_episodes, n_eval_envs=eval_envs)
callback_list.append(val_eval_callback)

print("reach step number .....................................3")

n_steps = 10000
model.learn(n_steps, callback=callback_list)


print("reach step number .....................................4")


# Visualization
# Visualize Tensorboard logs (!! run on local terminal !!)
#!tensorboard --logdir tb_log (we need to find away to import them from compute nodes)


rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = None
rollout_env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                       use_kinematic=True, train=False, return_info=True)

def rollout_episode(model, env, idx = 0):
    """Rollout a particular scene index and return the simulation output.

    :param model: the RL policy
    :param env: the gym environment
    :param idx: the scene index to be rolled out
    :return: the episode output of the rolled out scene
    """

    # Set the reset_scene_id to 'idx'
    env.reset_scene_id = idx

    # Rollout step-by-step
    obs = env.reset()
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

# Rollout one episode
sim_out = rollout_episode(model, rollout_env)
print("reach step number .....................................5")

# might change with different rasterizer
map_API = rollout_env.dataset.rasterizer.sem_rast.mapAPI

def visualize_outputs(sim_outs, map_API):
    for sim_out in sim_outs: # for each scene
        vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, map_API)
        save(visualize(sim_out.scene_id, vis_in), filename='./results/out.jif')
        show(visualize(sim_out.scene_id, vis_in))

output_notebook()
visualize_outputs([sim_out], map_API)

print("reach step number .....................................6")
