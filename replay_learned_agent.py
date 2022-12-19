import sys

import torch.random

sys.path.append('/home/tim/Documents/mushroom_rl_imitation_shared/')


from mushroom_rl.core.serialization import Serializable
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length


import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    agent = Serializable.load('/home/tim/Documents/IRL_unitreeA1/quadruped_gail_unitreeA1_2022-12-15_02-27-07/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_978_J_863.707600.msh')
    np.random.seed(2)
    torch.random.manual_seed(2)
    # action demo - need action clipping to be off
    env_freq = 1000  # hz, added here as a reminder simulation freq
    desired_contr_freq = 100  # hz contl freq.
    n_substeps =  env_freq // desired_contr_freq
    #to interpolate


    gamma = 0.99
    horizon = 1000

    # set a reward for logging
    reward_callback = lambda state, action, next_state: np.exp(- np.square(state[16] - 0.6))  # x-velocity as reward

    # prepare trajectory params
    traj_params = dict(traj_path="./data_current_Lcluster/dataset_only_states_unitreeA1_IRL.npz",
                       traj_dt=(1 / 500),
                       control_dt=(1 / desired_contr_freq))

    # create the environment
    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    traj_params=traj_params, random_start=True,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    core = Core(mdp=env, agent=agent)
    #core.agent.policy.deterministic = True
    dataset = core.evaluate(n_episodes=100, render=False)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset))
    L = np.mean(compute_episodes_length(dataset))

    print(J_mean)


"""
copy agent: scp -r ta11pajo@lcluster16.hrz.tu-darmstadt.de:/work/scratch/ta11pajo/quadruped_vail_2022-12-15_02-27-23/ /home/tim/Documents/IRL_unitreeA1/
log into ssh: ssh -L 16006:127.0.0.1:6006 ta11pajo@lcluster16.hrz.tu-darmstadt.de
execute in /work/scratch/ta11pajo/<experiment>: tensorboard --logdir quadruped_gail_unitreeA1_2022-12-15_02-27-07 --port 6006
-> open http://localhost:16006
"""

