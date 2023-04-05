import sys


sys.path.append('/home/tim/Documents/mushroom_rl_imitation_shared/')


from mushroom_rl.core.serialization import Serializable
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length

from mushroom_rl.environments.mujoco_envs.quadrupeds.unitreeA1 import interpolate_remap, interpolate_map, reward_callback



import numpy as np




if __name__ == '__main__':

    gamma = 0.99
    horizon = 1000# 400

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq

    use_torque_ctrl = True
    use_2d_ctrl = True
    setup_random_rot = False

    # prepare trajectory params
    traj_params = dict(traj_path='./data/states_2023_02_23_19_48_33.npz',#'./data/states_2023_02_23_19_22_49.npz', dataset with less states in all directions
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq),
                       interpolate_map=interpolate_map,
                       interpolate_remap=interpolate_remap)

    # create the environment
    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=use_torque_ctrl,
                    traj_params=traj_params, random_start=False, init_step_no=10, init_traj_no=3,
                    use_2d_ctrl=use_2d_ctrl, tmp_dir_name='.', setup_random_rot=setup_random_rot,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    agents = [ #best final agent
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_259_J_974.325803.msh'
        ),
    ]

    for i in range(len(agents)):# for runs with multple agents, outcomment self.mdp.stop() un _run_impl
        print('')
        print(i, agents[i])
        #plot_data_callbacks = PlotDataset(env.info)
        #core = Core(mdp=env, agent=agent, callback_step=plot_data_callbacks)
        core = Core(mdp=env, agent=agents[i])
        #core.agent.policy.deterministic = False
        dataset, env_info = core.evaluate(n_episodes=11, render=True, get_env_info=True)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))

        print("J_mean: ", J_mean)
        print("R_mean: ", R_mean)
        print("L_mean:", L)



