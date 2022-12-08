import sys

sys.path.append('/home/tim/Documents/mushroom_rl_imitation_shared/')


from mushroom_rl.core.serialization import Serializable
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.core import Core



if __name__ == '__main__':
    agent = Serializable.load('/home/tim/Documents/IRL_unitreeA1/'
                              'Gail/logs/quadruped_gail_unitreeA1_2022-12-07_18-37-46/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.9/2/agent_epoch_0_J_0.000000.msh')




    # action demo - need action clipping to be off
    env_freq = 1000  # hz, added here as a reminder simulation freq
    desired_contr_freq = 500  # hz contl freq.
    n_substeps =  env_freq // desired_contr_freq
    #to interpolate


    gamma = 0.99
    horizon = 1000



    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_action_clipping=False)




    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    core = Core(mdp=env, agent=agent)

    core.evaluate(n_episodes=10, render=True)



