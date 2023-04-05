import os.path
import sys

import torch.random

sys.path.append('/home/tim/Documents/mushroom_rl_imitation_shared/')


from mushroom_rl.core.serialization import Serializable
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.utils.callbacks import PlotDataset

from mushroom_rl.environments.mujoco_envs.quadrupeds.unitreeA1 import interpolate_remap, interpolate_map, reward_callback


import os

import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    """
    x = np.linspace(-1.5, 1.5, 200)
    y = [np.exp(-np.square(i)) for i in x]

    fig = plt.figure(figsize=[10,4])
    ax = fig.gca()
    ax.spines['left'].set_position('center')
    #ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    plt.plot(x,y, color='r')
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([0.001,1.2])
    plt.savefig("reward.pdf")
    exit()"""


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
    traj_params = dict(traj_path='./data/states_2023_02_23_19_48_33.npz',#'/home/tim/Documents/locomotion_simulation/locomotion/examples/log/2023_02_23_19_22_49/states.npz',#
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




    """
    agents =[ # all good agents from gail 8 rot
        #Serializable.load(
            # sicherer gut/ manchmal kleine schritte - echt gut                                       583
            # slided übern bodem bei diag, rückwärt weird; rest eigtl gut                                           580
            # ----------------- best -------------------
        #    '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_225_J_968.071239.msh'
        #),
        #Serializable.load(
            # echt gut - wriklich                                                                     580
            # leicht unsicher/slided bisschen aber eigtl gut                                                        575
            # ----------------- best -------------------
        #    '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_195_J_960.741299.msh'
        #),
        Serializable.load(
            # noch besser - nix u meckern                                                             583
            # wundervoll auch wenn manchmal bisschen zögerlich

            # ----------------- best -------------------
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_239_J_980.487975.msh'
        ),
        Serializable.load(
            # auch gut aber kleines bissche schelechter. slided mit fuß manchmal über boden           579
            # manchmal zögerlich aber sonst echt gut                                                                582

            # ----------------- best -------------------
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_259_J_974.325803.msh'
        ),
    ]

    agents=[ # states straight gail and vail
        Serializable.load( # near perfect
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_163_J_993.748105.msh'
        ),
        Serializable.load(# kleines bisschen doppelschritte in der luft
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_203_J_990.925083.msh'
        ),
        Serializable.load( # acuh fast perfekt, gaaaaanz kleine doppelschritte, aber eirde initialisierung
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_154_J_992.730184.msh'
        ),
        Serializable.load(# super weirde initialisierung, doppelschritte kleiens bisschen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_208_J_989.900737.msh'
        ),
        Serializable.load(# echt gut auch initialisierung
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_151_J_991.161858.msh'
        ),
        Serializable.load(# auch echt gut. vllt kleinere schritte?
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_234_J_991.283042.msh'
        ),
        Serializable.load(# bisschen doppelschritte in der luft
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_192_J_990.491126.msh'
        ),
        Serializable.load( # schlechte init, sonst bisschen bouncen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_243_J_990.572069.msh'
        ),
        Serializable.load( # bisschen rutschen und komischess bein zurück ziehen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_151_J_992.915275.msh'
        ),
        Serializable.load( # echt gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_243_J_990.542282.msh'
        ), # VAIL
        Serializable.load( #  perfekt, auch glaube ich bisschen langsamere schritte als rest
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_52_J_990.620967.msh'
        ),
        Serializable.load(# kleienes springen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_103_J_989.246972.msh'
        ),
        Serializable.load( # echt gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_58_J_991.576309.msh'
        ),
        Serializable.load(# kleines springen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_138_J_991.071305.msh'
        ),
        Serializable.load(# irgendwas stört bischen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_78_J_991.035533.msh'
        ),
        Serializable.load(#   super kleine doppelschritte eigtl gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_117_J_990.906804.msh'
        ),
        Serializable.load(# echt gut nicht opt init
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_72_J_992.725329.msh'
        ),
        Serializable.load(# kleine sprünge
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_108_J_993.477391.msh'
        ),
        Serializable.load(# kleine sprünge
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_61_J_992.382164.msh'
        ),
        Serializable.load(# gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_109_J_990.321945.msh'
        ),

    ]
    agents=[# action states gail and vail
        Serializable.load(# bad init sonst echt gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_162_J_991.818088.msh'
        ),
        Serializable.load(# really really bad init; breaks after length 12
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_221_J_988.592809.msh'
        ),
        Serializable.load(# kleine sprünge, nicht so gut init
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_158_J_989.156518.msh'
        ),
        Serializable.load(# weirde sprünge>; bricht ab
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_245_J_980.153742.msh'
        ),
        Serializable.load(# ganz schlecht init, fällt fast hin, doppelschritte leicht
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_178_J_990.136975.msh'
        ),
        Serializable.load(# bad init, sprünge zwischendurch
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_221_J_991.588157.msh'
        ),
        Serializable.load(# bad init, sprünge
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_192_J_991.759077.msh'
        ),
        Serializable.load(#  sprünge
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_225_J_989.890161.msh'
        ),
        Serializable.load(# echt gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_168_J_990.451724.msh'
        ),
        Serializable.load(# auch gut aber bisschen vibration in bewegungen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_220_J_991.132682.msh'
        ), # VAIL
        Serializable.load(# leicht luftschritte sonst gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/0/agent_epoch_96_J_988.079454.msh'
        ),
        Serializable.load(# echt gut, ok init
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/0/agent_epoch_110_J_988.601820.msh'
        ),
        Serializable.load( # kleine hüpfer, gut init
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/1/agent_epoch_73_J_991.051098.msh'
        ),
        Serializable.load( # kleine hüpfer
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/1/agent_epoch_123_J_990.560393.msh'
        ),
        Serializable.load( # echt gut - perfect
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/2/agent_epoch_85_J_990.096023.msh'
        ),
        Serializable.load(# doppelschritte/hüpfer
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/2/agent_epoch_112_J_988.219759.msh'
        ),
        Serializable.load(# vibrieren in bewegung, eigtl echt gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/3/agent_epoch_94_J_989.713770.msh'
        ),
        Serializable.load(# weirde init, schlecht, vibrieren in bewegung
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/3/agent_epoch_110_J_989.790610.msh'
        ),
        Serializable.load(# kleine schritte sonst gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/4/agent_epoch_59_J_989.836638.msh'
        ),
        Serializable.load(# echt gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_2023-03-29_02-10-35/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/info_constraint___1/horizon___1000/gamma___0.99/4/agent_epoch_126_J_989.911950.msh'
        ),

    ]
    
    agents=[# position control gail and vail
        Serializable.load(# keine schritte strampelt durch die gegen, virbiert in richtige richtung, sehr unsicher
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-31_20-13-06/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_152_J_948.781806.msh'
        ),
        Serializable.load(# man erkennt gait aber füße vibrieren trotzdem
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-31_20-13-06/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_198_J_963.699697.msh'
        ),

    ]
    

    agents=[# more detailed straight torque states; agents not mentioned in first/good range
        Serializable.load( # bisschen unsicher mini steps
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_150_J_985.442205.msh'
        ),
        Serializable.load( # doppelschritte vibration leicht
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_299_J_988.802128.msh'
        ),
        Serializable.load( # perfect
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_130_J_992.371727.msh'
        ),
        Serializable.load( # gaaanz kleine sbisschen doppelschritte
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_276_J_992.618401.msh'
        ),
        Serializable.load( #  top
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_143_J_991.690504.msh'
        ),
        Serializable.load( # mini steps/ kleines bisschen doppelschritte
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_253_J_989.405834.msh'
        ),
        Serializable.load( # top
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_144_J_991.166891.msh'
        ),
        Serializable.load( # mini hopser
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_251_J_992.609042.msh'
        ),
        Serializable.load( # mini biosschen zu viel bouncer
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_146_J_992.829254.msh'
        ),
        Serializable.load( # gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_296_J_988.031771.msh'
        ), # VAIL
        Serializable.load( # mini springer
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_32_J_991.909126.msh'
        ),
        Serializable.load( # mini zwischenspringer
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_152_J_987.161627.msh'
        ),
        Serializable.load(# manchmal freeze/beweget nicht - weniger weit gekommen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_245_J_987.086628.msh'
        ),
        Serializable.load( # gut miniiii zwischenschritte; generell kleine schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_40_J_990.756192.msh'
        ),
        Serializable.load( # humpelt - eine seite schneller als andere
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_200_J_964.547654.msh'
        ),
        Serializable.load( # humpelt auch nur mit keleineren schritten
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_215_J_969.670703.msh'
        ),
        Serializable.load( #  super
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_40_J_990.508615.msh'
        ),
        Serializable.load( #  noch besser größere schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_160_J_990.544543.msh'
        ),
        Serializable.load( # zuppper
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_235_J_990.016233.msh'
        ),
        Serializable.load( # kein gait, humpelt, stolpert, vibriert
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_50_J_972.551628.msh'
        ),
        Serializable.load( # miniiiii doppelschritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_165_J_991.042948.msh'
        ),
        Serializable.load( #  humpelt
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_207_J_988.876185.msh'
        ),
        Serializable.load( # mini schritte und bisschen doppel unregelmäßig
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_50_J_988.510832.msh'
        ),
        Serializable.load( # wunderbar auch große schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_151_J_987.368356.msh'
        ),
        Serializable.load( # bisschen leicht komische bewegungen in  der luft aber eigtl super
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_215_J_988.479815.msh'
        ),
    ]
    agents=[ # last vail agents for only states straight, torque
        Serializable.load( # humpelt
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_283_J_987.097776.msh'
        ),
        Serializable.load( #
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_285_J_970.580984.msh'
        ),
        Serializable.load( # humpelt/srpünge sehr kleine schritte dann große sprünge
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_254_J_991.109760.msh'
        ),
        Serializable.load(# gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_251_J_986.910696.msh'
        ),
        Serializable.load( # mini doppelt schritte mit sdprüngen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_261_J_989.179258.msh'
        ),

    ]

    agents = [ # vail vs gail 8 dir
        Serializable.load(#0 gaaanz kleines bisschend oppelschritte sonst gut; manchmal humpelt/abstände zwischen links rechts nicht gleich
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_197_J_978.195793.msh'
        ),
        Serializable.load(#1 mini hopser; gar nicht so mini; aber meistens ganz gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_208_J_981.670427.msh'
        ),
        Serializable.load(#2 kein schöner gait; kleine sprünge; unregelmäßige schrittgrößen; schritte in luft
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_290_J_971.706233.msh'
        ),
        Serializable.load(#3 vibriert viel im boden; sehr ruckartige bewegungen; kein schöner gait; manchmal stuck
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_200_J_963.966277.msh'
        ),
        Serializable.load(#4  bisschen vibration in luft aber ganz ok; bisschen humpeln
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_222_J_981.553741.msh'
        ),
        Serializable.load(#5 bisschen doppelschritte aber stabiler als davor; links schritt schneller als rechts
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_257_J_978.726052.msh'
        ),
        Serializable.load(#6 vibration & doppelschritte
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_194_J_977.581871.msh'
        ),
        Serializable.load(#7nicht ganz regelmäßige schritte aber oker laufstil
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_221_J_983.961019.msh'
        ),
        Serializable.load(#8 doppelschritte/kleine sprünge
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_264_J_978.260141.msh'
        ),
        Serializable.load(#9 viel vibration in luft, manchmal kleine sprünge
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_182_J_973.653327.msh'
        ),
        Serializable.load(#10 unregelmäßige bewegungen, gut kleine doppelschritte
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_236_J_977.421047.msh'
        ),
        Serializable.load(#11 kleine sprünge, virbrieren in luft
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_253_J_980.207687.msh'
        ),
        Serializable.load(#12 verzögertes absetzen vore linkssont gut; geradeaus top
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_200_J_974.737811.msh'
        ),
        Serializable.load(#13 vibration in Luft nicht immer; sonst echt gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_204_J_980.078185.msh'
        ),
        Serializable.load(#14 komiusch rückwärts springt unregelmäßig
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_254_J_971.428689.msh'
        ), # Vail
        Serializable.load(#15 tippelt; sehr kleine schritt - strehen; vibriert
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_50_J_974.668896.msh'
        ),
        Serializable.load(#16  vibriert immer noch merh als laufen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_78_J_980.790142.msh'
        ),
        Serializable.load(#17 same here; mini schritte vibriert
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_102_J_970.901197.msh'
        ),
        Serializable.load(#18 mehr schritte, doppelschritte hinkt bisschen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_193_J_971.091884.msh'
        ),
        Serializable.load(  # kleine hüpfer etwas unsicher
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_210_J_978.693075.msh'
        ),
        Serializable.load(#19  guter gait, stable, springt bisschen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_295_J_982.053098.msh'
        ),
        Serializable.load(#20 vibriert in richtung
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_49_J_958.778880.msh'
        ),
        Serializable.load(#21 same vibriert sogar mehr auf stelle
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_71_J_971.130995.msh'
        ),
        Serializable.load(#22 immer noch unsicher, hoch runter, hüpft mehr in richtung
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_102_J_971.809257.msh'
        ),
        Serializable.load(#23 same here, unsicher, steckt fest; manchmal schöne phasen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_179_J_961.702365.msh'
        ),
        Serializable.load(  # hüpft nur; sehr unsicher, steckt fest
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_235_J_959.991900.msh'
        ),
        Serializable.load(#24 schritte in luft, doppelschritte aber immerhin gait; meher oder weniger; manchmal auch starre
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_292_J_945.571601.msh'
        ),
        Serializable.load(#25 virbriert again; mini schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_41_J_976.242798.msh'
        ),
        Serializable.load(#26 besser aber noch nicht schön, immer noch sehr kleine schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_69_J_983.218680.msh'
        ),
        Serializable.load(#27 guuuut -.----- pefect
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_121_J_985.057980.msh'
        ),
        Serializable.load(#28 gaaanz kleines bisschen doppelschritte aber auch gut; kleinere schritte; für manche init pos echt gut für manche nicht so
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_169_J_979.578695.msh'
        ),
        Serializable.load(  # sehr kleine schritte, hüpft bisshcen, manchmal echt gut
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_202_J_981.276797.msh'
        ),
        Serializable.load(#29 löeome scjrotte aner eigtl auch echt gut; manchmal kleine jumps
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_269_J_979.873530.msh'
        ),
        Serializable.load(#30 vibration pur
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_41_J_960.521295.msh'
        ),
        Serializable.load(#31 same here nur weniger gait
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_86_J_983.125053.msh'
        ),
        Serializable.load(#32kleine unsichere schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_101_J_966.079990.msh'
        ),
        Serializable.load(#33 unsicher viel gespringe und virbration
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_190_J_952.830371.msh'
        ),
        Serializable.load(  # hüpft bisschen/doppelschritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_235_J_958.201936.msh'
        ),
        Serializable.load(#34 oker gait aber viel gespringe, mini schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_296_J_959.493092.msh'
        ),
        Serializable.load(#35 mini schritte
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_44_J_969.353797.msh'
        ),
        Serializable.load(#36 immer nich sehr kleine schritte; bisschen springen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_54_J_969.988014.msh'
        ),
        Serializable.load(#37  gute schritte gait aber klein
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_118_J_965.375245.msh'
        ),
        Serializable.load(#38 sehr viel vibration/feststecken/langsam
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_183_J_948.846372.msh'
        ),
        Serializable.load(#39 vorsichtig langsam, manchmak sprünge in luft
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_261_J_959.317084.msh'
        ),

    ]


    agents = [ # gail vs Vail 8 dir rot
        Serializable.load( # 0 bisschen sprünge und unsicher
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_200_J_953.776142.msh'
        ),
        Serializable.load(  # 1 ok aber auch bisschen ruckartig
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_225_J_968.071239.msh'
        ),
        Serializable.load(  # 2 mini sprünge echt gut eigtl
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_251_J_961.892534.msh'
        ),
        Serializable.load(  # 3 steckt manchmal fest, viel vibration in füßen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_197_J_931.670027.msh'
        ),
        Serializable.load(  # 4 besser, läuft manchmal auf stelle, etwas unsymmetrisch und kleine schritte aber egtl ganzt gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_232_J_953.546222.msh'
        ),
        Serializable.load(  # 5 rutscht bisschen weg/kleine doppelschritte, manchmal verhakt, aber eigtl echt gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_262_J_949.889729.msh'
        ),
        Serializable.load(  # 6 sehr schenlle bewegungen virl vibration
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_186_J_944.160390.msh'
        ),
        Serializable.load(  # 7 manchmal bisschen in luft absetzen aber eigtl gut
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_237_J_952.863260.msh'
        ),
        Serializable.load(  # 8 echt gut wirklich; perfekt
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_264_J_967.536016.msh'
        ),
        Serializable.load(  # 9 wiueder vibration drin, kl schritte, steckt manchmal fest
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_195_J_944.341154.msh'
        ),
        Serializable.load(  # 10 ganz gut, bisschen vibrieren/unregelmäßig
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_226_J_950.013917.msh'
        ),
        Serializable.load(  # 11 gut, kleines bisschen doppelschritte/ unrund
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_253_J_952.279527.msh'
        ),
        Serializable.load(  # 12 bisschen bopuncen aber eigtl echt gut für die epoch;
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_195_J_960.741299.msh'
        ),
        Serializable.load(  # 13 toop gut manchmak bisschen unsicer/fuß hoch runter, schöner gait etc
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_239_J_980.487975.msh'
        ),
        Serializable.load(  # 14 perfekt
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_259_J_974.325803.msh'
        ), # VAIL
        Serializable.load(  # 15 steht ohne schritte, nur vibrieren in beinen/ fängt richtig an aber zieht nicht durch; zu viel aangst zu fallen?
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_44_J_845.015713.msh'
        ),
        Serializable.load(  # 16 same mit bisschen weniger gesamt vibration
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_153_J_808.887219.msh'
        ),
        Serializable.load(  # 17 stlpert b isschen manchmal in richtung, größenteils auch sehr unsicher
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_253_J_840.600072.msh'
        ),
        Serializable.load(  # 18
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_20_J_831.228421.msh'
        ),
        Serializable.load(  # 19kleine schrtitte in richtige richtung, aber immer noch sehr unsicher
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_146_J_840.012245.msh'
        ),
        Serializable.load(  # 20 same here auch bisschen langsam in richtige richtung
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_283_J_862.180395.msh'
        ),
        Serializable.load(  # 21
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_29_J_839.184076.msh'
        ),
        Serializable.load(  # 22 eigtl echt gut aber sehr kleine schritte und manchmal fallen
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_112_J_864.482509.msh'
        ),
        Serializable.load(  # 23
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_278_J_920.411263.msh'
        ),
        Serializable.load(  # 24
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_42_J_815.106848.msh'
        ),
        Serializable.load(  # 25
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_192_J_811.203094.msh'
        ),
        Serializable.load(  # 26
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_280_J_833.981300.msh'
        ),
        Serializable.load(  # 27
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_33_J_843.094924.msh'
        ),
        Serializable.load(  # 28
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_136_J_895.463830.msh'
        ),
        Serializable.load(  # 29
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_288_J_914.502283.msh'
        ),

    ]"""

    agents = [ # bad vs good in dir rot
        Serializable.load(  # 3 steckt manchmal fest, viel vibration in füßen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_197_J_931.670027.msh'
        ),
        Serializable.load(  # 14 perfekt
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_259_J_974.325803.msh'
        ),

    ]

    agents = [  # position vs torque straight
        Serializable.load(  # keine schritte strampelt durch die gegen, virbiert in richtige richtung, sehr unsicher
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-31_20-13-06/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_152_J_948.781806.msh'
        ),
        Serializable.load(  # near perfect
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_163_J_993.748105.msh'
        ),

    ]
    agents = [  # compare actions vs states gail - to highlight jumps
        Serializable.load(  # bad init, sprünge
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_2023-03-29_02-10-23/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/3/agent_epoch_192_J_991.759077.msh'
        ),
        Serializable.load(  # near perfect
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_163_J_993.748105.msh'
        ),
    ]

    agents = [ # straight double steps vs good
        Serializable.load(  # super weirde initialisierung, doppelschritte kleiens bisschen
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_208_J_989.900737.msh'
        ),
        Serializable.load(  # near perfect
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_163_J_993.748105.msh'
        ),
    ]

    agents=[# double steps vs good

        Serializable.load(
            # 24 schritte in luft, doppelschritte aber immerhin gait; meher oder weniger; manchmal auch starre
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_292_J_945.571601.msh'
        ),
        Serializable.load(  # 27 guuuut -.----- pefect
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_121_J_985.057980.msh'
        ),

    ]

    agents = [ #best final agents

        Serializable.load(  # 14 perfekt
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_259_J_974.325803.msh'
        ),

    ]
    agents = [ # for front picture series
        Serializable.load(  # near perfect
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_163_J_993.748105.msh'
        ),
    ]

    agents = [ #best final agents

        Serializable.load(  # 14 perfekt
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/4/agent_epoch_259_J_974.325803.msh'
        ),

    ]




     # good:             '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-02-13_22-00-37/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_145_J_974.749549.msh'),
    # good best            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-02-13_22-00-37/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_200_J_973.474543.msh'),
    # good best            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-02-13_21-54-24/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_221_J_982.343283.msh'),
    # good             '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-02-13_21-54-24/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_304_J_970.560817.msh'),
    #TODO core zeile 183 nicht auskommentieren !!!!!!!!!!!!!!!!!!!!!!!!!
    a = [list() for i in range(len(agents))]
    for i in range(len(agents)):
        print('')
        print(i, agents[i])
        #plot_data_callbacks = PlotDataset(env.info)
        #core = Core(mdp=env, agent=agent, callback_step=plot_data_callbacks)
        core = Core(mdp=env, agent=agents[i])
        #core.agent.policy.deterministic = False
        dataset, env_info, foot_pos = core.evaluate(n_episodes=11, render=True, get_env_info=True) # todo
        a[i] = foot_pos
        A = compute_J(dataset)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))

        print("J_mean: ", J_mean)
        print("R_mean: ", R_mean)
        print("L_mean:", L)



    b0 = [
        a[0][0][0],
        a[0][1][0],
        a[0][2][0],
        a[0][3][0],
        #a[1][0][0],
        #a[1][1][0],
        #a[1][2][0],
        #a[1][3][0],
    ]

    data = {
        'FR': b0[0],
        'RR': b0[1],
        'FL': b0[2],
        'RL': b0[3],
    }
    fig = plt.figure()
    ax = fig.gca()
    plt.rcParams['font.size'] = 15
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.hlines(0.01, 0, len(b0[0]), linestyle='dashed', color='r')
    #ax.plot(np.zeros(len(foot_pos[0][0])), linestyle='dashed', color='r')
    ax.set_ylim([-0.01,0.17])
    plt.legend(loc=4)
    plt.xlabel("Steps", weight='bold', fontsize=15)
    plt.ylabel("Height", weight='bold', fontsize=15)
    plt.savefig("foot_pos_bad.pdf", bbox_inches='tight')

    data = {
        #'FR': b0[4],
        #'RR': b0[5],
        'FL': b0[6],
        'RL': b0[7],
    }
    fig = plt.figure()
    ax = fig.gca()
    plt.rcParams['font.size'] = 15
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.hlines(0.01, 0, len(b0[4]), linestyle='dashed', color='r')
    #ax.plot(np.zeros(len(foot_pos[0][0])), linestyle='dashed', color='r')
    ax.set_ylim([-0.01,0.17])
    plt.legend(loc=4)
    plt.xlabel("Steps", weight='bold', fontsize=15)
    plt.ylabel("Height", weight='bold', fontsize=15)
    plt.savefig("foot_pos_good.pdf",  bbox_inches='tight')



    # velocity ---------------------------------------------------------------------------------------------------------
    b1 = [list() for i in range(8)]
    for i in range(len(a[0][0][0]) - 1):
        b1[0].append((a[0][0][0][i + 1] - a[0][0][0][i]) * desired_contr_freq )
        b1[1].append((a[0][1][0][i + 1] - a[0][1][0][i]) * desired_contr_freq )
        b1[2].append((a[0][2][0][i + 1] - a[0][2][0][i]) * desired_contr_freq )
        b1[3].append((a[0][3][0][i + 1] - a[0][3][0][i]) * desired_contr_freq )
    for i in range(len(a[1][0][0]) - 1):
        b1[4].append((a[1][0][0][i + 1] - a[1][0][0][i]) * desired_contr_freq )
        b1[5].append((a[1][1][0][i + 1] - a[1][1][0][i]) * desired_contr_freq )
        b1[6].append((a[1][2][0][i + 1] - a[1][2][0][i]) * desired_contr_freq )
        b1[7].append((a[1][3][0][i + 1] - a[1][3][0][i]) * desired_contr_freq )

    data = {
        'FR': b1[0],
        'RR': b1[1],
        'FL': b1[2],
        'RL': b1[3],
    }
    fig = plt.figure()
    ax = fig.gca()
    plt.rcParams['font.size'] = 15
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.hlines(0, 0, len(b1[4]), linestyle='dashed', color='r')
    #ax.plot(np.zeros(len(foot_pos[0][0])), linestyle='dashed', color='r')
    ax.set_ylim([-3.2, 3.2])
    plt.legend(loc=4)
    plt.xlabel("Steps", weight='bold', fontsize=15)
    plt.ylabel("Velocity", weight='bold', fontsize=15)
    plt.savefig("foot_vel_bad.pdf",  bbox_inches='tight')

    data = {
        'FR': b1[4],
        'RR': b1[5],
        'FL': b1[6],
        'RL': b1[7],
    }
    fig = plt.figure()
    ax = fig.gca()
    plt.rcParams['font.size'] = 15
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.hlines(0, 0, len(b1[4]), linestyle='dashed', color='r')
    #ax.plot(np.zeros(len(foot_pos[0][0])), linestyle='dashed', color='r')
    ax.set_ylim([-3.2, 3.2])
    plt.legend(loc=4)
    plt.xlabel("Steps", weight='bold', fontsize=15)
    plt.ylabel("Velocity", weight='bold', fontsize=15)
    plt.savefig("foot_vel_good.pdf",  bbox_inches='tight')

    # Plotting foot acceleration --------------------------------------------------------------------------------------

    b2 = [list() for i in range(8)]
    for i in range(len(a[0][0][0])-1):
        b2[0].append((a[0][0][0][i + 1] - a[0][0][0][i]) * desired_contr_freq* desired_contr_freq)
        b2[1].append((a[0][1][0][i + 1] - a[0][1][0][i]) * desired_contr_freq* desired_contr_freq)
        b2[2].append((a[0][2][0][i + 1] - a[0][2][0][i]) * desired_contr_freq* desired_contr_freq)
        b2[3].append((a[0][3][0][i + 1] - a[0][3][0][i]) * desired_contr_freq* desired_contr_freq)
    for i in range(len(a[1][0][0])-1):
        b2[4].append((a[1][0][0][i + 1] - a[1][0][0][i]) * desired_contr_freq* desired_contr_freq)
        b2[5].append((a[1][1][0][i + 1] - a[1][1][0][i]) * desired_contr_freq* desired_contr_freq)
        b2[6].append((a[1][2][0][i + 1] - a[1][2][0][i]) * desired_contr_freq* desired_contr_freq)
        b2[7].append((a[1][3][0][i + 1] - a[1][3][0][i]) * desired_contr_freq* desired_contr_freq)

    data = {
        'FR': b2[0],
        'RR': b2[1],
        'FL': b2[2],
        'RL': b2[3],
    }
    fig = plt.figure()
    ax = fig.gca()
    plt.rcParams['font.size']=15
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.hlines(0, 0, len(b2[4]), linestyle='dashed', color='r')
    #ax.plot(np.zeros(len(foot_pos[0][0])), linestyle='dashed', color='r')
    ax.set_ylim([-310, 310])
    plt.legend(loc=4)
    plt.xlabel("Steps", weight='bold', fontsize=15)
    plt.ylabel("Acceleration", weight='bold', fontsize=15)
    plt.savefig("foot_acc_bad.pdf",  bbox_inches='tight')

    data = {
        'FR': b2[4],
        'RR': b2[5],
        'FL': b2[6],
        'RL': b2[7],
    }
    fig = plt.figure()
    ax = fig.gca()
    plt.rcParams['font.size'] = 15
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.hlines(0, 0, len(b2[4]), linestyle='dashed', color='r')
    #ax.plot(np.zeros(len(foot_pos[0][0])), linestyle='dashed', color='r')
    ax.set_ylim([-310, 310])
    plt.xlabel("Steps", weight='bold', fontsize=15)
    plt.ylabel("Acceleration", weight='bold', fontsize=15)
    plt.savefig("foot_acc_good.pdf",  bbox_inches='tight')

"""
copy agent: scp -r ta11pajo@lcluster16.hrz.tu-darmstadt.de:/work/scratch/ta11pajo/quadruped_vail_2022-12-15_02-27-23/ /home/tim/Documents/IRL_unitreeA1/
log into ssh: ssh -L 16006:127.0.0.1:6006 ta11pajo@lcluster16.hrz.tu-darmstadt.de
execute in /work/scratch/ta11pajo/<experiment>: tensorboard --logdir quadruped_gail_unitreeA1_2022-12-15_02-27-07 --port 6006
-> open http://localhost:16006
"""

"""
files in scratch:
quadruped_gail_unitreeA1_2022-12-15_02-27-07: first run - position ctrl; couldsn't find stable walking version (shaking and friction too low?)
quadruped_gail_unitreeA1_only_states_2022-12-15_11-30-08 - position; forgot to cut out init stepping
quadruped_gail_unitreeA1_only_states_2022-12-21_19-24-48 - torque ctrl; with 250k dataset
    stable; has peak at 250k whats after that; no as stable as vail for gait init
quadruped_vail_2022-12-15_02-27-23 - position ctrl; first try; couldsn't find stable walking version (shaking and friction too low?); quick okay solution gets worse then
quadruped_vail_unitreeA1_only_states_2022-12-15_11-29-47 - position; forgot to cut out init stepping
quadruped_vail_unitreeA1_only_states_2022-12-20_22-27-17 - torque control; good results but diverges/gets worse -> makes mini steps; with 250k dataset
quadruped_vail_unitreeA1_only_states_2022-12-21_19-42-53 - torque ctrl with range of info_constraints and changed gamme =.99
    view picture/discord for more details
quadruped_vail_unitreeA1_only_states_position_2022-12-21_19-45-30 - position ctrl with default info_constraint 
    shatering like before
quadruped_vail_unitreeA1_only_states_2022-12-22_22-12-17 - MIST - vergessen conda activate !!!!!!!!!!
quadruped_vail_unitreeA1_only_states_2022-12-22_23-43-19 - BUT WITH 15 SAMPLES: torque ctrl with range of info_constraints and changed gamme =.99
    info_constraint 0.01: tend to make mini steps at the end; some agents are good at the very first beginning, but all getting worse/more unsecure
    info_constraint 0.1: same like 0.01; ministeps/vibrating feet even worse
    info_constraint 0.5: same but every agent is good at the beginnin; but is getting wrse a the end/not so bad mini steps
    
quadruped_gail_unitreeA1_only_states_2022-12-23_16-11-37 - gail only states with stricter has_fallen
much better than pictures seem. One seed is stable and gets more and more unstable but all the other mostly stable/a little double step with the fornt left feet at the end

quadruped_vail_unitreeA1_only_states_2022-12-27_16-47-07 - vail only states with torque controlwith 15 samples each with info_constraint 0.001 and 1 
Forgot to remove stricter has fallen!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    info_constraint 0.001: all relative good 2. agent, but all are getting worse -> in the end vibrating (in the middle stumble/mini steps)
    info_constraint 1.0: in the beginning (2.) really good -> getting worse but less worse than 0.001; only stumbles a litlle bit and sometimes small steps
    
quadruped_gail_unitreeA1_only_states_2022-12-27_18-58-37 - gail only states with position  with default xml (loer gain)
    Forgot to remove stricter has fallen !!!!!!!!!!!!!!!!!!!!!!
    still very shaky legs/not a nice gait. Gets a little better to the end but definitive not good

quadruped_gail_unitreeA1_2023-01-01_18-03-41 - gail with optimal torques from data model with normal has_fallen
    makes lots of double steps/unstable walk

quadruped_gail_unitreeA1_only_states_2023-01-02_02-28-16 - gail only states with position  with default xml (loer gain)
    no good gait; very unstable

quadruped_vail_unitreeA1_only_states_2023-01-02_02-32-15 - vail only states with torque controlwith 15 samples each with info_constraint 0.001 and 1 
    info_constraint 0.001 most ok in the beginning (only a little double steps/limbing) but getting worse quick; but nearly all limbing in the beginning
    info_constraint 1.0 all in the beginning okay/good; but not as good as I remember with stricter has_fallen; getting quick worse -> starting to limbing; a few in the beginning perfect
        most also limb in the beginning a little bit
    but cannot compare earlier agents cause I store only 4 per seed; and these are the ones that are good with stricter has_fallen
    




quadruped_gail_unitreeA1_2023-01-03_02-43-04 - ICH GLAUBE/NICHT SICHER: gail with kp torques with reset at every step with normal has_fallen
    okay in the beginning but starts making small steps/double steps really fast


quadruped_gail_unitreeA1_only_states_2023-01-10_16-51-33 - gail with 2D walk, normal has_fallen. 8 dir a 50k dataset; multidm obs spec
quadruped_vail_unitreeA1_only_states_2023-01-10_16-44-04 - vail info_constraint=1 with 2D walk, normal has_fallen. 8 dir a 50k dataset; multidm obs spec

quadruped_vail_unitreeA1_only_states_2023-01-11_00-12-28 - vail only states torque with info_constraint 1.0 and less strict has_fallen; one sample to epoch 150 to test differences between has_fallen; tores as many agents as first run with less strict has fallen -> test second stred agent
    seems less stable than with stricter has fallen; slight limbing
    
    
    
quadruped_gail_unitreeA1_only_states_2023-01-19_01-11-20 - new stricter has fallen, updated dataset/correct direction arrow and adjusted traj with finetuned angles, troque only states gail
quadruped_vail_unitreeA1_only_states_2023-01-19_01-22-09 - new stricter has fallen, updated dataset/correct direction arrow and adjusted traj with finetuned angles, troque only states vail 
    problem: interpolation and draw only examples/traj from the first half -> backward walking


still npc not perfect-> slightly goes to the side
quadruped_gail_unitreeA1_only_states_2023-02-06_01-43-39 - gail, torque; new one long traj file; velo as goal, fixed npc model/dataset, interpolation .......
    gail is really really good in forward and backward and doesn't get worse over time but has the same problems with sideways and diagonal
quadruped_vail_unitreeA1_only_states_2023-02-06_01-45-29 - vail info_ =1 torque; new one long traj file; velo as goal, fixed npc model/dataset, interpolation .......
    I found epochs in vail where the forward and backward walking is okay but the sideways/diagonal walking isn't really a gait. Either it just jumps on the same place, it just walks for/backwards or it's just lifting the legs but setting it down again. From my point of view it seems like the has_fallen is to strict and when lifting the legs it tilts a little bit so it wants to fix that immediatly. But that would explain only one of the three effects


quadruped_gail_unitreeA1_only_states_2023-02-10_00-26-25 - gail same as aboe but with a dataset only sidewalking
    problem: wrong reward for logging/storing agent
    
    
quadruped_gail_unitreeA1_only_states_2023-02-12_17-18-30 - again only sidewalking but with new reward and better dataset
    seems to work
    

quadruped_gail_unitreeA1_only_states_2023-02-13_21-54-24 - all directions; updated reward function, better dataset with substeps in mpc model; new stricter has fallen
    takes longer to learn - ab 200 good-okay; manchmal verlent es auch wieder ab 300 -> vllt fängt danach wieder?

quadruped_vail_unitreeA1_only_states_2023-02-13_22-00-37 - vail with info 1; all directions; updated reward function, better dataset with substeps in mpc model; new stricter has fallen
    lern schnell aber verlernt auch schnell; wirklich gut läufe ca 10-200- scheint besser als gail
    
    
    
quadruped_vail_unitreeA1_only_states_2023-02-24_20-18-47 - new dataset with next steps calc only with desired velo;; rotate dataset with random angle
quadruped_gail_unitreeA1_only_states_2023-02-24_20-21-08 - new dataset with next steps calc only with desired velo;; rotate dataset with random angle

quadruped_gail_unitreeA1_only_states_2023-02-25_21-24-40 - new test run nothing changed except output vars if error#

quadruped_gail_unitreeA1_only_states_2023-03-01_20-03-29 - fit adapted: rotation also for next_states

quadruped_gail_unitreeA1_only_states_2023-03-02_01-15-14 - add different rot for each sample in fit_discriminator
same error

quadruped_gail_unitreeA1_only_states_2023-03-07_19-01-10 - fixed error with dir arrow and fixed dataset for gail

quadruped_gail_unitreeA1_only_states_2023-03-08_14-33-17 - same but in fit discriminator again only fixed random rotation not per step
same error

quadruped_gail_unitreeA1_only_states_2023-03-09_20-26-25 - same but m,ore logging points on critical places to catch error (probably decrease learning rate) 

quadruped_gail_unitreeA1_only_states_2023-03-11_15-00-11 - neu prints: aus shs und maxkl wenn lm nan is

quadruped_gail_unitreeA1_only_states_2023-03-12_19-29-22 - neue prints bei/um fisher matrix und mit genauerem traceback

quadruped_gail_unitreeA1_only_states_2023-03-13_23-22-45 - glaube fehler liegt bei gradient; aktueller stand conjugate gradient erstes mal fehler geworfen, aber gradient sagt davor nichts von nan 
    fehler zeile gefunden in compute gradient v= ... wahrscheinlich teilen durhc null weil gradient mist ist
    
quadruped_gail_unitreeA1_only_states_2023-03-14_16-49-55 - ohne rotation des datensets - test ob rest in ordnung

quadruped_gail_unitreeA1_only_states_2023-03-17_16-38-07 - neue rotate modified obs: begrenzung von velo rausgenommen

quadruped_gail_unitreeA1_only_states_2023-03-18_21-15-05 - nochmal neu begrenzungen von velo raus und in discriminator ein fester winkel
quadruped_vail_unitreeA1_only_states_2023-03-19_00-38-08 - same for vail

quadruped_gail_unitreeA1_only_states_2023-03-19_12-06-47 - same aber mit einem konstanten winkel rotation/nicht zufällig

quadruped_gail_unitreeA1_only_states_2023-03-20_01-19-26 - random zwischen zwei winkeln 0,pi/2; fit discrim für jeden unterschiedlichen winkel zwischen den beiden werten(choice); rotation von policy datensatz in setup in unitree geschoben -> weniger complex (2winkel) und policy daten kommen aus tatsächicher situatuion

quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25 - same aber mit winkel zwischen 0 und 2pi
    really good after epoch 250 - not sure how to increase it
quadruped_gail_unitreeA1_only_states_2023-03-20_13-08-13 - only one seed
quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54 - same mit 3 seeds
    not good; unstable-> seem like behavior is too complex
    wrong reward

- - - - - - - - --  -- - - - - - - -- - - - - - - - -  FALSCHES DATASET - - - - - - - - - - - - - - - - - - - - - - - - -
quadruped_gail_unitreeA1_only_states_2023-03-22_20-30-41 - new learning rates (5e-05, 2.5e-5) and n_eval_episodes to 25
quadruped_vail_unitreeA1_only_states_2023-03-22_20-42-39 - new learning rates (2.5e-05, 1.25e-5) and n_eval_episodes to 25
    both unstable
    wrong reward

quadruped_vail_unitreeA1_only_states_2023-03-23_18-14-50 - with info constraint 0.1, 0.5 and normal lrs
quadruped_gail_unitreeA1_only_states_2023-03-23_18-32-58 - nromal lrc but decreased lrD
    wrong reward


quadruped_vail_unitreeA1_only_states_2023-03-25_15-02-21 - info 0.1 - and new reward function - pls stores better agents
quadruped_gail_unitreeA1_only_states_2023-03-25_18-58-56 - with normal lrs


quadruped_vail_unitreeA1_only_states_2023-03-26_22-58-19 - half learning rates and info constraint 0.1, 0.01

quadruped_gail_unitreeA1_only_states_2023-03-28_19-37-59 - gail with best conditions but more seeds normal lr, torque etc
- - - - - - - - --  -- - - - - - - -- - - - - - - - -  FALSCHES DATASET - - - - - - - - - - - - - - - - - - - - - - - - -
Gail:
    letzte laufen mit richtigem reward hat sieht instabiler aus als der erste lauf bevor allen änderungen obwohl gleiche parameter
    auswerten noch
    
Vail:
    kp
    gerade laufen weniger lr mit verschiedenen info & normal mit info 0.1
    warten was kommt
    lernt sehr langsam mit geringeren lrs - vllt nur lrd niedriger
    vllt neuen run mit normalen parametern (glaube nicht sinnvoll)
    mit normalen lr lernt zu schnell: kann sofot stehen aber schritte erst irgendwann danach und auch nicht richtige richtung etc


jobs ür thesis: bis 300
    torque only states
    position only states
    torque action states
    torque only states (ablation)
    torque only states ohne rotation
    torque only states rotation
    
- - - - - - - - --  -- - - - - - - -- - - - - - - - -  FALSCHES DATASET - - - - - - - - - - - - - - - - - - - - - - - - -
quadruped_gail_unitreeA1_only_states_2023-03-27_17-15-58 torque only states, 5 seed, 300 epochs straight walking
quadruped_vail_unitreeA1_only_states_2023-03-27_17-16-15

quadruped_gail_unitreeA1_only_states_2023-03-27_17-18-31 position only states, 5 seed, 300 epochs straight walking
quadruped_vail_unitreeA1_only_states_2023-03-27_17-18-40

quadruped_gail_unitreeA1_2023-03-28_19-24-48
quadruped_vail_unitreeA1_2023-03-28_19-28-37 - torque action state 5 seed 300 epoch
- - - - - - - - --  -- - - - - - - -- - - - - - - - -  FALSCHES DATASET - - - - - - - - - - - - - - - - - - - - - - - - -

DRAN DENKEN ROTATION IN IMITTION WIEDER AN MACHEN
    
    
Thesis Jobs:
    quadruped_gail_unitreeA1_only_states_2023-03-29_01-47-29
    quadruped_vail_unitreeA1_only_states_2023-03-29_02-00-05 - only states, torque, straight
    
    Fehler in gail gemacht: vergessen zu pullen
    quadruped_gail_unitreeA1_only_states_2023-03-31_20-13-06
    quadruped_vail_unitreeA1_only_states_2023-03-29_02-01-57 - only states, position, straight
    
    
    quadruped_gail_unitreeA1_2023-03-29_02-10-23
    quadruped_vail_unitreeA1_2023-03-29_02-10-35 - state action, torque, straight
    
    quadruped_gail_unitreeA1_only_states_2023-03-29_02-16-14
    quadruped_vail_unitreeA1_only_states_2023-03-29_02-16-27 - only state, torque, 8 directions
    
    quadruped_gail_unitreeA1_only_states_2023-03-29_02-20-56
    quadruped_vail_unitreeA1_only_states_2023-03-29_02-21-04 - only state, torque, 8 directions, rotation, (info_constraint 1 and 0.1)
    
    quadruped_vail_unitreeA1_only_states_2023-03-30_15-48-13 - advanced vail info =1 and double lrD: 1e-4
    quadruped_vail_unitreeA1_only_states_2023-03-30_15-50-24 - normal lr 1e-4,5e-5 but info constraint 10
    quadruped_vail_unitreeA1_only_states_2023-04-03_14-38-12 - info 1 but lr 5e-4 and 2.5e-5
    quadruped_vail_unitreeA1_only_states_2023-04-03_14-39-49 - info 1 but lr 1e-4 and 2.5e-5
    
"""



