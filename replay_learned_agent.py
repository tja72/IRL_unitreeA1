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

    gamma = 0.99
    horizon = 1000

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq

    use_torque_ctrl = True
    use_2d_ctrl = True
    setup_random_rot = True

    # prepare trajectory params
    traj_params = dict(traj_path='./data/states_2023_02_23_19_48_33.npz',
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq),
                       interpolate_map=interpolate_map,
                       interpolate_remap=interpolate_remap)

    # create the environment
    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=use_torque_ctrl,
                    traj_params=traj_params, random_start=True,# init_step_no=0, init_traj_no=0
                    use_2d_ctrl=use_2d_ctrl, tmp_dir_name='.', setup_random_rot=setup_random_rot,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    agents = [ # GAIL
        #Serializable.load(
         #   '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_184_J_811.450205.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'), #zögerlich aber ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_177_J_766.393771.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_225_J_869.601562.msh'), # oft sehr unsicher/humpelt
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_195_J_868.019246.msh'), # auch ganz ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_250_J_851.637429.msh'),# auch ganz ok
    ]
    agents = [ # GAIL
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'),  # zögerlich aber ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'), #zögerlich aber ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'), #zögerlich aber ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'), #zögerlich aber ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'), #zögerlich aber ok
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-03-20_22-29-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_217_J_850.133653.msh'), #zögerlich aber ok
    ]
    """
    agents = [ # VAIL
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_35_J_848.501468.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_63_J_847.898821.msh'), # sehr unsicher/traut sich nicht zu gehen/ hebt hoch setzt sofort wieder hin
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_26_J_814.211735.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_85_J_816.831889.msh'), #same here wie bei allen anderen auch schon
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_39_J_829.566779.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_63_J_842.753146.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_190_J_802.761241.msh'),
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_179_J_774.394044.msh'),
    ]"""

     # good:             '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-02-13_22-00-37/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_145_J_974.749549.msh'),
    # good best            '/media/tim/929F-6E96/thesis/quadruped_vail_unitreeA1_only_states_2023-02-13_22-00-37/train_D_n_th_epoch___3/info_constraint___1/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/1/agent_epoch_200_J_973.474543.msh'),
    # good best            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-02-13_21-54-24/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/2/agent_epoch_221_J_982.343283.msh'),
    # good             '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-02-13_21-54-24/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_304_J_970.560817.msh'),
    #TODO core zeile 183 nicht auskommentieren !!!!!!!!!!!!!!!!!!!!!!!!!
    for i in range(len(agents)):
        print(agents[i])
        #plot_data_callbacks = PlotDataset(env.info)
        #core = Core(mdp=env, agent=agent, callback_step=plot_data_callbacks)
        core = Core(mdp=env, agent=agents[i])
        #core.agent.policy.deterministic = False
        dataset, env_info = core.evaluate(n_episodes=25, render=False, get_env_info=True)
        A = compute_J(dataset)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))

        print("J_mean: ", J_mean)
        print("R_mean: ", R_mean)
        print("L_mean:", L)


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
quadruped_gail_unitreeA1_only_states_2023-03-20_13-08-13 - only one seed
quadruped_vail_unitreeA1_only_states_2023-03-20_13-51-54 - same mit 3 seeds

quadruped_gail_unitreeA1_only_states_2023-03-22_19-38-43 - new learning rates (5e-05, 2.5e-5) and n_eval_episodes to 25
quadruped_vail_unitreeA1_only_states_2023-03-22_19-38-54 - new learning rates (2.5e-05, 1.25e-5) and n_eval_episodes to 25


"""
