import os
from copy import deepcopy
import sys

from time import perf_counter
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Core
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.environments.mujoco_envs.quadrupeds.unitreeA1 import interpolate_remap, interpolate_map

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.utils.callbacks import PlotDataset

from mushroom_rl_imitation.imitation import GAIL_TRPO
from mushroom_rl_imitation.utils import FullyConnectedNetwork, DiscriminatorNetwork, NormcInitializer,\
    Standardizer, GailDiscriminatorLoss
from mushroom_rl_imitation.utils import BestAgentSaver
from mushroom_rl_imitation.utils import behavioral_cloning, prepare_expert_data


from experiment_launcher import run_experiment

from collections import defaultdict


def _create_gail_agent(mdp, expert_data, use_cuda, discrim_obs_mask, disc_only_state=True,
                       train_D_n_th_epoch=3, lrc=1e-3, lrD=0.0003, sw=None, policy_entr_coef=0.0,
                       use_noisy_targets=False, last_policy_activation="identity", use_next_states=True):

    mdp_info = deepcopy(mdp.info)

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=(len(mdp.obs_helper.observation_spec)+10, ),
                         output_shape=mdp_info.action_space.shape,
                         std_0=1.0,
                         n_features=[512, 256],
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class':  optim.Adam,
                                    'params': {'lr':           lrc,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=(len(mdp.obs_helper.observation_spec)+10, ),
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    # TODO adapt/need?: remove hip rotations ---------------------------------------------------------------------------
    #assert disc_only_state, ValueError("This configuration file does not support actions for the discriminator") ---changed
    discrim_act_mask = [] if disc_only_state else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (2 * (len(discrim_obs_mask)+len(discrim_act_mask)),) if use_next_states else (len(discrim_obs_mask)+len(discrim_act_mask),)
    discrim_standardizer = Standardizer()
    discriminator_params = dict(optimizer={'class':  optim.Adam,
                                           'params': {'lr':           lrD,
                                                      'weight_decay': 0.0}},
                                batch_size=2000,
                                network=DiscriminatorNetwork,
                                use_next_states=use_next_states,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                squeeze_out=False,
                                n_features=[512, 256],
                                initializers=None,
                                activations=['tanh', 'tanh', 'identity'],
                                standardizer=discrim_standardizer,
                                use_actions=False if disc_only_state else True,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_D_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=25,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=GailDiscriminatorLoss(entcoeff=1e-3),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=5e-3,
                      use_next_states=use_next_states)

    agent = GAIL_TRPO(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstrations=expert_data, **alg_params)
    return agent



def experiment(n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               horizon: int = 1000,
               gamma: float = 0.99,
               goal_data_path: str = None,
               discr_only_state: bool = True,
               policy_entr_coef: float = 1e-3,
               train_D_n_th_epoch: int = 3,
               lrc: float = 1e-3,
               lrD: float = 0.0003,
               last_policy_activation: str = "identity",
               use_noisy_targets: bool = False,
               use_next_states: bool = False,
               use_cuda: bool = False,
               results_dir: str = './logs',
               seed: int = 0,
               use_torque_ctrl: bool = False,
               use_2d_ctrl: bool = False,
               tmp_dir_name: str = "."):

    if(discr_only_state):
        action_data_path = None
        states_data_path = ['../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_backward_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_backward_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_backward_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_BL_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_BL_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_BL_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_BR_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_BR_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_BR_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_FL_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_FL_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_FL_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_forward_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_forward_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_forward_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_FR_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_FR_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_FR_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_left_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_left_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_left_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_right_noise1_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_right_noise2_optimal.npz',
                            '../data/2D_Walking/dataset_only_states_unitreeA1_IRL_50k_right_optimal.npz'
                            ]
        """
        states_data_path = ["../data/dataset_only_states_unitreeA1_IRL_optimal_0.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_optimal_1.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_optimal_2.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_optimal_3.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_optimal_4.npz"]
                            """
    else:
        # first after number is action then states type
        action_data_path = ["../data/dataset_unitreeA1_IRL_new_0_opt_kp.npz",
                            "../data/dataset_unitreeA1_IRL_new_1_opt_kp.npz",
                            "../data/dataset_unitreeA1_IRL_new_2_opt_kp.npz",
                            "../data/dataset_unitreeA1_IRL_new_3_opt_kp.npz",
                            "../data/dataset_unitreeA1_IRL_new_4_opt_kp.npz"]
        states_data_path = ["../data/dataset_only_states_unitreeA1_IRL_new2_0_optimal.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_new2_1_optimal.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_new2_2_optimal.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_new2_3_optimal.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_new2_4_optimal.npz"]

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))

    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz
    n_substeps = env_freq // desired_contr_freq    # env_freq / desired_contr_freq

    # set a reward for logging
    reward_callback = lambda state, action, next_state: np.exp(- np.square(state[16] - 0.6))  # x-velocity as reward
    if use_2d_ctrl: # velocity in direction arrow as reward
        """
        Explanation of one liner/reward below:
        velo2 = np.array([self._data.qvel[0], self._data.qvel[2]])
        rot_mat2 = np.dot(self._direction_xmat.reshape((3,3)), np.array([[0, 0, 1],[0, 1, 0],[1, 0, 0]]))
        direction2 = np.dot(rot_mat2, np.array([1000, 0, 0]))[:2]  # TODO here is something wrong
        reward3 = np.dot(velo2, direction2) / np.linalg.norm(direction2) -0.4
        """
        reward_callback = lambda state, action, next_state: np.exp(- np.square(
            np.dot(np.array([state[16], state[18]]),
                   np.dot(np.dot(state[34:43].reshape((3, 3)), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),
                          np.array([1000, 0, 0]))[:2])
            / np.linalg.norm(np.dot(np.dot(state[34:43].reshape((3, 3)), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),
                                    np.array([1000, 0, 0]))[:2]) - 0.4))

    # prepare trajectory params
    if(type(states_data_path)==list) : # concatenate datasets and store for trajectory
        temp_states_dataset = [list() for j in range(37)]

        for path in states_data_path:
            trajectory_files = np.load(path, allow_pickle=True)
            trajectory_files = {k: d for k, d in trajectory_files.items()}
            trajectory = np.array([trajectory_files[key].flatten() for key in trajectory_files.keys()], dtype=object)
            assert len(temp_states_dataset) == len(trajectory)
            for i in np.arange(len(temp_states_dataset)):
                temp_states_dataset[i] = temp_states_dataset[i] + list(trajectory[i])
        if use_2d_ctrl:
            np.savez(os.path.join('.', 'dataset_temp_concatenated_optimal_states'+str(seed)+'.npz'),
                     q_trunk_tx=np.array(temp_states_dataset[0]),
                     q_trunk_ty=np.array(temp_states_dataset[1]),
                     q_trunk_tz=np.array(temp_states_dataset[2]),
                     q_trunk_tilt=np.array(temp_states_dataset[3]),
                     q_trunk_list=np.array(temp_states_dataset[4]),
                     q_trunk_rotation=np.array(temp_states_dataset[5]),
                     q_FR_hip_joint=np.array(temp_states_dataset[6]),
                     q_FR_thigh_joint=np.array(temp_states_dataset[7]),
                     q_FR_calf_joint=np.array(temp_states_dataset[8]),
                     q_FL_hip_joint=np.array(temp_states_dataset[9]),
                     q_FL_thigh_joint=np.array(temp_states_dataset[10]),
                     q_FL_calf_joint=np.array(temp_states_dataset[11]),
                     q_RR_hip_joint=np.array(temp_states_dataset[12]),
                     q_RR_thigh_joint=np.array(temp_states_dataset[13]),
                     q_RR_calf_joint=np.array(temp_states_dataset[14]),
                     q_RL_hip_joint=np.array(temp_states_dataset[15]),
                     q_RL_thigh_joint=np.array(temp_states_dataset[16]),
                     q_RL_calf_joint=np.array(temp_states_dataset[17]),
                     dq_trunk_tx=np.array(temp_states_dataset[18]),
                     dq_trunk_tz=np.array(temp_states_dataset[19]),
                     dq_trunk_ty=np.array(temp_states_dataset[20]),
                     dq_trunk_tilt=np.array(temp_states_dataset[21]),
                     dq_trunk_list=np.array(temp_states_dataset[22]),
                     dq_trunk_rotation=np.array(temp_states_dataset[23]),
                     dq_FR_hip_joint=np.array(temp_states_dataset[24]),
                     dq_FR_thigh_joint=np.array(temp_states_dataset[25]),
                     dq_FR_calf_joint=np.array(temp_states_dataset[26]),
                     dq_FL_hip_joint=np.array(temp_states_dataset[27]),
                     dq_FL_thigh_joint=np.array(temp_states_dataset[28]),
                     dq_FL_calf_joint=np.array(temp_states_dataset[29]),
                     dq_RR_hip_joint=np.array(temp_states_dataset[30]),
                     dq_RR_thigh_joint=np.array(temp_states_dataset[31]),
                     dq_RR_calf_joint=np.array(temp_states_dataset[32]),
                     dq_RL_hip_joint=np.array(temp_states_dataset[33]),
                     dq_RL_thigh_joint=np.array(temp_states_dataset[34]),
                     dq_RL_calf_joint=np.array(temp_states_dataset[35]),
                     dir_arrow=np.array(temp_states_dataset[36]))
        else:
            np.savez(os.path.join('.', 'dataset_temp_concatenated_optimal_states'+str(seed)+'.npz'),
                     q_trunk_tx=np.array(temp_states_dataset[0]),
                     q_trunk_ty=np.array(temp_states_dataset[1]),
                     q_trunk_tz=np.array(temp_states_dataset[2]),
                     q_trunk_tilt=np.array(temp_states_dataset[3]),
                     q_trunk_list=np.array(temp_states_dataset[4]),
                     q_trunk_rotation=np.array(temp_states_dataset[5]),
                     q_FR_hip_joint=np.array(temp_states_dataset[6]),
                     q_FR_thigh_joint=np.array(temp_states_dataset[7]),
                     q_FR_calf_joint=np.array(temp_states_dataset[8]),
                     q_FL_hip_joint=np.array(temp_states_dataset[9]),
                     q_FL_thigh_joint=np.array(temp_states_dataset[10]),
                     q_FL_calf_joint=np.array(temp_states_dataset[11]),
                     q_RR_hip_joint=np.array(temp_states_dataset[12]),
                     q_RR_thigh_joint=np.array(temp_states_dataset[13]),
                     q_RR_calf_joint=np.array(temp_states_dataset[14]),
                     q_RL_hip_joint=np.array(temp_states_dataset[15]),
                     q_RL_thigh_joint=np.array(temp_states_dataset[16]),
                     q_RL_calf_joint=np.array(temp_states_dataset[17]),
                     dq_trunk_tx=np.array(temp_states_dataset[18]),
                     dq_trunk_tz=np.array(temp_states_dataset[19]),
                     dq_trunk_ty=np.array(temp_states_dataset[20]),
                     dq_trunk_tilt=np.array(temp_states_dataset[21]),
                     dq_trunk_list=np.array(temp_states_dataset[22]),
                     dq_trunk_rotation=np.array(temp_states_dataset[23]),
                     dq_FR_hip_joint=np.array(temp_states_dataset[24]),
                     dq_FR_thigh_joint=np.array(temp_states_dataset[25]),
                     dq_FR_calf_joint=np.array(temp_states_dataset[26]),
                     dq_FL_hip_joint=np.array(temp_states_dataset[27]),
                     dq_FL_thigh_joint=np.array(temp_states_dataset[28]),
                     dq_FL_calf_joint=np.array(temp_states_dataset[29]),
                     dq_RR_hip_joint=np.array(temp_states_dataset[30]),
                     dq_RR_thigh_joint=np.array(temp_states_dataset[31]),
                     dq_RR_calf_joint=np.array(temp_states_dataset[32]),
                     dq_RL_hip_joint=np.array(temp_states_dataset[33]),
                     dq_RL_thigh_joint=np.array(temp_states_dataset[34]),
                     dq_RL_calf_joint=np.array(temp_states_dataset[35]))
        traj_params = dict(traj_path='./dataset_temp_concatenated_optimal_states'+str(seed)+'.npz',
                           traj_dt=(1 / traj_data_freq),
                           control_dt=(1 / desired_contr_freq))
    else:
        traj_params = dict(traj_path=states_data_path,
                           traj_dt=(1 / traj_data_freq),
                           control_dt=(1 / desired_contr_freq))

        # how to transform the samples/trajectories for interpolation -> get into oine dim; interpolate; retransform

    def interpolate_map(traj):
        traj_list = [list() for j in range(len(traj))]
        for i in range(len(traj_list)):
            if i in [3, 4, 5]:
                traj_list[i] = list(np.unwrap(traj[i]))
            else:
                traj_list[i] = list(traj[i])
        temp = []
        traj_list[36] = list(np.unwrap([
            np.arctan2(np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[3],
                       np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[0])
            for mat in traj[36]]))
        # for mat in traj[36].reshape((len(traj[0]), 9)):
        #    arrow = np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
        #   temp.append(np.arctan2(arrow[3], arrow[0]))
        # traj_list[36] = temp
        return np.array(traj_list)



    if use_2d_ctrl:
        traj_params["interpolate_map"] = interpolate_map  # transforms 9dim rot matrix into one rot angle
        traj_params["interpolate_remap"] = interpolate_remap  # and back



    # create the environment
    mdp = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    traj_params=traj_params, random_start=True, use_torque_ctrl=use_torque_ctrl,
                    use_2d_ctrl=use_2d_ctrl, tmp_dir_name=tmp_dir_name,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    if discr_only_state:
        data_path = states_data_path
    else:
        data_path = action_data_path

    # create a dataset
    if(type(data_path) == list): # call create dataset for every dataset in the states list ad concatenate
        temp_expert_data = defaultdict(lambda: [])
        for path in data_path:
            print(path)
            temp_data = mdp.create_dataset(data_path=path, only_state=discr_only_state,
                                           ignore_keys=["q_trunk_tx", "q_trunk_ty"], use_next_states=use_next_states)
            for key in temp_data:
                temp_expert_data[key] = temp_expert_data[key] + list(temp_data[key])
        expert_data = dict()
        for key in temp_expert_data:
            expert_data[key] = np.array(temp_expert_data[key])
    else:
        expert_data = mdp.create_dataset(data_path=data_path, only_state=discr_only_state,
                                         ignore_keys=["q_trunk_tx", "q_trunk_ty"], use_next_states=use_next_states)



    discrim_obs_mask = np.arange(expert_data["states"].shape[1])
    # logging stuff
    tb_writer = SummaryWriter(log_dir=results_dir)
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    # create agent and core
    agent = _create_gail_agent(mdp=mdp, expert_data=expert_data, use_cuda=use_cuda, disc_only_state=discr_only_state,
                               train_D_n_th_epoch=train_D_n_th_epoch, lrc=lrc,
                               lrD=lrD, sw=tb_writer, policy_entr_coef=policy_entr_coef,
                               use_noisy_targets=use_noisy_targets, use_next_states=use_next_states,
                               last_policy_activation=last_policy_activation, discrim_obs_mask=discrim_obs_mask)

    #plot_data_callbacks = PlotDataset(mdp.info)
    core = Core(agent, mdp)#, callback_step=plot_data_callbacks)

    print("Starting Training")
    # gail train loop
    for epoch in range(n_epochs):
        with catchtime() as t:
            core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False)
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            tb_writer.add_scalar("Eval_R", R_mean, epoch)
            agent_saver.save(core.agent, R_mean)
            print('Epoch %d | Time %fs ' % (epoch + 1, float(t())))

            # evaluate with deterministic policy
            core.agent.policy.deterministic = True
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            logger_deter.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
            tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-deterministic", L, epoch)
            core.agent.policy.deterministic = False

            # evaluate with stochastic policy
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            logger_stoch.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
            tb_writer.add_scalar("Eval_R-stochastic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-stochastic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-stochastic", L, epoch)
            #agent_saver.save(core.agent, R_mean)

    agent_saver.save_curr_best_agent()
    print("Finished.")

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    run_experiment(experiment)


    """
    Questions:
    
    
    

    continue gail and not RL#
    cluster
    
    trajectory stepping in one position -> bad for data?
    
    error in dataset: compute_episodes_length if lengths=[]
    ignore x,y
    imports smarter than now
    what else in mushroom_irl state than observations
    -> whats in state
    
    Did:
    tuned values in xml
    extend gail implementation to use actions
    -> error
    method to replay/restore learned agent



------------------------------------------------------------------------------------------

    why do I need absorbing/reward in the dataset
    difference between using traj_param in mdp and using only prepare_expert data (in respect to absorbing/reward...)
    Where Gail uses absorbing/reward
    do I need a normalizer?
    states and actions make sense to interpolate, absorbing and rewards don't
    
    continue fine tuning xml
    
    
    
    
    Info traj:params used in core.learn draws random init point -> where it uses reward
    """


