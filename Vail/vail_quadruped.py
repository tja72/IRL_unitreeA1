import os
from copy import deepcopy

from time import perf_counter
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Core
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.environments.mujoco_envs.humanoids import Trajectory

from mushroom_rl_imitation.imitation import VAIL_TRPO
from mushroom_rl_imitation.utils import FullyConnectedNetwork, NormcInitializer, Standardizer, VariationalNet, VDBLoss
from mushroom_rl_imitation.utils import BestAgentSaver

from experiment_launcher import run_experiment
from collections import defaultdict



def _create_vail_agent(mdp, expert_data, use_cuda, discrim_obs_mask, disc_only_state=True, info_constraint=0.5,
                       train_D_n_th_epoch=3, lrc=1e-3, lrD=0.0003, sw=None, policy_entr_coef=0.0,
                       use_noisy_targets=False, last_policy_activation="identity", use_next_states=True):

    mdp_info = deepcopy(mdp.info)

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=mdp_info.observation_space.shape,
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
                         input_shape=mdp_info.observation_space.shape,
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    # remove hip rotations
    #assert disc_only_state, ValueError("This configuration file does not support actions for the discriminator")
    discrim_act_mask = [] if disc_only_state else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (2 * (len(discrim_obs_mask)+len(discrim_act_mask)),) if use_next_states else (len(discrim_obs_mask)+len(discrim_act_mask),)
    discrim_standardizer = Standardizer()
    z_size = 128
    encoder_net = FullyConnectedNetwork(input_shape=discrim_input_shape, output_shape=(128,), n_features=[256],
                                        activations=['relu', 'relu'], standardizer=None,
                                        squeeze_out=False, use_cuda=use_cuda)
    decoder_net = FullyConnectedNetwork(input_shape=(z_size,), output_shape=(1,), n_features=[],
                                        # no features mean no hidden layer -> one layer
                                        activations=['identity'], standardizer=None,
                                        initializers=[NormcInitializer(std=0.1)],
                                        squeeze_out=False, use_cuda=use_cuda)

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': lrD,
                                                      'weight_decay': 0.0}},
                                batch_size=2048,
                                network=VariationalNet,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                z_size=z_size,
                                encoder_net=encoder_net,
                                decoder_net=decoder_net,
                                use_next_states=use_next_states,
                                use_actions=not disc_only_state,
                                standardizer=discrim_standardizer,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_D_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=25,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=VDBLoss(info_constraint=info_constraint, lr_beta=0.00001),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=5e-3,
                      use_next_states=use_next_states)

    agent = VAIL_TRPO(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
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
               info_constraint: float = 0.1,
               policy_entr_coef: float = 1e-3,
               train_D_n_th_epoch: int = 3,
               lrc: float = 1e-3,
               lrD: float = 0.0003,
               use_foot_forces: bool = True,
               last_policy_activation: str = "identity",
               use_noisy_targets: bool = False,
               use_next_states: bool = False,
               use_cuda: bool = False,
               results_dir: str = './logs',
               seed: int = 0):

    if discr_only_state:
        action_data_path = None
        states_data_path = ["../data/dataset_only_states_unitreeA1_IRL_optimal_0.npz", "../data/dataset_only_states_unitreeA1_IRL_optimal_1.npz", "../data/dataset_only_states_unitreeA1_IRL_optimal_2.npz", "../data/dataset_only_states_unitreeA1_IRL_optimal_3.npz", "../data/dataset_only_states_unitreeA1_IRL_optimal_4.npz"]
    else:
        action_data_path = ["../data/dataset_unitreeA1_IRL_0.npz",
                            "../data/dataset_unitreeA1_IRL_1.npz",
                            "../data/dataset_unitreeA1_IRL_2.npz",
                            "../data/dataset_unitreeA1_IRL_3.npz",
                            "../data/dataset_unitreeA1_IRL_4.npz"]
        states_data_path = ["../data/dataset_only_states_unitreeA1_IRL_0.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_1.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_2.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_3.npz",
                            "../data/dataset_only_states_unitreeA1_IRL_4.npz"]


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


    # prepare trajectory params
    if (type(states_data_path) == list):  # concatenate datasets and store for trajectory
        temp_states_dataset = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                               [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for path in states_data_path:
            trajectory_files = np.load(path, allow_pickle=True)
            trajectory_files = {k: d for k, d in trajectory_files.items()}
            trajectory = np.array([trajectory_files[key] for key in trajectory_files.keys()])
            assert len(temp_states_dataset) == len(trajectory)
            for i in np.arange(len(temp_states_dataset)):
                temp_states_dataset[i] = temp_states_dataset[i] + list(trajectory[i])
        np.savez(os.path.join('.', 'dataset_temp_concatenated_optimal_states.npz'),
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
        traj_params = dict(traj_path='./dataset_temp_concatenated_optimal_states.npz',
                           traj_dt=(1 / traj_data_freq),
                           control_dt=(1 / desired_contr_freq))
    else:
        traj_params = dict(traj_path=states_data_path,
                           traj_dt=(1 / traj_data_freq),
                           control_dt=(1 / desired_contr_freq))

    # set a reward for logging
    reward_callback = lambda state, action, next_state: np.exp(- np.square(state[16] - 0.6))  # x-velocity as reward

    # create the environment
    mdp = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    traj_params=traj_params, random_start=True,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    if discr_only_state:
        data_path = states_data_path
    else:
        data_path = action_data_path

    # create a dataset
    if (type(data_path) == list):
        temp_expert_data = defaultdict(lambda: [])
        for path in data_path:
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
    agent = _create_vail_agent(mdp=mdp, expert_data=expert_data, use_cuda=use_cuda, disc_only_state=discr_only_state,
                               info_constraint=info_constraint, train_D_n_th_epoch=train_D_n_th_epoch, lrc=lrc,
                               lrD=lrD, sw=tb_writer, policy_entr_coef=policy_entr_coef,
                               use_noisy_targets=use_noisy_targets, use_next_states=use_next_states,
                               last_policy_activation=last_policy_activation, discrim_obs_mask=discrim_obs_mask)
    core = Core(mdp=mdp, agent=agent)


    # gail train loop
    for epoch in range(n_epochs):
        with catchtime() as t:
            core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False)
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            J_mean = np.mean(compute_J(dataset))
            tb_writer.add_scalar("Eval_J", J_mean, epoch)
            agent_saver.save(core.agent, J_mean)
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
            agent_saver.save(core.agent, R_mean)

    agent_saver.save_curr_best_agent()
    print("Finished.")

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    run_experiment(experiment)
