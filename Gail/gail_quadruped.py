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
from mushroom_rl.environments.mujoco_envs.quadrupeds.unitreeA1 import interpolate_remap, interpolate_map, reward_callback

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.utils.callbacks import PlotDataset

from mushroom_rl_imitation.imitation import GAIL_TRPO
from mushroom_rl_imitation.utils import FullyConnectedNetwork, DiscriminatorNetwork, NormcInitializer,\
    Standardizer, GailDiscriminatorLoss
from mushroom_rl_imitation.utils import BestAgentSaver


from experiment_launcher import run_experiment

from collections import defaultdict


def _create_gail_agent(mdp, expert_data, use_cuda, discrim_obs_mask, disc_only_state=True,
                       train_D_n_th_epoch=3, lrc=1e-3, lrD=0.0003, sw=None, policy_entr_coef=0.0,
                       use_noisy_targets=False, last_policy_activation="identity", use_next_states=True):

    mdp_info = deepcopy(mdp.info)

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=(49, ),
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
                         input_shape=(49, ),
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

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



def experiment(states_data_path: str,
               action_data_path: str = None,
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               horizon: int = 1000,
               gamma: float = 0.99,
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

    assert discr_only_state == (action_data_path is None)
    # either only states or with actions


    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))

    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500    # hz, added here as a reminder
    desired_contr_freq = 100     # hz
    n_substeps = env_freq // desired_contr_freq

    traj_params = dict(traj_path=states_data_path,
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))

    # defines transformation needed for interpolation over rotation matrix
    if use_2d_ctrl:
        traj_params["interpolate_map"] = interpolate_map  # transforms 9dim rot matrix into one rot angle
        traj_params["interpolate_remap"] = interpolate_remap  # and back

    # create the environment
    mdp = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    traj_params=traj_params, random_start=True, setup_random_rot=True,
                    use_torque_ctrl=use_torque_ctrl, use_2d_ctrl=use_2d_ctrl, tmp_dir_name=tmp_dir_name,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    #create dataset
    expert_data = mdp.create_dataset(actions_path=action_data_path, only_state=discr_only_state,
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

    #plot_data_callbacks = PlotDataset(mdp.info) # for error finding purposes
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


