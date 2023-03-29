import os
from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster


if __name__ == '__main__':
    LOCAL = bool_local_cluster()
    TEST = False
    USE_CUDA = False

    JOBLIB_PARALLEL_JOBS = 1  # or os.cpu_count() to use all cores
    N_SEEDS = 5

    launcher = Launcher(exp_name='quadruped_vail_unitreeA1',
                        python_file='vail_quadruped',
                        partition="amd2,amd",
                        n_exps=N_SEEDS,
                        n_cores=6,
                        memory_per_core=1000,
                        days=3,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(states_data_path='../data/states_2023_02_23_19_48_33_straight.npz',
                          action_data_path='../data/actions_torque_2023_02_23_19_48_33_straight.npz',
                          n_epochs=300,
                          n_steps_per_epoch=100000,
                          n_epochs_save=50,
                          n_eval_episodes=25,
                          n_steps_per_fit=1000,
                          use_next_states=False,
                          discr_only_state=False,
                          use_cuda=USE_CUDA,
                          use_torque_ctrl=True,
                          use_2d_ctrl=True,
                          tmp_dir_name=".",
                          setup_random_rot=False)

    lrs = [(1e-4, 5e-5)]
    d_delays = [3]
    plcy_ent_coefs = [1e-3]
    info_constraints = [1]
    use_noisy_targets = [0]
    lpa = ["identity"]
    use_next_states = [1]
    horizons = [1000]
    gammas = [0.99]

    for lr, d, p_ent_coef, use_nt, last_pa, info_constraint, horizon, gamma in product(lrs, d_delays, plcy_ent_coefs,
                                                                                       use_noisy_targets, lpa,
                                                                                       info_constraints, horizons,
                                                                                       gammas):
        lrc, lrD = lr
        launcher.add_experiment(train_D_n_th_epoch__=d,
                                policy_entr_coef=p_ent_coef, last_policy_activation=last_pa, lrc=lrc, lrD__=lrD,
                                use_noisy_targets__=use_nt, info_constraint__=info_constraint,
                                horizon__=horizon, gamma__=gamma, **default_params)

    launcher.run(LOCAL, TEST)
