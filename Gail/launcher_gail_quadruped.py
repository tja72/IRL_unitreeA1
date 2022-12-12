import os
from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster


if __name__ == '__main__':
    LOCAL = bool_local_cluster()
    TEST = False
    USE_CUDA = False

    JOBLIB_PARALLEL_JOBS = 1  # or os.cpu_count() to use all cores
    N_SEEDS = 3

    launcher = Launcher(exp_name='quadruped_gail_unitreeA1',
                        python_file='gail_quadruped',
                        n_exps=N_SEEDS,
                        #joblib_n_jobs=JOBLIB_PARALLEL_JOBS, n_cores=JOBLIB_PARALLEL_JOBS * 1, memory=JOBLIB_PARALLEL_JOBS * 1000,
                        n_cores=6,
                        memory_per_core=500, #not in other file
                        days=4,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        ) #partition, conda_env, use_underscore_argparse

    default_params = dict(n_epochs=500,
                          n_steps_per_epoch=100000,
                          n_epochs_save=50,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          expert_data_path="../data/dataset_unitreeA1_IRL.npz",
                          init_data_path="../data/dataset_only_states_unitreeA1_IRL.npz",
                          use_next_states=False,
                          use_cuda=USE_CUDA,
                          discr_only_state=False) #changed
    #not use_next_states, n_epochs_save but has discr_only_state

    lrs = [(1e-4, 5e-5)]
    d_delays = [3]
    plcy_ent_coefs = [1e-3]
    use_noisy_targets = [0]
    lpa = ["identity"]
    use_next_states = [1]
    horizons = [1000]
    gammas = [0.99]

    for lr, d, p_ent_coef, use_nt, last_pa, horizon, gamma in product(lrs, d_delays, plcy_ent_coefs,
                                                                      use_noisy_targets, lpa, horizons,
                                                                      gammas):
        lrc, lrD = lr
        launcher.add_experiment(train_D_n_th_epoch__=d,
                                policy_entr_coef=p_ent_coef, last_policy_activation=last_pa, lrc=lrc, lrD__=lrD,
                                use_noisy_targets__=use_nt, horizon__=horizon, gamma__=gamma, **default_params)

    launcher.run(LOCAL, TEST)