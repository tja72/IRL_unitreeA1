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

    launcher = Launcher(exp_name='quadruped_vail_unitreeA1_only_states',
                        python_file='vail_quadruped',
                        partition="amd2,amd",
                        n_exps=N_SEEDS,
                        n_cores=6,
                        memory_per_core=500,
                        days=3,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=500,
                          n_steps_per_epoch=100000,
                          n_epochs_save=150 if N_SEEDS == 15 else 50,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_next_states=True,
                          discr_only_state=True,
                          use_cuda=USE_CUDA,
                          use_torque_ctrl=True)

    lrs = [(1e-4, 5e-5)]
    d_delays = [3]
    plcy_ent_coefs = [1e-3]
    use_noisy_targets = [0]
    lpa = ["identity"]
    use_next_states = [1]
    horizons = [1000]
    gammas = [0.99]
    info_constraints = [0.1, 0.5, 0.01]

    for lr, d, p_ent_coef, use_nt, last_pa, horizon, gamma, info_constraint in product(lrs, d_delays, plcy_ent_coefs,
                                                                      use_noisy_targets, lpa, horizons,
                                                                      gammas, info_constraints):
        lrc, lrD = lr
        launcher.add_experiment(train_D_n_th_epoch__=d, info_constraint__=info_constraint, #__ can change/will not overwrite
                                policy_entr_coef=p_ent_coef, last_policy_activation=last_pa, lrc=lrc, lrD__=lrD,
                                use_noisy_targets__=use_nt, horizon__=horizon, gamma__=gamma, **default_params)

    launcher.run(LOCAL, TEST)
