import numpy as np
import pandas as pd
import optuna

# import your simulation function + constants
from .droplet_2d import *


# ---------------------------------------
# OBJECTIVE FUNCTION
# ---------------------------------------

def compute_objective(rho, u, int_thick, rho_l, rho_g, interface_thickness_target):

    offset = nx // 6

    rho_north = rho[nx // 2, ny // 2 - offset, 0]
    rho_south = rho[nx // 2, ny // 2 + offset, 0]
    rho_west = rho[nx // 2 - offset, ny // 2, 0]
    rho_east = rho[nx // 2 + offset, ny // 2, 0]

    rho_g_pred = 0.25 * (rho_north + rho_south + rho_west + rho_east)
    rho_l_pred = rho[nx // 2, ny // 2, 0]

    err_l = abs((rho_l_pred - rho_l) / rho_l)
    err_g = abs((rho_g_pred - rho_g) / rho_g)
    err_int_th = abs((int_thick - interface_thickness_target) / interface_thickness_target)

    objective = err_l + err_g + err_int_th

    return objective, err_l, err_g, err_int_th

# ---------------------------------------
# OPTIMIZATION PER TEMPERATURE
# ---------------------------------------

def optimize_T(T_X, rho_l_local, rho_g_local, k_val_ini, A_val_ini, interface_thickness_target, k_rad, A_rad, n_trials=30):

    T = T_X * Tc

    trial_log = []

    study = optuna.create_study(direction="minimize")

    best_k = k_val_ini
    best_A = A_val_ini
    best_val = np.inf

    def objective_opt(trial):
        nonlocal best_k, best_A, best_val
        k_center = best_k
        A_center = best_A

        #k_val = trial.suggest_float("k", max(k_val_ini - 0.1, 0.005), k_val_ini * 5)
        #A_val = trial.suggest_float("A", max(A_val_ini - 0.25, 0.0), A_val_ini + 0.25)
        k_val = trial.suggest_float("k", max(k_center - k_rad, 0.005), k_center + k_rad)
        A_val = trial.suggest_float("A", max(A_center - A_rad, 0.0), A_center + A_rad)
        #k_val = trial.suggest_float("k", 0.0, 0.5)
        #A_val = trial.suggest_float("A", 0.0, 1.0)
        try:
            rho, u, int_thick = run_simulation(
                T_X,
                k_val,
                A_val,
                rho_l_local,
                rho_g_local,
                steps=5000   # shorter during optimization
            )

            val, err_l, err_g, err_int_th = compute_objective(rho, u, int_thick, rho_l_local, rho_g_local, interface_thickness_target)

            # store every trial
            trial_log.append({
                "X": T_X,
                "T": T,
                "k": k_val,
                "A": A_val,
                "rho_l": rho_l_local,
                "rho_g": rho_g_local,
                "objective": val,
                "err_l": err_l,
                "err_g": err_g,
                "err_int_thickness": err_int_th
            })

            # handle unstable runs
            if np.isnan(val) or val > 1e3:
                return 1e6
        
        except Exception as e:
            print(f"Instability for k={k_val}, A={A_val}: {e}")
            return 1e6  # penalize failed run
        
        if val < best_val:
                best_val = val
                best_k = k_val
                best_A = A_val

        print(f"optimization k: {k_val}, k_b: {k_center}, A: {A_val}, A_b: {A_center}, obj: {val}")
        return val

    study.enqueue_trial({"k": k_val_ini, "A": A_val_ini})
    study.optimize(objective_opt, n_trials=n_trials, catch=(Exception,))

    return study.best_params, trial_log


# ---------------------------------------
# MAIN LOOP
# ---------------------------------------

if __name__ == "__main__":

    T_X_vals = np.array([0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
    rho_l_vals = np.array([9.654071475,9.464574999,9.266219465,9.057781195,8.837838688,8.604722001,8.356422485,8.090447852,7.80359137,7.49154892,7.148238315,6.7644704,6.324991146,5.800445742,5.116045703,3.5])
    rho_g_vals = np.array([1.79E-04,1.40E-03,5.91E-03,0.017188114,3.93E-02,0.076113825,0.131530159,0.209223388,0.31316375,0.480780526,0.620231522,0.838834226,1.119054876,1.490095732,2.026552244,3.5])
    k_vals = np.array([0.0083,0.009,0.009,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.04,0.05])
    A_vals = np.array([0.27022,0.27,0.26,0.26,0.25,0.24,0.23,0.22,0.2,0.18,0.26,0.24,0.22,0.185,0.18,0.2])
    # T_X_vals = np.array([0.45])
    # rho_l_vals = np.array([8.837838688])
    # rho_g_vals = np.array([3.93E-02])
    # k_vals = np.array([0.01])
    # A_vals = np.array([0.26])

    interface_thickness_target = 5
    k_rad = 0.025
    A_rad = 0.05
    opt_runs = 50
    
    assert len(T_X_vals) == len(rho_l_vals) == len(rho_g_vals), \
        "Arrays must have same length!"

    all_trials = []
    best_results = []

    for T_X, rho_l_local, rho_g_local, k_val_ini, A_val_ini in zip(T_X_vals, rho_l_vals, rho_g_vals, k_vals, A_vals):
        print(f"\n=== Optimizing X = {T_X:.2f} ===")

        best_params, trials = optimize_T(
            T_X,
            rho_l_local,
            rho_g_local,
            k_val_ini,
            A_val_ini,
            interface_thickness_target,
            k_rad, A_rad,
            n_trials=opt_runs
        )

        all_trials.extend(trials)

        best_results.append({
            "X": T_X,
            "T": T_X * Tc,
            "best_k": best_params["k"],
            "best_A": best_params["A"]
        })

        print(f"Best for X={T_X:.2f}: k={best_params['k']:.4f}, A={best_params['A']:.4f}")

    # ---------------------------------------
    # SAVE RESULTS
    # ---------------------------------------

    df_all = pd.DataFrame(all_trials)
    df_best = pd.DataFrame(best_results)

    df_all.to_csv("optuna_all_trials.csv", index=False)
    df_best.to_csv("optuna_best_params.csv", index=False)

    print("\nSaved:")
    print("- optuna_all_trials.csv")
    print("- optuna_best_params.csv")