import numpy as np
import pandas as pd
import optuna

# import your simulation function + constants
from .droplet_2d_sigma import *


# ---------------------------------------
# OBJECTIVE FUNCTION
# ---------------------------------------

def compute_objective(rho, int_thick, rho_l, rho_g, gamma_local, gamma, interface_thickness_target):

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
    err_gamma = abs((gamma - gamma_local) / gamma_local)

    objective = err_l + err_g + err_int_th + err_gamma

    return objective, err_l, err_g, err_int_th, err_gamma

# ---------------------------------------
# OPTIMIZATION PER TEMPERATURE
# ---------------------------------------

def optimize_T(T_X, rho_l_local, rho_g_local, gamma_local, k_pred, A_pred, interface_thickness_target, n_trials=30):

    T = T_X * Tc

    trial_log = []

    study = optuna.create_study(direction="minimize")

    best_kappa = 0.0

    def objective_opt(trial):
        nonlocal best_kappa

        #k_val = trial.suggest_float("k", max(k_val_ini - 0.1, 0.005), k_val_ini * 5)
        #A_val = trial.suggest_float("A", max(A_val_ini - 0.25, 0.0), A_val_ini + 0.25)
        kappa_val = trial.suggest_float("kappa", 0.0, 1.0)
        #k_val = trial.suggest_float("k", 0.0, 0.5)
        #A_val = trial.suggest_float("A", 0.0, 1.0)
        try:
            rho, int_thick, gamma = run_simulation(
                T_X,
                kappa_val,
                k_pred,
                A_pred,
                rho_l_local,
                rho_g_local,
                steps=50000   # shorter during optimization
            )

            val, err_l, err_g, err_int_thick, err_gamma = compute_objective(rho, int_thick, rho_l_local, rho_g_local, gamma_local, gamma, interface_thickness_target)

            # store every trial
            trial_log.append({
                "X": T_X,
                "T": T,
                "k": k_pred,
                "A": A_pred,
                "kappa": kappa_val,
                "rho_l": rho_l_local,
                "rho_g": rho_g_local,
                "objective": val,
                "err_l": err_l,
                "err_g": err_g,
                "err_int_thick": err_int_thick,
                "err_gamma": err_gamma
            })

            # handle unstable runs
            if np.isnan(val) or val > 1e3:
                return 1e6
        
        except Exception as e:
            print(f"Instability for k={k_pred}, A={A_pred}, kappa={kappa_val}: {e}")
            return 1e6  # penalize failed run
        
        if val < best_val:
                best_val = val
                best_kappa = kappa_val

        print(f"optimization kappa: {kappa_val}, k_b: {best_kappa}, obj: {val}")
        return val

    study.enqueue_trial({"kappa": best_kappa})
    study.optimize(objective_opt, n_trials=n_trials, catch=(Exception,))

    return study.best_params, trial_log


# ---------------------------------------
# MAIN LOOP
# ---------------------------------------

if __name__ == "__main__":

    # T_X_vals = np.array([0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
    # gamma_vals = np.array([28.68083224,25.83603628,23.09572013,20.44792849,17.88847033,15.41879843,13.04512701,10.77594319,8.622937019,6.600959307,4.730479196,3.043068624,1.59454414,0.49520152])*1e-3 # N/m
    # rho_l_vals = np.array([9.464574999,9.266219465,9.057781195,8.837838688,8.604722001,8.356422485,8.090447852,7.80359137,7.49154892,7.148238315,6.7644704,6.324991146,5.800445742,5.116045703])
    # rho_g_vals = np.array([1.40E-03,5.91E-03,0.017188114,3.93E-02,0.076113825,0.131530159,0.209223388,0.31316375,0.480780526,0.620231522,0.838834226,1.119054876,1.490095732,2.026552244])
    T_X_vals = np.array([0.5])
    gamma_vals = np.array([17.88847033])*1e-3 # N/m
    rho_l_vals = np.array([8.604722001])
    rho_g_vals = np.array([0.076113825])

    interface_thickness_target = 5
    opt_runs = 5
    
    assert len(T_X_vals) == len(gamma_vals) == len(rho_l_vals) == len(rho_g_vals), \
        "Arrays must have same length!"

    all_trials = []
    best_results = []

    for T_X, gamma_local, rho_l_local, rho_g_local in zip(T_X_vals, gamma_vals, rho_l_vals, rho_g_vals):
        print(f"\n=== Optimizing X = {T_X:.2f} ===")

        k_pred = 0.3181*T_X-0.07127
        A_pred = 0.08109*T_X+0.2338

        all_TX = []
        best_params, trials = optimize_T(
            T_X,
            rho_l_local,
            rho_g_local,
            gamma_local,
            k_pred, A_pred,
            interface_thickness_target,
            n_trials=opt_runs
        )

        all_TX.extend(trials)
        all_trials.extend(trials)

        best_results.append({
            "X": T_X,
            "T": T_X * Tc,
            "best_kappa": best_params["kappa"]
        })

        print(f"Best for X={T_X:.2f}: kappa={best_params['kappa']:.4f}")
        df_TX = pd.DataFrame(all_TX)
        df_TX.to_csv(f'optuna_all_{T_X}.csv', index=False)

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