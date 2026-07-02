from .droplet_2d_sigma import *
import numpy as np
import pandas as pd

# T_X_vals = np.array([0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
# rho_l_vals = np.array([9.057781195,8.837838688,8.604722001,8.356422485,8.090447852,7.80359137,7.49154892,7.148238315,6.7644704,6.324991146,5.800445742,5.116045703])
# rho_g_vals = np.array([0.017188114,3.93E-02,0.076113825,0.131530159,0.209223388,0.31316375,0.480780526,0.620231522,0.838834226,1.119054876,1.490095732,2.026552244])

T_X_vals = np.array([0.8])
rho_l_vals = np.array([6.7644704])
rho_g_vals = np.array([0.838834226])

for T_X, rho_l, rho_g in zip(T_X_vals,rho_l_vals,rho_g_vals):
    print(f"\n=== Kappa for T = {T_X:.2f} ===")

    k_pred = 0.3181*T_X-0.07127
    A_pred = 0.08109*T_X+0.2338
    #k_pred = 0.02
    #A_pred = 0.24   #np.array([0.1157, 0.08, 0.032])

    for kappa_val in np.array([0, 0.5, 1]):
    #for kappa_val, A_pred in zip(np.array([0, 0.5, 1]), A_pred):
        try:
            rho, p = run_simulation(
                T_X,
                kappa_val,
                k_pred,
                A_pred,
                rho_l,
                rho_g,
                steps=20000
            )

        except Exception as e:
            print(f"Instability for k={k_pred}, A={A_pred}, kappa={kappa_val}: {e}")