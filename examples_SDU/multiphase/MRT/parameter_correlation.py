import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

# =========================
# SETTINGS
# =========================
data_folder = Path(r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_total")
data_file = data_folder / "optuna_all_merged.csv"

beta = 0.5         # weight sharpness (tune this!)
k_degree = 1    # degree of final model
A_degree = 1    # degree of final model

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(data_file)

# =========================
# COLUMN SELECTION
# =========================
T_col = df.columns[0]
k_col = df.columns[2]
A_col = df.columns[3]
value_col = df.columns[6]

counts = df.groupby(T_col).size()
df["T_count"] = df[T_col].map(counts)

# =========================
# WEIGHTING FROM PENALTY
# =========================
# Lower penalty → higher weight
df["weight"] = np.exp(-beta * df[value_col]) * (1.0 / df[T_col]**2) * (1.0 / df["T_count"])

# Optional normalization (safer numerically)
df["weight"] /= df["weight"].max()

# =========================
# FIT FUNCTIONS k(T), A(T)
# =========================
T = df[T_col].values

# Weighted polynomial fit
coeff_k = np.polyfit(T, df[k_col], deg=k_degree, w=df["weight"])
coeff_A = np.polyfit(T, df[A_col], deg=A_degree, w=df["weight"])

# =========================
# BUILD FUNCTIONS
# =========================
def k_of_T(Tval):
    return np.polyval(coeff_k, Tval)

def A_of_T(Tval):
    return np.polyval(coeff_A, Tval)

# =========================
# PRINT RESULTS
# =========================
def poly_to_string(coeff, name):
    terms = []
    deg = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = deg - i
        if power == 0:
            terms.append(f"{c:.4g}")
        elif power == 1:
            terms.append(f"{c:.4g}*T")
        else:
            terms.append(f"{c:.4g}*T^{power}")
    return f"{name}(T) = " + " + ".join(terms)

print("\n=== FITTED FUNCTIONS ===")
print(poly_to_string(coeff_k, "k"))
print(poly_to_string(coeff_A, "A"))

# =========================
# OPTIONAL: SAVE RESULTS
# =========================
output_file = data_folder / "fitted_functions.txt"
with open(output_file, "w") as f:
    f.write(poly_to_string(coeff_k, "k") + "\n")
    f.write(poly_to_string(coeff_A, "A") + "\n")

print(f"\nSaved equations to {output_file}")

# =========================
# OPTIONAL: VISUALIZATION
# =========================
import matplotlib.pyplot as plt

T_plot = np.linspace(df[T_col].min(), df[T_col].max(), 200)

plt.figure()
plt.scatter(df[T_col], df[k_col], s=5, alpha=0.3, label="data")
plt.plot(T_plot, k_of_T(T_plot), color="red", label="fit")
plt.xlabel("T")
plt.ylabel("k")
plt.legend()
plt.title("k(T)")
plt.show()

plt.figure()
plt.scatter(df[T_col], df[A_col], s=5, alpha=0.3, label="data")
plt.plot(T_plot, A_of_T(T_plot), color="red", label="fit")
plt.xlabel("T")
plt.ylabel("A")
plt.legend()
plt.title("A(T)")
plt.show()