import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# === Load CSV ===
file_path = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_total\optuna_all_0.65.csv"  
df = pd.read_csv(file_path)
# keep only files where T_X = ...
df = df[df.iloc[:, 0] == 0.65]

# === Choose columns ===
x_col = df.columns[2]
y_col = df.columns[3]
value_col = df.columns[6]

# === Extract where Z is min ===
idx_min = df[value_col].idxmin()

x_min = df.loc[idx_min, x_col]
y_min = df.loc[idx_min, y_col]
z_min = df.loc[idx_min, value_col]

# === Pivot data into grid ===

pivot_table = df.pivot_table(
    index=y_col,
    columns=x_col,
    values=value_col,
    aggfunc="mean"   # or "min", "max", "first"
)


# === Convert to arrays ===
X = pivot_table.columns.values
Y = pivot_table.index.values
Z = pivot_table.values
Z = np.ma.masked_invalid(Z)


# === Plot heatmap ===
plt.figure()

sc = plt.scatter(
    df[x_col],
    df[y_col],
    c=df[value_col],
    cmap="viridis",
    norm=colors.LogNorm()
)

plt.scatter(
    x_min,
    y_min,
    color="red",
    s=100,
    edgecolor="black",
    label="Minimum"
)


plt.colorbar(label=value_col)
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("Heatmap from CSV")

plt.show()