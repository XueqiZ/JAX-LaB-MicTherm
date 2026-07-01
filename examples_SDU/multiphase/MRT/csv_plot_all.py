import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import glob
import re
import os

# =========================
# CONFIG
# =========================
file_pattern = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_total\optuna_all_*.csv"


# =========================
# LOAD ALL FILES ROBUSTLY
# =========================
files = glob.glob(file_pattern)

if len(files) == 0:
    raise ValueError("No files found. Check file_pattern.")

dfs = []

for f in files:
    try:
        # skip zero-size files early
        if os.path.getsize(f) == 0:
            print(f"Skipping empty file (0 bytes): {f}")
            continue

        df_temp = pd.read_csv(f)

        # skip empty dataframes
        if df_temp.empty:
            print(f"Skipping empty dataframe: {f}")
            continue

        # skip malformed
        if df_temp.shape[1] < 7:
            print(f"Skipping malformed file (<7 cols): {f}")
            continue

        # ---- Extract temperature ----
        # case 1: filename like optuna_all_0.9.csv
        match = re.search(r"_(\d+\.?\d*)\.csv", f)

        if match:
            T = float(match.group(1))
        else:
            # fallback: assume first column is temperature
            T = df_temp.iloc[0, 0]

        df_temp["temperature"] = T

        dfs.append(df_temp)

    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable file: {f}")
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Check result
if len(dfs) == 0:
    raise ValueError("No valid files loaded.")

df = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(dfs)} files.")
print(f"Temperatures found: {sorted(df['temperature'].unique())}")

# =========================
# COLUMN SELECTION
# =========================
x_col = df.columns[2]
y_col = df.columns[3]
value_col = df.columns[6]

# =========================
# GLOBAL COLOR NORMALIZATION
# =========================
vmin = df[value_col].min()
vmax = df[value_col].max()

# =========================
# GLOBAL AXIS LIMITS
# =========================
x_min_global = df[x_col].min()
x_max_global = df[x_col].max()

y_min_global = df[y_col].min()
y_max_global = df[y_col].max()

# avoid invalid LogNorm
if vmin <= 0:
    print("WARNING: Non-positive values detected. Switching to linear scale.")
    norm = None
else:
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

# =========================
# PREPARE PLOT GRID
# =========================
temps = sorted(df["temperature"].unique())

ncols = min(4, len(temps))
nrows = int(np.ceil(len(temps) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

# make iterable
axes = np.array(axes).reshape(-1)

# =========================
# PLOTTING LOOP
# =========================
for i, temp in enumerate(temps):
    ax = axes[i]

    subset = df[df["temperature"] == temp]

    if subset.empty:
        continue

    # find minimum
    idx_min = subset[value_col].idxmin()

    x_min = subset.loc[idx_min, x_col]
    y_min = subset.loc[idx_min, y_col]

    sc = ax.scatter(
        subset[x_col],
        subset[y_col],
        c=subset[value_col],
        cmap="viridis",
        norm=norm,
        s=20
    )

    # mark minimum
    ax.scatter(
        x_min,
        y_min,
        color="red",
        edgecolor="black",
        s=80,
        label="min"
    )

    # mark prediction/correlation
    ax.scatter(
        0.228*temp**2+0.06781*temp-0.008165,
        1.476*temp**2-1.46*temp+0.5845,
        color="green",
        edgecolor="black",
        s=80,
        label="corr"
    )
    ax.scatter(
        0.3403*temp-0.07948,
        0.1833*temp+0.1962,
        color="yellow",
        edgecolor="black",
        s=80,
        label="corr"
    )
    ax.scatter(
        0.3181*temp-0.07127,
        0.08109*temp+0.2338,
        color="orange",
        edgecolor="black",
        s=80,
        label="corr"
    )

    ax.set_title(f"T = {temp}")

    ax.set_xlim(x_min_global, x_max_global)
    ax.set_ylim(y_min_global, y_max_global)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


# =========================
# CLEAN UNUSED AXES
# =========================
for j in range(len(temps), len(axes)):
    fig.delaxes(axes[j])

# =========================
# COLORBAR
# =========================
#fig.colorbar(sc, ax=axes[:len(temps)], label=value_col)

#plt.tight_layout()
plt.show()