import os

dir_A = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_1"
dir_B = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_2"
dir_C = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_total"

# Create output directory if it doesn't exist
os.makedirs(dir_C, exist_ok=True)

for fname in os.listdir(dir_A):
    if not fname.startswith("optuna_all_"):
        continue
    
    path_A = os.path.join(dir_A, fname)
    path_B = os.path.join(dir_B, fname)
    path_C = os.path.join(dir_C, fname)

    # Only process if file exists in both A and B
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        with open(path_C, "w") as out:
            
            # Write header + data from A
            with open(path_A, "r") as fA:
                header = fA.readline()   # read header
                if not header:
                    continue
                out.write(header)        # write header ONCE
                out.writelines(fA)       # rest of A

            # Append only data from B (skip header)
            with open(path_B, "r") as fB:
                next(fB, None)           # skip header
                out.writelines(fB)