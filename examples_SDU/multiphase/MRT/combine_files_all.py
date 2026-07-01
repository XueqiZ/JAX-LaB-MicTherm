import os

dir_A = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_1"
dir_B = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_2"
dir_C = r"C:\Users\mlaut\GitRepos\JAX-LaB-MicTherm\examples_SDU\multiphase\MRT\results_total"

os.makedirs(dir_C, exist_ok=True)

output_file = os.path.join(dir_C, "optuna_all_merged.csv")

header_written = False

with open(output_file, "w") as out:

    for fname in os.listdir(dir_A):
        if not fname.startswith("optuna_all_"):
            continue
        
        path_A = os.path.join(dir_A, fname)
        path_B = os.path.join(dir_B, fname)

        if os.path.isfile(path_A) and os.path.isfile(path_B):

            # --- Process file A ---
            with open(path_A, "r") as fA:
                header = fA.readline()
                if not header:
                    continue

                if not header_written:
                    out.write(header)
                    header_written = True

                out.writelines(fA)

            # --- Process file B ---
            with open(path_B, "r") as fB:
                next(fB, None)  # skip header
                out.writelines(fB)