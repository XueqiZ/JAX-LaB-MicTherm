from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import pandas as pd


X_COLUMN = "Tr"
Y_COLUMNS = (
    "rho_g (Maxwell)",
    "rho_g (Actual)",
    "rho_l (Maxwell)",
    "rho_l (Actual)",
)

# Y_COLUMNS = (
#     "rho_g (Maxwell)",
#     "rho_l (Maxwell)",
# )

# Y_COLUMNS = (
#     "rho_g (Actual)",
#     "rho_l (Actual)",
# )


def select_folder() -> Path | None:
    """Open a dialog and return the folder selected by the user."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_folder = filedialog.askdirectory(
        title="Select the folder containing the two CSV files"
    )
    root.destroy()
    return Path(selected_folder) if selected_folder else None


def main() -> None:
    folder = select_folder()
    if folder is None:
        print("No folder selected.")
        return

    csv_files = sorted(folder.glob("*.csv"))
    if len(csv_files) != 2:
        raise ValueError(
            f"Expected exactly two CSV files in '{folder}', found {len(csv_files)}."
        )

    # Each CSV file is stored in its own pandas DataFrame.
    dataframes = {csv_file.stem: pd.read_csv(csv_file) for csv_file in csv_files}

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = iter(plt.colormaps["tab10"].colors)
    plotted_series = 0

    for name, dataframe in dataframes.items():
        if X_COLUMN not in dataframe.columns:
            raise KeyError(f"'{name}.csv' is missing the '{X_COLUMN}' column.")

        for y_column in Y_COLUMNS:
            if y_column not in dataframe.columns:
                print(f"Skipping '{name}_{y_column}': column not found.")
                continue

            ax.scatter(
                dataframe[X_COLUMN],
                dataframe[y_column],
                color=next(colors),
                label=f"{name}_{y_column}",
                s=55,
                alpha=0.8,
            )
            plotted_series += 1

    if plotted_series == 0:
        raise KeyError(
            "None of the requested density columns were found in the CSV files."
        )

    ax.set_xlabel(X_COLUMN)
    ax.set_ylabel("Density")
    ax.set_title("Density comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_file = folder / "comparison_plot.png"
    fig.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
