
from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd


CSV_PATH = Path(__file__).with_name("Thermodynamic Consistency - VdW.csv")


def read_thermodynamic_consistency(csv_path: Path = CSV_PATH) -> tuple[list[str], pd.DataFrame, float | None]:
    first_row = pd.read_csv(csv_path, nrows=1, header=None, sep=";").iloc[0]
    general_information = [str(item).strip() for item in first_row.dropna().tolist() if str(item).strip()]

    df = pd.read_csv(csv_path, skiprows=1, header=0, sep=";")
    df = df.dropna(how="all")

    rms_error = None
    rms_rows = df[df.astype(str).eq("RMS Error").any(axis=1)]
    if not rms_rows.empty:
        rms_values = rms_rows.iloc[0].dropna().to_list()
        for value in reversed(rms_values):
            try:
                rms_error = float(value)
                break
            except (TypeError, ValueError):
                continue
        df = df.drop(index=rms_rows.index)

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return general_information, df, rms_error


def main() -> None:
    general_information, pd_values, rms_error = read_thermodynamic_consistency()

    print("General information:")
    for item in general_information:
        print(f"- {item}")

    print("\nValidation data:")
    print(pd_values)
    print(f"\nRMS Error: {rms_error}")


if __name__ == "__main__":
    main()
