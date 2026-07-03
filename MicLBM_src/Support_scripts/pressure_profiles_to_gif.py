"""
Create an animated pressure-profile GIF from pressure_profile_x_mid_*.csv files.

The script reads all profile CSV files from an input folder, writes a combined CSV
with one shared y column and one pressure column per timestep, then creates a GIF
showing p(y) over time.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


PROFILE_PATTERN = re.compile(r"pressure_profile_x_mid_(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine pressure profile CSV files and create a p(y) GIF."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output"),
        help="Folder containing pressure_profile_x_mid_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "pressure_profile_animation",
        help="Folder for the combined CSV, optional frames, and GIF.",
    )
    parser.add_argument(
        "--gif-name",
        default="pressure_profile_x_mid.gif",
        help="Name of the generated GIF file.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for the GIF.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save every animation frame as a PNG.",
    )
    return parser.parse_args()


def find_profile_files(input_dir: Path) -> list[tuple[int, Path]]:
    profile_files = []
    for csv_file in input_dir.glob("pressure_profile_x_mid_*.csv"):
        match = PROFILE_PATTERN.match(csv_file.name)
        if match:
            profile_files.append((int(match.group(1)), csv_file))

    profile_files.sort(key=lambda item: item[0])
    if not profile_files:
        raise FileNotFoundError(f"No pressure_profile_x_mid_*.csv files found in {input_dir}")

    return profile_files


def load_profiles(profile_files: list[tuple[int, Path]]) -> tuple[np.ndarray, list[int], np.ndarray]:
    first_data = np.loadtxt(profile_files[0][1], delimiter=",", skiprows=1)
    y_positions = first_data[:, 0]
    timesteps = []
    pressure_columns = []

    for timestep, csv_file in profile_files:
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        if data.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in {csv_file}")
        if not np.array_equal(y_positions, data[:, 0]):
            raise ValueError(f"y positions differ in {csv_file}")

        timesteps.append(timestep)
        pressure_columns.append(data[:, 1])

    return y_positions, timesteps, np.column_stack(pressure_columns)


def save_combined_csv(
    output_dir: Path, y_positions: np.ndarray, timesteps: list[int], pressures: np.ndarray
) -> Path:
    combined = np.column_stack((y_positions, pressures))
    header = ",".join(["y", *[f"pressure_t{timestep:07d}" for timestep in timesteps]])
    output_path = output_dir / "pressure_profiles_combined.csv"
    np.savetxt(output_path, combined, delimiter=",", header=header, comments="")
    return output_path


def save_gif(
    output_dir: Path,
    gif_name: str,
    y_positions: np.ndarray,
    timesteps: list[int],
    pressures: np.ndarray,
    fps: int,
    save_frames: bool,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Creating the GIF requires matplotlib and Pillow. Install them with "
            "`pip install matplotlib pillow`, or run this script in the environment "
            "where the simulation dependencies from requirements.txt are installed."
        ) from exc

    gif_path = output_dir / gif_name
    y_min, y_max = float(np.min(y_positions)), float(np.max(y_positions))
    p_min, p_max = float(np.min(pressures)), float(np.max(pressures))
    p_pad = 0.05 * (p_max - p_min) if p_max > p_min else 1.0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    (line,) = ax.plot([], [], lw=2)
    title = ax.set_title("")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(p_min - p_pad, p_max + p_pad)
    ax.set_xlabel("y position")
    ax.set_ylabel("pressure p")
    ax.grid(True, alpha=0.3)

    def update(frame_index: int):
        line.set_data(y_positions, pressures[:, frame_index])
        title.set_text(f"Pressure profile at x = x_max / 2, timestep {timesteps[frame_index]}")
        return line, title

    animation = FuncAnimation(fig, update, frames=len(timesteps), interval=1000 / fps, blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=fps))

    if save_frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, timestep in enumerate(timesteps):
            update(frame_index)
            fig.savefig(frames_dir / f"pressure_profile_{timestep:07d}.png", bbox_inches="tight")

    plt.close(fig)
    return gif_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile_files = find_profile_files(args.input_dir)
    y_positions, timesteps, pressures = load_profiles(profile_files)
    combined_csv = save_combined_csv(args.output_dir, y_positions, timesteps, pressures)
    gif_path = save_gif(
        args.output_dir,
        args.gif_name,
        y_positions,
        timesteps,
        pressures,
        args.fps,
        args.save_frames,
    )

    print(f"Read {len(profile_files)} pressure profiles from {args.input_dir}")
    print(f"Wrote combined CSV: {combined_csv}")
    print(f"Wrote GIF: {gif_path}")


if __name__ == "__main__":
    main()
