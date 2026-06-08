"""Create an MP4 video from a folder of PNG frames.

Examples:
    python scripts/pngs_to_video.py
    python scripts/pngs_to_video.py --input . --pattern "cavity2d_*.png" --fps 20
    python scripts/pngs_to_video.py --input results/images --output cavity.mp4
"""

from __future__ import annotations

import argparse
from datetime import datetime
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def natural_key(path: Path) -> list[int | str]:
    """Sort frame_2.png before frame_10.png."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_frames(input_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(input_dir.glob(pattern), key=natural_key)
    if not frames:
        raise SystemExit(f"No PNG files found in {input_dir} matching {pattern!r}.")
    return frames


def write_with_ffmpeg(frames: list[Path], output: Path, fps: int, crf: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg was not found on PATH.")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", encoding="utf-8", delete=False
    ) as frame_list:
        frame_list_path = Path(frame_list.name)
        for frame in frames:
            escaped = frame.resolve().as_posix().replace("'", "'\\''")
            frame_list.write(f"file '{escaped}'\n")

    try:
        command = [
            ffmpeg,
            "-y",
            "-r",
            str(fps),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(frame_list_path),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        subprocess.run(command, check=True)
    finally:
        frame_list_path.unlink(missing_ok=True)


def write_with_imageio(frames: list[Path], output: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Neither ffmpeg nor imageio is available. Install ffmpeg, or run "
            "`pip install imageio imageio-ffmpeg`."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output, fps=fps, codec="libx264", pixelformat="yuv420p"
    ) as video:
        for frame in frames:
            video.append_data(imageio.imread(frame))


def timestamped_output_dir(base_dir: Path) -> Path:
    """Create a run folder using the current local date and time."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return base_dir / stamp


def archive_matching_files(source_dir: Path, destination_dir: Path) -> None:
    """Move top-level PNG and VTK files into the output folder."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("*.png", "*.vtk"):
        for file_path in sorted(source_dir.glob(pattern), key=natural_key):
            shutil.move(str(file_path), destination_dir / file_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an MP4 video from PNG image frames and archive outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Folder containing PNG frames. Default: current folder.",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help='Frame filename pattern. Example: "cavity2d_*.png". Default: "*.png".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Default: output/<timestamp>/output.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base folder for timestamped run outputs. Default: output/.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second. Default: 30.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="FFmpeg quality value, lower is better/larger. Default: 18.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    run_dir = timestamped_output_dir(args.output_dir.resolve())
    output = args.output.resolve() if args.output is not None else run_dir / "output.mp4"

    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0.")

    frames = collect_frames(input_dir, args.pattern)
    print(f"Found {len(frames)} frame(s). First: {frames[0].name}, last: {frames[-1].name}")
    print(f"Writing video: {output}")

    try:
        write_with_ffmpeg(frames, output, args.fps, args.crf)
    except FileNotFoundError:
        print("ffmpeg not found; trying Python imageio fallback...", file=sys.stderr)
        write_with_imageio(frames, output, args.fps)

    archive_matching_files(input_dir, run_dir)
    if output.parent != run_dir:
        shutil.move(str(output), run_dir / output.name)

    print(f"Archived PNG and VTK files to: {run_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
