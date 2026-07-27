"""MicTherm-specific output helpers."""

import os
from time import time

import numpy as np
import pyvista as pv
from termcolor import colored


def save_fields_vtk(timestep, fields, output_dir=".", prefix="fields"):
    """Save scalar and vector cell fields to a VTK image-data file.

    The first dictionary entry must be a 2D or 3D scalar field and defines the
    spatial grid. Additional entries may be scalar fields with the same shape
    or vector fields whose final dimension contains two or three components.
    Two-component vectors are padded with a zero z component for VTK tools.
    """
    if not fields:
        raise ValueError("At least one field is required.")

    first_value = np.asarray(next(iter(fields.values())))
    if first_value.ndim not in (2, 3):
        raise ValueError("The first VTK field must be a 2D or 3D scalar field.")
    spatial_shape = first_value.shape

    normalized_fields = {}
    for key, value in fields.items():
        value = np.asarray(value)
        is_scalar = value.shape == spatial_shape
        is_vector = (
            value.ndim == len(spatial_shape) + 1
            and value.shape[:-1] == spatial_shape
            and value.shape[-1] in (2, 3)
        )
        if not (is_scalar or is_vector):
            raise ValueError(
                f"Field {key!r} has shape {value.shape}; expected "
                f"{spatial_shape}, {spatial_shape + (2,)}, or "
                f"{spatial_shape + (3,)}."
            )
        normalized_fields[key] = value

    if not os.path.exists(output_dir):
        print(
            colored(
                "Directory does not exist, creating the directory " + output_dir,
                "yellow",
            )
        )
        os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(
        output_dir, prefix + "_" + f"{timestep:07d}.vtk"
    )
    dimensions = tuple(dimension + 1 for dimension in spatial_shape)
    if len(spatial_shape) == 2:
        dimensions += (1,)
    grid = pv.ImageData(dimensions=dimensions)

    for key, value in normalized_fields.items():
        if value.shape == spatial_shape:
            grid[key] = value.flatten(order="F")
            continue

        components = [
            value[..., component].flatten(order="F")
            for component in range(value.shape[-1])
        ]
        if value.shape[-1] == 2:
            components.append(np.zeros_like(components[0]))
        grid[key] = np.column_stack(components)

    start = time()
    grid.save(output_filename, binary=True)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")
