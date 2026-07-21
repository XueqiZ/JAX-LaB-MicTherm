from functools import partial

import jax
from jax import jit
from jax.tree import map

import jax.numpy as jnp


class EOS:
    """
    Base class for all equation of state. By default isothermal temperature field is used, which requires specifying temperature T during initialization.
    """

    def __init__(self, temperature_field_type="isothermal", **kwargs):
        self.debugging = bool(kwargs.get("debugging", False))
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.R = kwargs.get("R")
        self.temperature_field_type = temperature_field_type
        if self.temperature_field_type == "isothermal":
            self.T = kwargs.get("T")

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if value is None:
            raise ValueError("Gas constant value must be provided")
        if isinstance(value, float) or isinstance(value, int):
            self._R = [value]
        elif isinstance(value, list):
            self._R = value
        else:
            raise ValueError("Gas constant must be int, float or a list (for a multi-component flows)")

    @property
    def temperature_field_type(self):
        return self._temperature_field_type

    @temperature_field_type.setter
    def temperature_field_type(self, value):
        if value in ["isothermal", "thermal"]:
            self._temperature_field_type = value
        else:
            raise ValueError("Invalid temperature_field_type, it can only be 'isothermal' or 'thermal'")

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value is None and self.temperature_field_type == "isothermal":
            raise ValueError("Temperature value must be provided for isothermal case")
        if self.temperature_field_type == "isothermal" and bool(jnp.any(jnp.asarray(value) < 0)):
            raise ValueError("Temperature cannot be negative")
        self._T = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("EOS parameter a must be provided EOS")
        if isinstance(value, float) or isinstance(value, int):
            self._a = [value]
        elif isinstance(value, list):
            self._a = value
        else:
            raise ValueError("EOS parameter a must be int, float or a list (for multi-component flows)")

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("EOS parameter b must be provided EOS")
        if isinstance(value, float) or isinstance(value, int):
            self._b = [value]
        elif isinstance(value, list):
            self._b = value
        else:
            raise ValueError("EOS parameter b must be int, float or a list (for multi-component flows)")

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        pass


class VanderWaal(EOS):
    """
    Define multiphase model using the VanderWaals EOS.

    Parameters
    ----------
    a: list
    b: list
    R: list
    T: float or jax.numpy.ndarray

    Reference
    ---------
    1. Reprint of: The Equation of State for Gases and Liquids. The Journal of Supercritical Fluids,
    100th year Anniversary of van der Waals' Nobel Lecture, 55, no. 2 (2010): 403–14. https://doi.org/10.1016/j.supflu.2010.11.001.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - a*rho^2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        def eos(a, b, R, rho):
            return (rho * R * self.T) / (1.0 - b * rho) - a * rho**2

        def eos_with_debug(a, b, R, rho):
            denom = 1.0 - b * rho
            p = (rho * R * self.T) / denom - a * rho**2
            jax.debug.print("[EOS] denom min/max = {}/{}", jnp.nanmin(denom), jnp.nanmax(denom))
            jax.debug.print(
                "[EOS] nan counts rho/denom/p = {}/{}/{}",
                jnp.sum(jnp.isnan(rho)),
                jnp.sum(jnp.isnan(denom)),
                jnp.sum(jnp.isnan(p)),
            )
            jax.debug.print(
                "[EOS] non-finite counts rho/denom/p = {}/{}/{}",
                jnp.sum(~jnp.isfinite(rho)),
                jnp.sum(~jnp.isfinite(denom)),
                jnp.sum(~jnp.isfinite(p)),
            )
            jax.debug.print("VdW EOS — p min/max = {}/{}", jnp.min(p), jnp.max(p))
            jax.debug.print("VdW EOS — rho min/max = {}/{}", jnp.min(rho), jnp.max(rho))
            return p

        if self.debugging:
            return map(eos_with_debug, self.a, self.b, self.R, rho_tree)
        return map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        eos = lambda a, b, R, rho: (rho * R * T) / (1.0 - b * rho) - a * rho**2
        return map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        drho_dT = lambda b, R, rho: (rho * R) / (1.0 - b * rho)
        return map(lambda b, R, rho: drho_dT(b, R, rho), self.b, self.R, rho_tree)



class originalVdW(EOS):
    """
    Define multiphase model using the VanderWaals EOS.

    Parameters
    ----------
    a: list
    b: list
    R: list
    T: float or jax.numpy.ndarray

    Reference
    ---------
    1. Reprint of: The Equation of State for Gases and Liquids. The Journal of Supercritical Fluids,
    100th year Anniversary of van der Waals' Nobel Lecture, 55, no. 2 (2010): 403–14. https://doi.org/10.1016/j.supflu.2010.11.001.

    Notes
    -----

    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - a*rho^2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, R, rho: (rho * R * self.T) / (1.0 - b * rho) - a * rho**2
        return map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        eos = lambda a, b, R, rho: (rho * R * T) / (1.0 - b * rho) - a * rho**2
        return map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        drho_dT = lambda b, R, rho: (rho * R) / (1.0 - b * rho)
        return map(lambda b, R, rho: drho_dT(b, R, rho), self.b, self.R, rho_tree)



class MicTherm(EOS):
    """
    Define a tabulated multiphase EOS model with lookup/interpolation.

    Parameters
    ----------
    T : float or jax.numpy.ndarray
        Temperature used for isothermal lookup (`EOS`) when `T_grid` is not
        provided, or as runtime temperature input in thermal mode.
    p_grid : jax.numpy.ndarray
        Pressure table values. Supports:

        - 1D shape `(Nrho,)`: pressure as a function of `rho_grid` only.
        - 2D shape `(NT, Nrho)`: pressure as a function of `(T_grid, rho_grid)`.
    rho_grid : jax.numpy.ndarray
        Density coordinates for the pressure table. Accepts:

        - 1D axis `(Nrho,)`, or
        - 2D mesh grid matching `p_grid` shape.
    T_grid : jax.numpy.ndarray, optional
        Temperature coordinates for 2D table mode. Accepts:

        - 1D axis `(NT,)`, or
        - 2D mesh grid matching `p_grid` shape.
    dpdT_grid : jax.numpy.ndarray, optional
        Precomputed \(\partial p / \partial T\) table with same shape as `p_grid`.
        If omitted in 2D mode, it is computed from `p_grid` using
        `jnp.gradient(..., axis=0)`.

    Notes
    -----
        Testcase/table initialization behavior:

        1. Input validation
             - `p_grid` must be 1D or 2D.
             - `rho_grid` is mandatory.
             - `T_grid` is mandatory only for 2D `p_grid`.
        2. 1D vs 2D grid handling
             - 1D mode (`p_grid.ndim == 1`): interpolates only along `rho_grid`.
             - 2D mode (`p_grid.ndim == 2`): uses bilinear interpolation in
                 `(T, rho)` on a table shaped `(len(T_grid), len(rho_grid))`.
             - 2D mesh-style `rho_grid` / `T_grid` are accepted and converted to
                 1D axes when they represent valid mesh grids.
        3. Sorting and orientation
             - `rho_grid` and `T_grid` are sorted ascending.
             - `p_grid`/`dpdT_grid` are reordered consistently.
             - If mesh axes indicate transposed orientation, `p_grid` is transposed
                 to standard `(T, rho)` layout.
        4. Uniform-grid optimization
             - If axis spacing is uniform, index-based interpolation is used for
                 faster evaluation.
             - Otherwise, non-uniform interpolation is used (`jnp.interp` for 1D,
                 searchsorted + bilinear weights for 2D).

        Runtime behavior during simulation:

        - `EOS(rho_tree)`:
            pressure lookup at model temperature `self.T`.
        - `EOS_thermal(rho_tree, T)`:
            pressure lookup at runtime temperature `T`.
        - `drho_dT(rho_tree, T)`:
            lookup of `dpdT_grid` at `(rho, T)`.
    """

    @staticmethod
    def _axis_from_grid(grid, expected_shape, name):
        if grid is None:
            return None, None

        grid = jnp.asarray(grid)
        if grid.ndim == 1:
            return grid, None

        if grid.ndim != 2:
            raise ValueError(f"{name} must be a 1D axis or 2D mesh grid")
        if grid.shape != expected_shape:
            raise ValueError(f"2D {name} must have the same shape as p_grid")

        if bool(jnp.all(grid == grid[0:1, :])):
            return grid[0, :], 1
        if bool(jnp.all(grid == grid[:, 0:1])):
            return grid[:, 0], 0

        raise ValueError(f"2D {name} must be a mesh grid with one varying axis")

    @staticmethod
    def _uniform_spacing(axis):
        if axis.shape[0] < 2:
            return False, None

        spacing = axis[1] - axis[0]
        is_uniform = bool(jnp.allclose(jnp.diff(axis), spacing))
        return is_uniform, spacing
    # initialization step bevore the simulation starts, where the EOS object is created and the grids are processed and stored for later use during the simulation. The `__init__` method handles
    #  all the necessary checks and preparations to ensure that the EOS can be evaluated efficiently during the simulation run.
    def __init__(self, **kwargs):
        temperature_field_type = kwargs.get("temperature_field_type")
        if temperature_field_type is None:
            temperature_field_type = "thermal" if kwargs.get("T") is None and kwargs.get("T_grid") is not None else "isothermal"
        self.temperature_field_type = temperature_field_type
        self.T = kwargs.get("T")
        self.p_grid = jnp.asarray(kwargs.get("p_grid"))
        self.dpdT_grid = kwargs.get("dpdT_grid")
        self.eta_grid = kwargs.get("eta_grid")
        self.gamma_surface_grid = kwargs.get("gamma_surface_grid")

        property_tables = {
            "eta_grid": self.eta_grid,
            "gamma_surface_grid": self.gamma_surface_grid,
        }
        for name, table in property_tables.items():
            if table is not None:
                table = jnp.asarray(table)
                if table.shape != self.p_grid.shape:
                    raise ValueError(f"{name} must have the same shape as p_grid")
                property_tables[name] = table

        if self.p_grid.ndim not in (1, 2):
            raise ValueError("p_grid must be 1D or 2D")

        self.rho_grid, rho_axis_dim = self._axis_from_grid(
            kwargs.get("rho_grid"),
            self.p_grid.shape,
            "rho_grid",
        )
        self.T_grid, t_axis_dim = self._axis_from_grid(
            kwargs.get("T_grid"),
            self.p_grid.shape,
            "T_grid",
        )
        if self.rho_grid is None:
            raise ValueError("rho_grid must be provided")

        if self.p_grid.ndim == 1 and rho_axis_dim is not None:
            raise ValueError("2D rho_grid can only be used when p_grid is 2D")

        if self.p_grid.ndim == 2:
            if rho_axis_dim == 0 and t_axis_dim == 1:
                self.p_grid = self.p_grid.T
                if self.dpdT_grid is not None:
                    self.dpdT_grid = jnp.asarray(self.dpdT_grid).T
                property_tables = {
                    name: None if table is None else table.T
                    for name, table in property_tables.items()
                }
            elif rho_axis_dim in (None, 1) and t_axis_dim in (None, 0):
                pass
            else:
                raise ValueError("2D grids must orient p_grid as (T, rho) or provide matching mesh grids")

        sort_idx = jnp.argsort(self.rho_grid)
        self.rho_grid = self.rho_grid[sort_idx]

        if self.p_grid.ndim == 1:
            if self.p_grid.shape[0] != self.rho_grid.shape[0]:
                raise ValueError("1D p_grid must have the same length as rho_grid")
            self.p_grid = self.p_grid[sort_idx]
            property_tables = {
                name: None if table is None else table[sort_idx]
                for name, table in property_tables.items()
            }
            self.T_grid = None
            self._T_grid_uniform = False
            self._T_grid_spacing = None
        else:
            if self.p_grid.shape[1] != self.rho_grid.shape[0]:
                raise ValueError("2D p_grid must have shape (len(T_grid), len(rho_grid))")
            if self.T_grid is None:
                raise ValueError("T_grid must be provided when p_grid is 2D")
            if self.rho_grid.shape[0] < 2:
                raise ValueError("2D interpolation requires at least two rho_grid points")

            self.T_grid = jnp.asarray(self.T_grid)
            if self.T_grid.ndim != 1 or self.T_grid.shape[0] != self.p_grid.shape[0]:
                raise ValueError("T_grid must be 1D with length p_grid.shape[0]")
            if self.T_grid.shape[0] < 2:
                raise ValueError("2D interpolation requires at least two T_grid points")

            t_sort_idx = jnp.argsort(self.T_grid)
            self.T_grid = self.T_grid[t_sort_idx]
            self.p_grid = self.p_grid[t_sort_idx, :][:, sort_idx]
            property_tables = {
                name: None if table is None else table[t_sort_idx, :][:, sort_idx]
                for name, table in property_tables.items()
            }
            self._T_grid_uniform, self._T_grid_spacing = self._uniform_spacing(self.T_grid)

        self.eta_grid = property_tables["eta_grid"]
        self.gamma_surface_grid = property_tables["gamma_surface_grid"]

        self._rho_grid_uniform, self._rho_grid_spacing = self._uniform_spacing(self.rho_grid)

        if self.dpdT_grid is not None:
            self.dpdT_grid = jnp.asarray(self.dpdT_grid)
            if self.dpdT_grid.shape != self.p_grid.shape:
                raise ValueError("dpdT_grid must have the same shape as p_grid")
            if self.dpdT_grid.ndim == 1:
                self.dpdT_grid = self.dpdT_grid[sort_idx]
            else:
                self.dpdT_grid = self.dpdT_grid[t_sort_idx, :][:, sort_idx]
        elif self.p_grid.ndim == 2:
            self.dpdT_grid = jnp.gradient(self.p_grid, self.T_grid, axis=0)
        else:
            self.dpdT_grid = jnp.zeros_like(self.p_grid)

    # three interpolation methods (`_interp_rho`, `_interp_rho_uniform`, and `_interp_rho_T`) that handle the actual interpolation logic for looking up 
    # pressure values based on the input density and temperature. The `_lookup` method serves as a dispatcher to call the appropriate interpolation method 
    # based on whether a temperature grid is present. 
    def _interp_rho(self, table, rho):
        if self._rho_grid_uniform:
            return self._interp_rho_uniform(table, rho)
        return jnp.interp(rho, self.rho_grid, table)

    # Explanation:
    # _interp_rho/_interp_rho_uniform: 1-D interpolation in density (rho) for a
    # table defined over self.rho_grid. If the density grid is uniform we use
    # an index+weight approach for efficiency (_interp_rho_uniform), otherwise
    # we fall back to jnp.interp.

    def _interp_rho_uniform(self, table, rho):
        rho = jnp.asarray(rho)
        idx_float = (rho - self.rho_grid[0]) / self._rho_grid_spacing
        rho_idx = jnp.floor(idx_float).astype(jnp.int32)
        rho_idx = jnp.clip(rho_idx, 0, self.rho_grid.shape[0] - 2)

        rho0 = self.rho_grid[0] + rho_idx * self._rho_grid_spacing
        rho_weight = (rho - rho0) / self._rho_grid_spacing

        p0 = table[rho_idx]
        p1 = table[rho_idx + 1]
        return p0 + rho_weight * (p1 - p0)
    
    def _interp_rho_T(self, table, rho, T):
        rho = jnp.asarray(rho)
        T = jnp.asarray(T)
        rho, T = jnp.broadcast_arrays(rho, T)

        # Explanation:
        # This function performs bilinear interpolation on a 2-D table
        # organized as table[T_index, rho_index]. Steps:
        # 1) Convert rho and T to indices (floor) into the grid. If grids are
        #    uniform we compute fractional indices directly, otherwise we use
        #    searchsorted to find the bracketing index.
        # 2) Clip indices to valid ranges so we always have a lower and upper
        #    neighbor for both dimensions.
        # 3) Compute interpolation weights for rho and T. For uniform grids
        #    weights are computed from the start + spacing; otherwise use the
        #    neighboring grid values.
        # 4) Fetch the four corner values p00, p10, p01, p11 from the table:
        #      p00 = f(T0, rho0), p10 = f(T0, rho1),
        #      p01 = f(T1, rho0), p11 = f(T1, rho1)
        # 5) Linearly interpolate in rho between p00/p10 and p01/p11, then
        #    interpolate the resulting values in T to obtain the final value.

        if self._rho_grid_uniform and self._T_grid_uniform:
            rho_idx_float = (rho - self.rho_grid[0]) / self._rho_grid_spacing
            T_idx_float = (T - self.T_grid[0]) / self._T_grid_spacing
            rho_idx = jnp.floor(rho_idx_float).astype(jnp.int32)
            T_idx = jnp.floor(T_idx_float).astype(jnp.int32)
        else:
            rho_idx = jnp.searchsorted(self.rho_grid, rho, side="right") - 1
            T_idx = jnp.searchsorted(self.T_grid, T, side="right") - 1

        rho_idx = jnp.clip(rho_idx, 0, self.rho_grid.shape[0] - 2)
        T_idx = jnp.clip(T_idx, 0, self.T_grid.shape[0] - 2)

        if self._rho_grid_uniform:
            rho0 = self.rho_grid[0] + rho_idx * self._rho_grid_spacing
            rho_weight = (rho - rho0) / self._rho_grid_spacing
        else:
            rho0 = self.rho_grid[rho_idx]
            rho1 = self.rho_grid[rho_idx + 1]
            rho_weight = (rho - rho0) / (rho1 - rho0)

        if self._T_grid_uniform:
            T0 = self.T_grid[0] + T_idx * self._T_grid_spacing
            T_weight = (T - T0) / self._T_grid_spacing
        else:
            T0 = self.T_grid[T_idx]
            T1 = self.T_grid[T_idx + 1]
            T_weight = (T - T0) / (T1 - T0)

        p00 = table[T_idx, rho_idx]
        p10 = table[T_idx, rho_idx + 1]
        p01 = table[T_idx + 1, rho_idx]
        p11 = table[T_idx + 1, rho_idx + 1]

        p0 = p00 + rho_weight * (p10 - p00)
        p1 = p01 + rho_weight * (p11 - p01)
        return p0 + T_weight * (p1 - p0)

    def _lookup(self, table, rho, T):
        if self.T_grid is None:
            return self._interp_rho(table, rho)
        return self._interp_rho_T(table, rho, T)

    def interpolate_properties(self, rho, T):
        """Return interpolated p, eta, and gamma_surface at (rho, T).

        ``rho`` and ``T`` may be scalars or broadcast-compatible arrays.
        """
        if self.T_grid is None:
            raise ValueError("interpolate_properties(rho, T) requires a 2D p_grid and T_grid")
        if self.eta_grid is None:
            raise ValueError("eta_grid must be provided to interpolate viscosity")
        if self.gamma_surface_grid is None:
            raise ValueError("gamma_surface_grid must be provided to interpolate surface tension")

        return {
            "p": self._interp_rho_T(self.p_grid, rho, T),
            "eta": self._interp_rho_T(self.eta_grid, rho, T),
            "gamma_surface": self._interp_rho_T(self.gamma_surface_grid, rho, T),
        }

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        return map(lambda rho: self._lookup(self.p_grid, rho, self.T), rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        return map(lambda rho: self._lookup(self.p_grid, rho, T), rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        return map(lambda rho: self._lookup(self.dpdT_grid, rho, T), rho_tree)


def interpolate_mictherm_properties(kwargs, rho, T):
    """Interpolate MicTherm properties from grid data stored in ``kwargs``.

    Parameters
    ----------
    kwargs : dict
        Must contain ``rho_grid``, ``T_grid``, ``p_grid``, ``eta_grid``, and
        ``gamma_surface_grid``.
    rho : scalar or array-like
        Density value or field at which to evaluate the property tables.
    T : scalar or array-like
        Temperature value or field. It must be broadcast-compatible with
        ``rho``.

    Returns
    -------
    dict
        Interpolated values with keys ``p``, ``eta``, and
        ``gamma_surface``.
    """
    required = (
        "rho_grid",
        "T_grid",
        "p_grid",
        "eta_grid",
        "gamma_surface_grid",
    )
    missing = [name for name in required if kwargs.get(name) is None]
    if missing:
        raise ValueError(f"Missing MicTherm property grids: {', '.join(missing)}")

    eos = MicTherm(**kwargs)
    return eos.interpolate_properties(rho, T)


class Redlich_Kwong(EOS):
    """
    Define multiphase model using the Redlich-Kwong EOS.

    Parameters
    ----------
    a: list
    b: list
    R: list
    T: float or jax.numpy.ndarray

    Reference
    ---------
    1. Redlich O., Kwong JN., "On the thermodynamics of solutions; an equation of state; fugacities of gaseous solutions."
    Chem Rev. 1949 Feb;44(1):233-44. https://doi.org/10.1021/cr60137a013.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*rho^2)/(sqrt(T) * (1 + b*rho))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (a * rho**2) / (jnp.sqrt(self.T) * (1.0 + b * rho))
        return map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        eos = lambda a, b, R, rho: (rho * R * T) / (1.0 - b * rho) - (a * rho**2) / (jnp.sqrt(T) * (1.0 + b * rho))
        return map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        drho_dT = lambda a, b, R, rho: (rho * R) / (1.0 - b * rho) + (0.5 * a * rho**2) / ((T**1.5) * (1.0 + b * rho))
        return map(lambda a, b, R, rho: drho_dT(a, b, R, rho), self.a, self.b, self.R, rho_tree)


class Redlich_Kwong_Soave(EOS):
    """
    Define multiphase model using the Redlich-Kwong-Soave EOS.

    Parameters
    ----------
    a: list
    b: list
    R: list
    T: float or jax.numpy.ndarray

    Reference
    ---------
    1. Giorgio Soave, "Equilibrium constants from a modified Redlich-Kwong equation of state",
    Chemical Engineering Science 27, no. 6(1972), 1197-1203, https://doi.org/10.1016/0009-2509(72)80096-4.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + b*rho)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rks_omega = kwargs.get("RKS_omega")
        self.set_alpha()

    @property
    def rks_omega(self):
        return self._rks_omega

    @rks_omega.setter
    def rks_omega(self, value):
        if value is None:
            raise ValueError("rks_omega value must be provided for using Redlich-Kwong EOS")
        self._rks_omega = value

    def set_alpha(self):
        if self.temperature_field_type == "isothermal":
            Tc_tree = map(
                lambda a, b, R: (a / b) * (0.08664 / 0.42784) * (1 / R),
                self.a,
                self.b,
                self.R,
            )
            self.alpha = map(
                lambda rks_omega, Tc: (1 + (0.480 + 1.574 * rks_omega - 0.176 * rks_omega**2) * (1 - jnp.sqrt(self.T / Tc))) ** 2,
                self.rks_omega,
                Tc_tree,
            )

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, alpha, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + b * rho)
        return map(eos, self.a, self.b, self.alpha, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        self.Tc_tree = map(
            lambda a, b, R: (a / b) * (0.08664 / 0.42784) * (1 / R),
            self.a,
            self.b,
            self.R,
        )
        alpha_tree = map(
            lambda rks_omega, Tc: (1 + (0.480 + 1.574 * rks_omega - 0.176 * rks_omega**2) * (1 - jnp.sqrt(T / Tc))) ** 2,
            self.rks_omega,
            self.Tc_tree,
        )
        eos = lambda a, b, alpha, R, rho: (rho * R * T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + b * rho)
        return map(eos, self.a, self.b, alpha_tree, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        dalpha_dT = lambda rks_omega, Tc: (
            -1.0
            * (T / Tc) ** 0.5
            * ((1 - (T / Tc) ** 0.5) * (-0.176 * rks_omega**2 + 1.574 * rks_omega + 0.48) + 1)
            * (-0.176 * rks_omega**2 + 1.574 * rks_omega + 0.48)
            / T
        )
        drho_dT = lambda a, b, R, rho, rks_omega, Tc: (rho * R) / (1.0 - b * rho) - (a * dalpha_dT(rks_omega, Tc) * rho**2) / (1.0 + b * rho)
        return map(
            lambda a, b, R, rho, rks_omega, Tc: drho_dT(a, b, R, rho, rks_omega, Tc), self.a, self.b, self.R, rho_tree, self.rks_omega, self.Tc_tree
        )


class Peng_Robinson(EOS):
    """
    Define multiphase model using the Peng-Robinson EOS.

    Parameters
    ----------
    a: list
    b: list
    R: list
    T: float or jax.numpy.ndarray

    Reference
    ---------
    1. Peng, Ding-Yu, and Donald B. Robinson. "A new two-constant equation of state."
    Industrial & Engineering Chemistry Fundamentals 15, no. 1 (1976): 59-64. https://doi.org/10.1021/i160057a011

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pr_omega = kwargs.get("pr_omega")
        self.set_alpha()

    @property
    def pr_omega(self):
        return self._pr_omega

    @pr_omega.setter
    def pr_omega(self, value):
        if value is None:
            raise ValueError("pr_omega value must be provided for using Peng-Robinson EOS")
        if isinstance(value, int) or isinstance(value, float):
            self._pr_omega = [value]
        if isinstance(value, list):
            self._pr_omega = value

    def set_alpha(self):
        if self.temperature_field_type == "isothermal":
            Tc_tree = map(
                lambda a, b, R: (a / b) * (0.0778 / 0.45724) * (1 / R),
                self.a,
                self.b,
                self.R,
            )
            self.alpha = map(
                lambda pr_omega, Tc: (1 + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2) * (1 - jnp.sqrt(self.T / Tc))) ** 2,
                self.pr_omega,
                Tc_tree,
            )

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, alpha, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + 2 * b * rho - b**2 * rho**2)
        return map(eos, self.a, self.b, self.alpha, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        self.Tc_tree = map(
            lambda a, b, R: (a / b) * (0.0778 / 0.45724) * (1 / R),
            self.a,
            self.b,
            self.R,
        )
        alpha_tree = map(
            lambda pr_omega, Tc: (1 + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2) * (1 - jnp.sqrt(T / Tc))) ** 2,
            self.pr_omega,
            self.Tc_tree,
        )
        eos = lambda a, b, alpha, R, rho: (rho * R * T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + 2 * b * rho - b**2 * rho**2)
        return map(eos, self.a, self.b, alpha_tree, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        dalpha_dT = lambda pr_omega, Tc: (
            -1.0
            * (T / Tc) ** 0.5
            * ((1 - (T / Tc) ** 0.5) * (-0.26992 * pr_omega**2 + 1.54226 * pr_omega + 0.37464) + 1)
            * (-0.26992 * pr_omega**2 + 1.54226 * pr_omega + 0.37464)
            / T
        )
        drho_dT = lambda a, b, R, rho, pr_omega, Tc: (
            (rho * R) / (1.0 - b * rho) - (a * dalpha_dT(pr_omega, Tc) * rho**2) / (1.0 + 2 * b * rho - b**2 * rho**2)
        )
        return map(
            lambda a, b, R, rho, pr_omega, Tc: drho_dT(a, b, R, rho, pr_omega, Tc), self.a, self.b, self.R, rho_tree, self.pr_omega, self.Tc_tree
        )


class Carnahan_Starling(EOS):
    """
    Define multiphase model using the Carnahan-Starling EOS.

    Parameters
    ----------
    a: list
    b: list
    R: list
    T: float or jax.numpy.ndarray

    Reference
    ---------
    1.  Carnahan, Norman F., and Kenneth E. Starling. "Equation of state for nonattracting rigid spheres."
    The Journal of chemical physics 51, no. 2 (1969): 635-636. https://doi.org/10.1063/1.1672048

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)*(1 + (0.25*b*rho) + (0.25*b*rho)^2 - (0.25*b*rho)^3)/(1 - b*rho)^3 - (a*rho^2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        x_tree = map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        eos = lambda a, R, rho, x: (rho * R * self.T * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)) - (a * rho**2)
        return map(eos, self.a, self.R, rho_tree, x_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        x_tree = map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        eos = lambda a, R, rho, x: (rho * R * T * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)) - (a * rho**2)
        return map(eos, self.a, self.R, rho_tree, x_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        x_tree = map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        drho_dT = lambda x, R, rho: (rho * R) * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)
        return map(lambda x, R, rho: drho_dT(x, R, rho), x_tree, self.R, rho_tree)
