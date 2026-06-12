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
        def eos_with_debug(a, b, R, rho):
            p = (rho * R * self.T) / (1.0 - b * rho) - a * rho**2
            # jax.debug.print("VdW EOS — p min/max = {}/{}", jnp.min(p), jnp.max(p))
            # jax.debug.print("VdW EOS — rho min/max = {}/{}", jnp.min(rho), jnp.max(rho))
            return p

        return map(eos_with_debug, self.a, self.b, self.R, rho_tree)

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

    def __init__(self, **kwargs):
        self.temperature_field_type = kwargs.get("temperature_field_type", "isothermal")
        self.T = kwargs.get("T")
        self.p_grid = jnp.asarray(kwargs.get("p_grid"))
        self.dpdT_grid = kwargs.get("dpdT_grid")

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
            self.T_grid = None
        else:
            if self.p_grid.shape[1] != self.rho_grid.shape[0]:
                raise ValueError("2D p_grid must have shape (len(T_grid), len(rho_grid))")
            if self.T_grid is None:
                raise ValueError("T_grid must be provided when p_grid is 2D")

            self.T_grid = jnp.asarray(self.T_grid)
            if self.T_grid.ndim != 1 or self.T_grid.shape[0] != self.p_grid.shape[0]:
                raise ValueError("T_grid must be 1D with length p_grid.shape[0]")

            t_sort_idx = jnp.argsort(self.T_grid)
            self.T_grid = self.T_grid[t_sort_idx]
            self.p_grid = self.p_grid[t_sort_idx, :][:, sort_idx]

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

    def _interp_rho(self, table, rho):
        return jnp.interp(rho, self.rho_grid, table)

    def _interp_rho_T(self, table, rho, T):
        values_at_T = jax.vmap(lambda table_row: jnp.interp(rho, self.rho_grid, table_row))(table)
        return jnp.interp(T, self.T_grid, values_at_T)

    def _lookup(self, table, rho, T):
        if self.T_grid is None:
            return self._interp_rho(table, rho)
        return self._interp_rho_T(table, rho, T)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        return map(lambda rho: self._lookup(self.p_grid, rho, self.T), rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        return map(lambda rho: self._lookup(self.p_grid, rho, T), rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def drho_dT(self, rho_tree, T):
        return map(lambda rho: self._lookup(self.dpdT_grid, rho, T), rho_tree)


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
