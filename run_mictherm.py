import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from MicTherm_API import compute_mictherm


def run_mictherm_func(
    mode="criticalpoint",
    T=None,
    rho=None,
    p=None,
    x=None,
    t_iso=None,
    init_mode="uninitialized",
    print_output=True,
):
    """Run compute_mictherm with provided inputs and return (names, units, values).

    Defaults match previous behavior. T, rho, p, x default to empty lists when None.
    """
    if T is None:
        T = []
    if rho is None:
        rho = []
    if p is None:
        p = []
    if x is None:
        x = []

    names, units, values = compute_mictherm(
        T=T,
        rho=rho,
        p=p,
        x=x,
        mode=mode,  # 'userproperties', 'criticalpoint', 'VLE_full', or 'VLE_Iso'
        init_mode=init_mode,
        t_iso=t_iso,
        # t_iso is available for callers if needed; pass via other interfaces if supported
    )

    if print_output:
        print("Names:", names)
        print("Units:", units)
        print("Values:")
        print(values)

    return names, units, values

def mictherm_grid(
    T=None,
    rho=None,
    p=None,
    x=None,
    mode="userproperties",
    step=None,
    rho_range=None,
    T_range=None,
    p_range=None,
    x_range=None,
):
    if rho_range is not None:
        if isinstance(rho_range, (int, float)):
            rho = np.full(step, rho_range)
        else:
            rho = np.linspace(rho_range[0], rho_range[1], step)
    if T_range is not None:
        if isinstance(T_range, (int, float)):
            T = np.full(step, T_range)
        else:
            T = np.linspace(T_range[0], T_range[1], step)
    if p_range is not None:
        if isinstance(p_range, (int, float)):
            p = np.full(step, p_range)
        else:
            p = np.linspace(p_range[0], p_range[1], step)
    if x_range is not None:
        if isinstance(x_range, (int, float)):
            x = np.full(step, x_range)
        else:
            x = np.linspace(x_range[0], x_range[1], step)
    
    names, units, values = run_mictherm_func(
        mode=mode,
        T=T,
        rho=rho,
        p=p,
        x=x,
        print_output=False,
    )
    return names, units, values, T, p, rho, x


def main():
    # preserve previous default behavior
    run_mictherm_func()


if __name__ == "__main__":
    main()
