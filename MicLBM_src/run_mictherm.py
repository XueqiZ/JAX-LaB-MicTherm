try:
    from .MicTherm_API import compute_mictherm
except ImportError:
    if __package__:
        raise
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
    **mictherm_kwargs,
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
        **mictherm_kwargs,
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
    print_output=False,
    **mictherm_kwargs,
):
    
    names, units, values = run_mictherm_func(
        mode=mode,
        T=T,
        rho=rho,
        p=p,
        x=x,
        print_output=print_output,
        **mictherm_kwargs,
    )
    return names, units, values, T, p, rho, x


def main():
    # preserve previous default behavior
    run_mictherm_func()


if __name__ == "__main__":
    main()
