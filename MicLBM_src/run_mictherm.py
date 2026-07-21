try:
    from .MicTherm_API import compute_mictherm
except ImportError:
    if __package__:
        raise
    from MicTherm_API import compute_mictherm


def extract_mictherm_properties(properties, values, first_property_column=1):
    """Return MicTherm output columns keyed by their requested property names.

    ``properties`` may be a comma-separated string (as used in the MicTherm
    parameters) or an iterable of names. MicTherm's user-properties output has
    an input column before the requested properties by default, hence
    ``first_property_column=1``.
    """
    if isinstance(properties, str):
        property_names = [name.strip() for name in properties.split(",") if name.strip()]
    else:
        property_names = [str(name).strip() for name in properties if str(name).strip()]

    if not property_names:
        raise ValueError("At least one MicTherm property must be specified")
    if len(property_names) != len(set(property_names)):
        raise ValueError(f"MicTherm property names must be unique: {property_names}")

    if getattr(values, "ndim", None) != 2:
        raise ValueError("MicTherm values must be a two-dimensional array")

    last_property_column = first_property_column + len(property_names)
    if values.shape[1] < last_property_column:
        raise ValueError(
            f"MicTherm returned {values.shape[1]} columns, but columns "
            f"{first_property_column} through {last_property_column - 1} are required "
            f"for {property_names}"
        )

    return {
        name: values[:, first_property_column + index]
        for index, name in enumerate(property_names)
    }


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
