
import os
# ensure MATLAB Runtime paths are included in PATH for Windows
# before running code, read the Mictherm API manual to installation instructions
os.environ["PATH"] = (
    r"C:\Program Files\MATLAB\MATLAB Runtime\R2024a\runtime\win64;"
    r"C:\Program Files\MATLAB\MATLAB Runtime\R2024a\bin\win64;"
    r"C:\Program Files\MATLAB\MATLAB Runtime\R2024a\sys\os\win64;"
    + os.environ["PATH"]
)

import numpy
import MicTherm
import matlab


class MatlabDoubleConverter:
    """Helper class to automatically convert Python values to matlab.double format."""
    
    @staticmethod
    def convert(value):
        """
        Convert Python values to matlab.double (Nx1 column format).
        
        Args:
            value: Can be int, float, list, tuple, or already matlab.double
            
        Returns:
            matlab.double: Column vector in Nx1 format
        """
        if isinstance(value, matlab.double):
            return value
        
        if isinstance(value, (int, float)):
            return matlab.double([[value]])
        
        if isinstance(value, numpy.ndarray):
            if value.size == 0:
                return matlab.double([])

            # Convert numpy arrays to an Nx1 column vector.
            return matlab.double([[item] for item in value.reshape(-1).tolist()])

        if isinstance(value, (list, tuple)):
            if not value:  # Empty list/tuple
                return matlab.double([])
            # Convert to Nx1 format (column vector)
            return matlab.double([[v] for v in value])
        
        raise TypeError(f"Unsupported type: {type(value)}")


class MicThermAPIClient:
    # substance properties for API calls (can be overridden by user parameters
    DEFAULT_USER_PARAMETERS = {
        # General settings for VdW EOS from reference paper 
        "N_components": 1,
        "Substance_ID1": 0,
        "PotModel_1": "LJ.pm",
        "units": "SI",
        "EOS": "vanderWaals",
        "Output": "no",
        "Debug": "no",

        "chainlength_1": 1,
        "b_VDW_1": 0.0952,
        "a_VDW_1": 0.1837,
        "molar_mass_1": 114.04,
        "dT": 1.0,
    }
    # method-specific parameters to be added on top of base_user_parameters
    def __init__(self, base_user_parameters=None, **user_parameter_kwargs):
        # (1) API initialisieren
        self.api = MicTherm.initialize()

        self.base_user_parameters = self._build_base_user_parameters(base_user_parameters)

        self.mode_parameters = {
            "userproperties": [
                "APIMode = UserProperties",
                "properties = p",
            ],
            "criticalpoint": [
                "calculationmode = critical_point",
            ],
            "vle_full": [
                "calculationmode = phaseEquilib",
            ],
            "vle_iso": [
                "calculationmode = phaseEquilib",
            ],
        }

        self.user_parameter_kwargs = user_parameter_kwargs

    @staticmethod
    def _format_user_parameter(key, value):
        if isinstance(value, bool):
            value = "yes" if value else "no"
        return f"{key} = {value}"

    @classmethod
    def _build_base_user_parameters(cls, base_user_parameters):
        if base_user_parameters is None:
            return [
                cls._format_user_parameter(key, value)
                for key, value in cls.DEFAULT_USER_PARAMETERS.items()
            ]

        if isinstance(base_user_parameters, dict):
            parameters = {**cls.DEFAULT_USER_PARAMETERS, **base_user_parameters}
            return [
                cls._format_user_parameter(key, value)
                for key, value in parameters.items()
                if value is not None
            ]

        return list(base_user_parameters)

    def build_user_parameters(self, mode, t_iso=None, **user_parameter_kwargs):
        mode_key = mode.lower()
        if mode_key not in self.mode_parameters:
            raise ValueError("mode must be 'userproperties', 'criticalpoint', 'VLE_full', or 'VLE_Iso'")
        user_parameters = self.base_user_parameters + self.mode_parameters[mode_key]

        if mode_key == "vle_iso" and t_iso is not None:
            user_parameters = user_parameters + [f"T_iso = {t_iso}"]

        extra_parameters = {**self.user_parameter_kwargs, **user_parameter_kwargs}
        user_parameters.extend(
            self._format_user_parameter(key, value)
            for key, value in extra_parameters.items()
            if value is not None
        )

        return user_parameters

    def call_example(self, init_mode, rho, T, p, x, mode, t_iso=None, **user_parameter_kwargs):
        user_parameters_in = self.build_user_parameters(mode, t_iso=t_iso, **user_parameter_kwargs)
        return self.api.API_example(
            init_mode,
            rho,
            T,
            p,
            x,
            user_parameters_in,
            nargout=3,
        )


def compute_mictherm(
    T,
    rho,
    p,
    x,
    mode,
    t_iso=None,
    init_mode="uninitialized",
    base_user_parameters=None,
    **user_parameter_kwargs,
):
    """
    Call MicTherm API with given inputs and return Name, Units, Value.

    Inputs T, rho, p, x can be Python scalars/lists/tuples or matlab.double.
    Value is returned as a numpy array.
    """
    client = MicThermAPIClient(
        base_user_parameters=base_user_parameters,
        **user_parameter_kwargs,
    )

    T_in = T if isinstance(T, matlab.double) else MatlabDoubleConverter.convert(T)
    rho_in = rho if isinstance(rho, matlab.double) else MatlabDoubleConverter.convert(rho)
    p_in = p if isinstance(p, matlab.double) else MatlabDoubleConverter.convert(p)
    x_in = x if isinstance(x, matlab.double) else MatlabDoubleConverter.convert(x)

    Name, Units, Value = client.call_example(
        init_mode,
        rho_in,
        T_in,
        p_in,
        x_in,
        mode=mode,
        t_iso=t_iso,
        **user_parameter_kwargs,
    )

    return Name, Units, numpy.asarray(Value)

