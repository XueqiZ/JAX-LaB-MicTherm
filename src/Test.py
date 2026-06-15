
import os

os.environ["PATH"] = (
    r"C:\Program Files\MATLAB\MATLAB Runtime\R2024a\runtime\win64;"
    r"C:\Program Files\MATLAB\MATLAB Runtime\R2024a\bin\win64;"
    r"C:\Program Files\MATLAB\MATLAB Runtime\R2024a\sys\os\win64;"
    + os.environ["PATH"]
)

import numpy
import MicTherm
import matlab


# (1) API initialisieren
API = MicTherm.initialize()

userParametersIn = [
    # General settings
    "N_components = 1",
    "Substance_ID1 = 0",
    "PotModel_1 = LJ.pm",
    "units = SI",
    "EOS = vanderWaals",
    "APIMode = UserProperties",
    "Output = no",
    "Debug = no",
    "IDEAL = IdealQM",

    # Substance-specific parameters (R-1234yf)
    "chainlength_1 = 1",
    "b_VDW_1 = 0.0952",  # VDW_b
    "a_VDW_1 = 0.1837",  # VDW_a
    "molar_mass_1 = 114.04",
    "CAS_number_1 = 754-12-1",


    # Properties to calculate
    "properties = p"
]

initModeIn = "uninitialized"


# (2) Zustandspunkte definieren (N x 1 Format!)
rho = matlab.double([[0.5], [0.3], [1.2]])
T   = matlab.double([[300], [290], [310]])
p   = matlab.double([[1.0], [1.1], [1.2]])
x   = matlab.double([[1.0], [1.0], [1.0]])


# (3) Berechnung
Value, Name, Unit = API.API_example(
    initModeIn,
    rho,
    T,
    p,
    x,
    userParametersIn,
    nargout=3
)

# (4) Convert to numpy for easier handling
Value = numpy.asarray(Value)
print("Names:", Name)
print("Units:", Unit)
print("Values:\n", Value)
