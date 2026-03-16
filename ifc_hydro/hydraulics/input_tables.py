"""
Input tables for hydraulic calculations.

This module contains reference tables used in hydraulic calculations,
including pressure drop coefficients, design flow rates, and pipe dimensions.

Equivalent length values are sourced from standard reference tables for
PVC pipes (NBR 5626) and are indexed by nominal diameter in meters.
"""


# Nominal pipe diameter to real internal diameter mapping (in meters)
# Keys: nominal diameter (m), Values: actual internal diameter (m)
# Source: Tigre PVC pipe specifications
internal_diameter_table = {
    0.020: 0.0170,
    0.025: 0.0216,
    0.032: 0.0278,
    0.040: 0.0352,
    0.050: 0.0440,
    0.065: 0.0666,
    0.075: 0.0756,
    0.110: 0.0978,
}

"""
# Nominal pipe diameter to real internal diameter mapping (in meters)
# Keys: nominal diameter (m), Values: actual internal diameter (m)
# Source: NBR 5648/2018 - Table 2
internal_diameter_table = {
    0.015: 0.0170,
    0.020: 0.0216,
    0.025: 0.0278,
    0.032: 0.0352,
    0.040: 0.0440,
    0.050: 0.0534,
    0.065: 0.0666,
    0.075: 0.0756,
    0.100: 0.0978,
}
"""

# Equivalent length factors for fittings (in meters)
# Structure: {fitting_type: {angle: {nominal_diameter: equivalent_length}}}
# For non-angle types: {fitting_type: {nominal_diameter: equivalent_length}}
# Nominal diameters in meters: 0.015, 0.020, 0.025, 0.032, 0.040, 0.050, 0.065, 0.075, 0.100
fitting_pressure_drop_table = {
    'JUNCTION': {
        0: {
            0.015: 0.7, 0.020: 0.8, 0.025: 0.9, 0.032: 1.5,
            0.040: 1.5, 0.050: 2.2, 0.065: 2.3, 0.075: 3.0, 0.100: 3.4
        },
        180: {
            0.015: 0.7, 0.020: 0.8, 0.025: 0.9, 0.032: 1.5,
            0.040: 1.5, 0.050: 2.2, 0.065: 2.3, 0.075: 3.0, 0.100: 3.4
        },
        90: {
            0.015: 2.3, 0.020: 2.4, 0.025: 3.1, 0.032: 3.5,
            0.040: 4.6, 0.050: 5.1, 0.065: 6.9, 0.075: 7.2, 0.100: 9.5
        },
    },
    'BEND': {
        90: {
            0.015: 1.1, 0.020: 1.2, 0.025: 1.5, 0.032: 2.0,
            0.040: 2.1, 0.050: 2.7, 0.065: 3.4, 0.075: 3.7, 0.100: 4.9
        },
        45: {
            0.015: 0.4, 0.020: 0.5, 0.025: 0.7, 0.032: 0.9,
            0.040: 1.0, 0.050: 1.3, 0.065: 1.6, 0.075: 1.8, 0.100: 2.4
        },
    },
    'EXIT': {
        0.015: 0.7, 0.020: 1.0, 0.025: 1.2, 0.032: 1.5,
        0.040: 2.0, 0.050: 2.5, 0.065: 3.2, 0.075: 3.7, 0.100: 4.9
    },
    'ENTRY': {
        0.015: 0.2, 0.020: 0.3, 0.025: 0.4, 0.032: 0.5,
        0.040: 0.5, 0.050: 0.7, 0.065: 0.8, 0.075: 0.9, 0.100: 1.3
    },
}

# Equivalent length factors for valves (in meters)
# Structure: {valve_type: {nominal_diameter: equivalent_length}}
# Nominal diameters in meters: 0.015, 0.020, 0.025, 0.032, 0.040, 0.050, 0.065, 0.075, 0.100
valve_pressure_drop_table = {
    'ISOLATING': {
        0.015: 0.1, 0.020: 0.1, 0.025: 0.2, 0.032: 0.2,
        0.040: 0.3, 0.050: 0.3, 0.065: 0.4, 0.075: 0.5, 0.100: 0.6
    },
    'REGULATING': {
        0.015: 4.9, 0.020: 6.7, 0.025: 8.2, 0.032: 10.8,
        0.040: 13.0, 0.050: 17.0, 0.065: 21.0, 0.075: 24.0, 0.100: 30.0
    },
    'CHECK': {
        0.015: 2.5, 0.020: 3.2, 0.025: 4.1, 0.032: 5.5,
        0.040: 6.4, 0.050: 8.3, 0.065: 10.3, 0.075: 11.7, 0.100: 15.7
    },
}

# Score values by sanitary terminal type (used for design flow calculations)
score_table = {
    'BATH': 1.0,
    'BIDET': 0.1,
    'CISTERN': 0.7,
    'SANITARYFOUNTAIN': 0.1,
    'SHOWER': 0.1,
    'SINK': 0.7,
    'TOILETPAN': 0.3,
    'URINAL': 2.8,
    'WASHHANDBASIN': 0.3,
    'WCSEAT': 32,
    # Use for washing machines, dishwashers, so on
    'USERDEFINED': 1.0,
    # Use for any terminal types not explicitly listed above
    'NOTDEFINED': 0.4
}

# Tank height adjustment (meters)
# Accounts for water level above the pipe connection point at the tank bottom
# This value represents the static head contribution from the water column
# Inside the tank, measured from the outlet pipe connection to the water surface
tank_height_adjustment = 0.5
