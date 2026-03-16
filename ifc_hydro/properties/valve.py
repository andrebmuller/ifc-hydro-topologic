"""
Valve property extraction module.

This module provides methods to extract geometric and type properties
from valves in the IFC model.
"""

from ..core.base import Base
from ..core.vector import Vector


class Valve:
    """
    Extracts properties from IFC valves.

    This class provides methods to extract geometric and type properties
    from valves in the IFC model.
    """

    @staticmethod
    def properties(valv, path: list) -> dict:
        """
        Extract properties from a valve for a specific path.

        Analyzes the valve's position in the provided hydraulic path to determine
        dimensions and flow directions.

        Args:
            valv: IFC valve object
            path (list): The specific hydraulic path containing this valve

        Returns:
            dict: Dictionary containing dimensions ('dim'), direction ('dir'), and type ('type')

        Raises:
            ValueError: If valve is not found in path or IFC properties cannot be extracted
        """
        Base.append_log(None, f"> Getting valve properties for valve with ID {valv.id()}...")
        valv_prop = {}

        # Find valve position in the provided path
        valv_index = None
        for i, component in enumerate(path):
            if component.id() == valv.id():
                valv_index = i
                break

        if valv_index is None:
            error_msg = f"> ERROR: Valve with ID {valv.id()} not found in the provided path"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Validate that valve has adjacent components
        if valv_index == 0 or valv_index >= len(path) - 1:
            error_msg = f"> ERROR: Valve with ID {valv.id()} does not have both incoming and outgoing pipes in the path"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Get adjacent components (incoming pipe, valve, outgoing pipe)
        incoming_pipe = path[valv_index - 1]
        outgoing_pipe = path[valv_index + 1]

        # Extract diameters from adjacent pipes with error handling
        try:
            pipe_dim_1 = incoming_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
            pipe_dim_2 = outgoing_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
            valv_prop['dim'] = (round(pipe_dim_1, 3), round(pipe_dim_2, 3))
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract diameter from adjacent pipes for valve ID {valv.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Calculate unit vectors and angles for flow direction change
        # Get center points from IFC geometry with error handling
        try:
            incoming_pipe_center = incoming_pipe[6][2][0][3][0][1][0][0]
            valve_center = valv[5][1][0][0]
            outgoing_pipe_center = outgoing_pipe[6][2][0][3][0][1][0][0]
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract center points for valve ID {valv.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Create direction vectors between points
        # Incoming: from incoming pipe center TO valve center
        incoming_dir = Vector.create_direction_vector(incoming_pipe_center, valve_center)
        # Outgoing: from valve center TO outgoing pipe center
        outgoing_dir = Vector.create_direction_vector(valve_center, outgoing_pipe_center)

        # Normalize to unit vectors
        incoming_unit = Vector.normalize(incoming_dir)
        outgoing_unit = Vector.normalize(outgoing_dir)

        # Calculate angle between vectors
        angle = Vector.angle_between(incoming_dir, outgoing_dir)

        # Store as dictionary with all relevant information
        valv_prop['dir'] = {
            'incoming_unit_vector': incoming_unit,
            'outgoing_unit_vector': outgoing_unit,
            'direction_change_angle': angle
        }

        # Extract valve type with error handling
        try:
            valv_type = valv[8]
            valv_prop['type'] = valv_type
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> WARNING: Failed to extract type for valve ID {valv.id()}. Using 'Unknown' as type. Details: {str(e)}"
            Base.append_log(None, error_msg)
            valv_prop['type'] = 'Unknown'

        Base.append_log(None, f"> Valve properties:")
        Base.append_log(None, f"> {valv_prop}")
        return valv_prop
