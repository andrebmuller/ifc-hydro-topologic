"""
Fitting property extraction module.

This module provides methods to extract geometric and type properties
from pipe fittings in the IFC model.
"""

from ..core.base import Base
from ..core.vector import Vector
from ..core.geom import Geom


class Fitting:
    """
    Extracts properties from IFC pipe fittings.

    This class provides methods to extract geometric and type properties
    from pipe fittings in the IFC model.
    """

    @staticmethod
    def properties(fitt, path: list) -> dict:
        """
        Extract properties from a pipe fitting for a specific path.

        Analyzes the fitting's position in the provided hydraulic path to determine
        dimensions, directions, and type.

        Args:
            fitt: IFC pipe fitting object
            path (list): The specific hydraulic path containing this fitting

        Returns:
            dict: Dictionary containing dimensions ('dim'), directions ('dir'), and type ('type')

        Raises:
            ValueError: If fitting is not found in path or IFC properties cannot be extracted
        """
        Base.append_log(None, f"> Getting fitting properties for fitting with ID {fitt.id()}...")
        fitt_prop = {}

        # Find fitting position in the provided path
        fitt_index = None
        for i, component in enumerate(path):
            if component.id() == fitt.id():
                fitt_index = i
                break

        if fitt_index is None:
            error_msg = f"> ERROR: Fitting with ID {fitt.id()} not found in the provided path"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Validate that fitting has adjacent components
        if fitt_index == 0 or fitt_index >= len(path) - 1:
            error_msg = f"> ERROR: Fitting with ID {fitt.id()} does not have both incoming and outgoing pipes in the path"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Get adjacent components (incoming pipe, fitting, outgoing pipe)
        incoming_pipe = path[fitt_index - 1]
        outgoing_pipe = path[fitt_index + 1]

        # Extract diameters from adjacent pipes with error handling
        try:
            try:

                if incoming_pipe.is_a() == 'IfcArbitraryClosedProfileDef':
                    pipe_dim_1 = incoming_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
                    pipe_dim_2 = outgoing_pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
                    fitt_prop['dim'] = (round(pipe_dim_1, 3), round(pipe_dim_2, 3))

                elif incoming_pipe.is_a() == 'IfcCircleProfileDef':
                    pipe_dim_1 = incoming_pipe[6][2][0][3][0][0][3] * 2
                    pipe_dim_2 = outgoing_pipe[6][2][0][3][0][0][3] * 2
                    fitt_prop['dim'] = (round(pipe_dim_1, 3), round(pipe_dim_2, 3))

            except (NotImplementedError, TypeError) as e:
                error_msg = f"> ERROR: Representation type not yet implemented: {[6][2][0][3][0][0].is_a()}. Details: {str(e)}"
                Base.append_log(None, error_msg)
                raise NotImplementedError(error_msg)

        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract diameter from adjacent pipes for fitting ID {fitt.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Calculate unit vectors and angles for flow direction change
        # Get center points from IFC geometry with error handling
        try:

            incoming_pipe_center = Geom.get_bbox_center(incoming_pipe)
            outgoing_pipe_center = Geom.get_bbox_center(outgoing_pipe)
            fitting_center = Geom.get_bbox_center(fitt)

            '''
            # Legacy method using hardcoded IFC structure indices
            incoming_pipe_center = incoming_pipe[6][2][0][3][0][1][0][0]
            fitting_center = fitt[5][1][0][0]
            outgoing_pipe_center = outgoing_pipe[6][2][0][3][0][1][0][0]
            '''
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract center points for fitting ID {fitt.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Create direction vectors between points
        # Incoming: from incoming pipe center TO fitting center
        incoming_dir = Vector.create_direction_vector(incoming_pipe_center, fitting_center)
        # Outgoing: from fitting center TO outgoing pipe center
        outgoing_dir = Vector.create_direction_vector(fitting_center, outgoing_pipe_center)

        # Normalize to unit vectors
        incoming_unit = Vector.normalize(incoming_dir)
        outgoing_unit = Vector.normalize(outgoing_dir)

        # Calculate angle between vectors
        angle = Vector.angle_between(incoming_dir, outgoing_dir)

        # Store as dictionary with all relevant information
        fitt_prop['dir'] = {
            'incoming_unit_vector': incoming_unit,
            'outgoing_unit_vector': outgoing_unit,
            'direction_change_angle': angle
        }

        # Extract fitting type from IFC properties with error handling
        try:
            fitt_type = fitt[8]
            fitt_prop['type'] = fitt_type
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> WARNING: Failed to extract type for fitting ID {fitt.id()}. Using 'Unknown' as type. Details: {str(e)}"
            Base.append_log(None, error_msg)
            fitt_prop['type'] = 'Unknown'

        Base.append_log(None, f"> Fitting properties:")
        Base.append_log(None, f"> {fitt_prop}")
        return fitt_prop
