"""
Pipe property extraction module.

This module provides methods to extract geometric and type properties
from pipe segments in the IFC model.
"""

from ..core.base import Base
from ..hydraulics.input_tables import internal_diameter_table


class Pipe:
    """
    Extracts properties from IFC pipe segments.

    This class provides methods to extract geometric properties including
    length and diameter from pipe segments in the IFC model.
    """

    @staticmethod
    def properties(pipe) -> dict:
        """
        Extract properties from a pipe segment.

        Args:
            pipe: IFC pipe segment object

        Returns:
            dict: Dictionary containing pipe length ('len') and diameter ('dim')

        Raises:
            ValueError: If pipe properties cannot be extracted from the IFC element
        """
        Base.append_log(None, f"> Getting pipe properties for pipe with ID {pipe.id()}...")
        pipe_prop = {}

        # Extract pipe length from IFC geometry representation with error handling
        try:
            pipe_len = pipe[6][2][0][3][0][3]
            pipe_prop['len'] = round(pipe_len, 3)
        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract length from pipe ID {pipe.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Extract pipe diameter (radius * 2) from IFC geometry with error handling
        try:
            try:

                if pipe[6][2][0][3][0][0].is_a() == 'IfcArbitraryClosedProfileDef':
                    pipe_dim = pipe[6][2][0][3][0][0][2][0][0][0][0] * 2
                elif pipe[6][2][0][3][0][0].is_a() == 'IfcCircleProfileDef':
                    pipe_dim = pipe[6][2][0][3][0][0][3] * 2

            except (NotImplementedError, TypeError) as e:
                error_msg = f"> ERROR: Representation type not yet implemented: {[6][2][0][3][0][0].is_a()}. Details: {str(e)}"
                Base.append_log(None, error_msg)
                raise NotImplementedError(error_msg)


        except (IndexError, TypeError, KeyError) as e:
            error_msg = f"> ERROR: Failed to extract diameter from pipe ID {pipe.id()}. IFC geometry structure may be invalid. Details: {str(e)}"
            Base.append_log(None, error_msg)
            raise ValueError(error_msg)

        # Map nominal pipe diameter to real internal diameter using the input table
        nominal_dim = round(pipe_dim, 3)
        real_dim = internal_diameter_table.get(nominal_dim, 0.000)

        if real_dim == 0.000:
            Base.append_log(None, f"> WARNING: Nominal diameter {nominal_dim * 1000:.1f} mm not found in internal diameter table. Using 0.000 m.")
        else:
            Base.append_log(None, f"> Mapped nominal diameter {nominal_dim * 1000:.1f} mm to internal diameter...")

        pipe_prop['dim'] = real_dim

        Base.append_log(None, f"> Pipe properties:")
        Base.append_log(None, f"> {pipe_prop}")
        return pipe_prop
