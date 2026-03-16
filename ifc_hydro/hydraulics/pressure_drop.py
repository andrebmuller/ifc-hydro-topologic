"""
Pressure drop calculation module for water supply systems.

This module implements pressure drop analysis for pipes, fittings, and valves
using industry standard equations such as Fair Whipple-Hsiao.
"""

from ..core.base import Base
from .design_flow import DesignFlow
from .input_tables import fitting_pressure_drop_table, valve_pressure_drop_table, internal_diameter_table


class PressureDrop:
    """
    Calculates pressure drops in hydraulic system components.

    This class implements pressure drop calculations for pipes (linear losses)
    and fittings/valves (local losses) using industry standard equations.
    """

    def __init__(self) -> None:
        """Initialize the PressureDrop calculator."""
        self.design_flow = DesignFlow()

    def linear(self, pipe, all_paths: list) -> float:
        """
        Calculate linear pressure drop in a pipe using Fair Whipple-Hsiao equations.

        Implements the Fair Whipple-Hsiao equation for PVC pipes, recommended
        for pipes with diameter between 12.5 mm and 100 mm.

        Args:
            pipe: IFC pipe segment object
            all_paths (list): List of all hydraulic paths

        Returns:
            float: Linear pressure drop in meters of water column
        """
        from ..properties.pipe import Pipe
        
        # Initialize calculation components
        pipe_prop = Pipe.properties(pipe)
        flow = self.design_flow.calculate(all_paths)
        score_sum = 0

        Base.append_log(self, f"> Getting linear pressure drop for pipe with ID {pipe.id()}...")

        # Calculate cumulative design flow for the specified pipe
        for path in flow:
            for component in path:
                if component[0] == pipe[0]:
                    score_sum += component[1]

        pipe_length = pipe_prop.get('len')
        internal_diameter = pipe_prop.get('dim')

        Base.append_log(self, f"> Pipe length: {pipe_length} m, Internal diameter: {internal_diameter * 1000:.1f} mm")

        design_flow = 0.3 * (score_sum ** 0.5)
        Base.append_log(self, f"> Design flow: {round(design_flow, 3)} L/s")

        # Fair Whipple-Hsiao equation for PVC pipes
        # J = 0.000869 * Q^1.75 * D^-4.75
        # Recommended for pipes with d between 12.5 mm and 100 mm
        unit_loss = 0.000869 * ((design_flow * 0.001) ** 1.75) * (internal_diameter ** -4.75)
        pressure_drop = pipe_length * unit_loss

        # Legacy Hazen-Williams equation
        # pressure_drop = (10.67 * pipe_prop.get('len') * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (pipe_prop.get('dim') ** 4.87))

        Base.append_log(self, f"> Fair Whipple-Hsiao: J = 0.000859 * Q^1.75 * D^-4.75 = {round(unit_loss, 6)} m/m")
        Base.append_log(self, f"> Linear pressure drop: {pipe_length} * {round(unit_loss, 6)} = {round(pressure_drop, 3)} m")
        return pressure_drop

    def local(self, conn, path: list, all_paths: list) -> float:
        """
        Calculate local pressure drop in fittings and valves using equivalent length method.

        Uses tabulated equivalent length values for different connection types,
        indexed by nominal diameter, and applies the Fair Whipple-Hsiao equation.

        Args:
            conn: IFC connection object (fitting or valve)
            path (list): The specific hydraulic path containing this connection
            all_paths (list): List of all hydraulic paths (for flow calculation)

        Returns:
            float: Local pressure drop in meters of water column
        """
        from ..properties.fitting import Fitting
        from ..properties.valve import Valve
        
        # Initialize calculation components
        flow = self.design_flow.calculate(all_paths)
        score_sum = 0

        Base.append_log(self, f"> Getting local pressure drop for connection with ID {conn.id()}...")

        # Calculate cumulative design flow for the specified connection
        for flow_path in flow:
            for component in flow_path:
                if component[0] == conn[0]:
                    score_sum += component[1]

        # Get connection properties and select the appropriate table
        if conn.is_a() == 'IfcValve':
            conn_prop = Valve.properties(conn, path)
            pressure_drop_table = valve_pressure_drop_table
            table_name = 'valve'
        elif conn.is_a() == 'IfcPipeFitting':
            conn_prop = Fitting.properties(conn, path)
            pressure_drop_table = fitting_pressure_drop_table
            table_name = 'fitting'
        else:
            return 0

        # Get nominal diameter from adjacent pipe dimensions
        conn_dims = conn_prop.get('dim', (0.025, 0.025))
        nominal_diameter = round(conn_dims[0], 3)

        # Look up internal diameter for the equation
        internal_diameter = internal_diameter_table.get(nominal_diameter)
        if internal_diameter is None:
            Base.append_log(self, f"> WARNING: No internal diameter mapping for nominal {nominal_diameter * 1000:.1f} mm. Using nominal as fallback.")
            internal_diameter = nominal_diameter

        conn_type = conn_prop.get('type')

        design_flow = 0.3 * (score_sum ** 0.5)
        Base.append_log(self, f"> Design flow: {round(design_flow, 3)} L/s")

        # Look up equivalent length from the appropriate table
        table_value = pressure_drop_table.get(conn_type)

        if table_value is None:
            Base.append_log(self, f"> WARNING: No {table_name} table entry found for type '{conn_type}'. Returning 0.")
            return 0

        # Resolve the coefficient based on table structure
        # Two-level dict: angle → {diameter → coefficient} (for JUNCTION, BEND)
        # One-level dict: {diameter → coefficient} (for EXIT, ENTRY, valves)
        first_value = next(iter(table_value.values()))
        if isinstance(first_value, dict):
            # Angle-based: get direction change angle, then look up by diameter
            direction_info = conn_prop.get('dir', {})
            direction_angle = direction_info.get('direction_change_angle', None)

            angle_data = table_value.get(direction_angle)
            if angle_data is None:
                Base.append_log(self, f"> WARNING: No entry for angle {direction_angle} in {table_name} type '{conn_type}'. Returning 0.")
                return 0

            coefficient = angle_data.get(nominal_diameter)
            Base.append_log(self, f"> Equivalent length lookup: type={conn_type}, angle={direction_angle}, diameter={nominal_diameter * 1000:.1f} mm -> {coefficient} m")
        else:
            # Direct diameter lookup
            coefficient = table_value.get(nominal_diameter)
            Base.append_log(self, f"> Equivalent length lookup: type={conn_type}, diameter={nominal_diameter * 1000:.1f} mm -> {coefficient} m")

        if coefficient is None:
            Base.append_log(self, f"> WARNING: No coefficient found for type '{conn_type}' at nominal diameter {nominal_diameter * 1000:.1f} mm. Returning 0.")
            return 0

        # Fair Whipple-Hsiao equation for PVC pipes
        # J = 0.000869 * Q^1.75 * D^-4.75
        # Recommended for pipes with d between 12.5 mm and 100 mm
        unit_loss = 0.000869 * ((design_flow * 0.001) ** 1.75) * (internal_diameter ** -4.75)
        pressure_drop = coefficient * unit_loss

        # Hazen-Williams equation with equivalent length for PVC (C = 140)
        # pressure_drop = (10.67 * coefficient * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (internal_diameter ** 4.87))

        Base.append_log(self, f"> Fair Whipple-Hsiao: J = 0.000859 * Q^1.75 * D^-4.75 = {round(unit_loss, 6)} m/m")
        Base.append_log(self, f"> Local pressure drop: {coefficient} * {round(unit_loss, 6)} = {round(pressure_drop, 3)} m")
        return pressure_drop
