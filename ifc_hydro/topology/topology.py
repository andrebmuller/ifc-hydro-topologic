"""
Topology creation module for hydraulic systems.

This module analyzes IFC files to extract hydraulic component relationships
and create a graph representation of the system topology.
"""

from ..core.base import Base
from ..core.graph import Graph


class Topology:
    """
    Creates hydraulic system topology from IFC models.

    This class analyzes IFC files to extract hydraulic component relationships
    and create a graph representation of the system topology.

    Attributes:
        model: The loaded IFC model
    """

    def __init__(self, model):
        """
        Initialize the Topology with an IFC model.

        Args:
            model: An opened IFC model object (from ifcopenshell)
        """
        self.model = model

    def graph_creator(self) -> Graph:
        """
        Create a graph representing the hydraulic system topology.

        Analyzes the IFC model to find connections between hydraulic components
        using IfcRelNests and IfcRelConnectsPorts relationships.

        Returns:
            Graph: Undirected graph representing the hydraulic system topology

        Raises:
            ValueError: If required IFC relationships are not found in the model
        """
        model = self.model

        connections = []

        # Extract nest and connection relationships from IFC model
        nest_list = model.by_type("IfcRelNests")
        conn_list = model.by_type("IfcRelConnectsPorts")

        # Validate required IFC relationships exist
        if not nest_list:
            error_msg = "> ERROR: No IfcRelNests relationships found in the IFC model. The model must contain nesting relationships for topology creation."
            Base.append_log(Base, error_msg)
            raise ValueError(error_msg)

        if not conn_list:
            error_msg = "> ERROR: No IfcRelConnectsPorts relationships found in the IFC model. The model must contain port connections for topology creation."
            Base.append_log(Base, error_msg)
            raise ValueError(error_msg)

        # Create connections by analyzing port relationships and nesting
        for conn in conn_list:
            for nest in nest_list:
                for other_nest in nest_list:
                    if conn[4] in nest[5] or conn[5] in nest[5]:
                        if conn[5] in other_nest[5] or conn[4] in other_nest[5]:
                            nest1 = nest
                            nest2 = other_nest

                if nest1 != nest2:
                    connections.append((nest1[4], nest2[4]))

        graph = Graph(connections)

        return graph

    def path_finder(self, term_guid: str, tank_guid: str) -> list:
        """
        Find the path between a specific sanitary terminal and tank.

        Args:
            term_guid (str): GUID of the sanitary terminal
            tank_guid (str): GUID of the tank

        Returns:
            list: Path from terminal to tank, wrapped in a list

        Raises:
            ValueError: If terminal or tank GUID is not found in the model
        """
        model = self.model
        graph = self.graph_creator()

        path = []

        # Find components by GUID and calculate path
        try:
            term = model.by_guid(term_guid)
        except RuntimeError:
            error_msg = f"> ERROR: Sanitary terminal with GUID '{term_guid}' not found in the IFC model."
            Base.append_log(Base, error_msg)
            raise ValueError(error_msg)

        try:
            tank = model.by_guid(tank_guid)
        except RuntimeError:
            error_msg = f"> ERROR: Tank with GUID '{tank_guid}' not found in the IFC model."
            Base.append_log(Base, error_msg)
            raise ValueError(error_msg)

        path.append(graph.find_path(term, tank))

        return path

    def all_paths_finder(self) -> list:
        """
        Find paths between all sanitary terminals and tanks in the system.

        Returns:
            list: List of all paths from terminals to tanks

        Raises:
            ValueError: If no sanitary terminals or tanks are found in the model
        """
        model = self.model
        graph = self.graph_creator()

        all_paths = []

        # Get all terminals and tanks from the model
        term_list = model.by_type("IfcSanitaryTerminal")
        tank_list = model.by_type("IfcTank")

        # Validate required IFC elements exist
        if not term_list:
            error_msg = "> ERROR: No IfcSanitaryTerminal elements found in the IFC model. The model must contain at least one sanitary terminal for hydraulic analysis."
            Base.append_log(Base, error_msg)
            raise ValueError(error_msg)

        if not tank_list:
            error_msg = "> ERROR: No IfcTank elements found in the IFC model. The model must contain at least one tank for hydraulic analysis."
            Base.append_log(Base, error_msg)
            raise ValueError(error_msg)

        Base.append_log(Base, f"> Creating topology...")
        Base.append_log(Base, f"> Found {len(term_list)} sanitary terminal(s) and {len(tank_list)} tank(s)...")

        # Calculate paths between all terminal-tank combinations
        for tank in tank_list:
            for term in term_list:
                all_paths.append(graph.find_path(term, tank))

        Base.append_log(Base, f"> Topology created with {len(all_paths)} paths...")
        Base.append_log(Base, f"{'-'*100}")

        return all_paths
