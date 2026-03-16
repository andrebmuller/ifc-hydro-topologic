"""
Graph data structure for representing hydraulic system topology.

This module implements an undirected graph structure used to represent
connections between hydraulic components in the system.
"""

from collections import defaultdict


class Graph(object):
    """
    Graph data structure for representing hydraulic system topology.

    This class implements an undirected graph by default, used to represent
    connections between hydraulic components in the system.

    Attributes:
        _graph (defaultdict): Internal graph representation using adjacency lists
        _directed (bool): Flag indicating if the graph is directed
    """

    def __init__(self, connections: list, directed: bool = False):
        """
        Initialize the graph with connections.

        Args:
            connections (list): List of tuple pairs representing connections
            directed (bool, optional): Whether the graph is directed. Defaults to False.
        """
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections: list):
        """
        Add multiple connections to the graph.

        Args:
            connections (list): List of tuple pairs representing node connections
        """
        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """
        Add a single connection between two nodes.

        Args:
            node1: First node to connect
            node2: Second node to connect
        """
        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """
        Remove all references to a node from the graph.

        Args:
            node: The node to remove
        """
        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2) -> bool:
        """
        Check if two nodes are directly connected.

        Args:
            node1: First node
            node2: Second node

        Returns:
            bool: True if nodes are directly connected, False otherwise
        """
        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path: list = []) -> list:
        """
        Find any path between two nodes using depth-first search.

        Note: This may not be the shortest path.

        Args:
            node1: Starting node
            node2: Destination node
            path (list, optional): Current path being explored. Defaults to [].

        Returns:
            list: Path from node1 to node2, or None if no path exists
        """
        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self) -> str:
        """
        String representation of the graph.

        Returns:
            str: String representation showing class name and graph structure
        """
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
