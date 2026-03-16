"""
Graph plotting interface for hydraulic system topology visualization.

This module provides tools for visualizing hydraulic system graphs using
matplotlib and networkx. It supports various visualization modes including
coloring by component type, highlighting paths, and displaying flow/pressure data.

Dependencies:
    - matplotlib: For creating static visualizations
    - networkx: For graph manipulation and layout algorithms
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from ..core.graph import Graph


# Color scheme for different IFC component types
COMPONENT_COLORS = {
    'IfcSanitaryTerminal': '#4CAF50',  # Green - water outlets
    'IfcPipeSegment': '#2196F3',        # Blue - pipes
    'IfcPipeFitting': '#FF9800',        # Orange - fittings
    'IfcValve': '#F44336',              # Red - valves
    'IfcTank': '#9C27B0',               # Purple - tanks
    'IfcFlowController': '#795548',     # Brown - flow controllers
    'default': '#9E9E9E'                # Grey - unknown types
}

# Node shapes for component types (matplotlib markers)
COMPONENT_SHAPES = {
    'IfcSanitaryTerminal': 'o',  # Circle
    'IfcPipeSegment': 's',       # Square
    'IfcPipeFitting': '^',       # Triangle up
    'IfcValve': 'D',             # Diamond
    'IfcTank': 'p',              # Pentagon
    'default': 'o'               # Circle
}


class GraphPlotter:
    """
    Interface for plotting hydraulic system topology graphs.

    This class converts ifc_hydro Graph objects to networkx format and
    provides various visualization methods using matplotlib.

    Attributes:
        graph: The internal networkx graph representation
        node_labels: Dictionary mapping nodes to their display labels
        node_types: Dictionary mapping nodes to their IFC component types
    """

    def __init__(self, graph: Optional[Graph] = None):
        """
        Initialize the GraphPlotter.

        Args:
            graph: Optional ifc_hydro Graph object to convert and visualize
        """
        self.graph = nx.Graph()
        self.node_labels = {}
        self.node_types = {}
        self._paths = []

        if graph is not None:
            self.from_ifc_graph(graph)

    def from_ifc_graph(self, graph: Graph):
        """
        Convert an ifc_hydro Graph object to networkx format.

        Args:
            graph: The ifc_hydro Graph object to convert
        """
        self.graph = nx.Graph()
        self.node_labels = {}
        self.node_types = {}

        # Access the internal adjacency list structure
        adjacency = graph._graph

        for node, neighbors in adjacency.items():
            # Add node with attributes
            node_id = self._get_node_id(node)
            node_type = self._get_node_type(node)
            node_label = self._get_node_label(node)

            self.graph.add_node(node_id,
                               node_type=node_type,
                               label=node_label)
            self.node_labels[node_id] = node_label
            self.node_types[node_id] = node_type

            # Add edges
            for neighbor in neighbors:
                neighbor_id = self._get_node_id(neighbor)
                self.graph.add_edge(node_id, neighbor_id)

    def from_topology_paths(self, all_paths: List[List[Any]]):
        """
        Create graph from topology paths (from Topology.all_paths_finder()).

        This method builds the graph from a list of paths, where each path
        is a sequence of connected IFC components.

        Args:
            all_paths: List of paths from Topology.all_paths_finder()
        """
        self.graph = nx.Graph()
        self.node_labels = {}
        self.node_types = {}
        self._paths = all_paths

        for path in all_paths:
            if path is None:
                continue
            for i, node in enumerate(path):
                node_id = self._get_node_id(node)

                # Use node_types as the check since add_edge implicitly
                # adds nodes to the graph, bypassing our attribute setup
                if node_id not in self.node_types:
                    node_type = self._get_node_type(node)
                    node_label = self._get_node_label(node)
                    self.graph.add_node(node_id,
                                       node_type=node_type,
                                       label=node_label)
                    self.node_labels[node_id] = node_label
                    self.node_types[node_id] = node_type

                # Add edge to next node in path
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    next_id = self._get_node_id(next_node)
                    self.graph.add_edge(node_id, next_id)

    def _get_node_id(self, node) -> str:
        """Extract unique identifier from IFC node object."""
        try:
            # IFC objects have a GlobalId attribute
            if hasattr(node, 'GlobalId'):
                return node.GlobalId
            # Fallback to id()
            return str(node.id()) if hasattr(node, 'id') else str(id(node))
        except Exception:
            return str(id(node))

    def _get_node_type(self, node) -> str:
        """Extract IFC type from node object, mapping to known component types."""
        try:
            if hasattr(node, 'is_a'):
                ifc_type = node.is_a()
                # Direct match against known types
                if ifc_type in COMPONENT_COLORS:
                    return ifc_type
                # Check parent IFC types for subtypes
                parent_checks = [
                    ('IfcSanitaryTerminal', 'IfcSanitaryTerminal'),
                    ('IfcTank', 'IfcTank'),
                    ('IfcValve', 'IfcValve'),
                    ('IfcPipeSegment', 'IfcPipeSegment'),
                    ('IfcPipeFitting', 'IfcPipeFitting'),
                    ('IfcFlowController', 'IfcFlowController'),
                    # Broader parent types as fallback
                    ('IfcFlowTerminal', 'IfcSanitaryTerminal'),
                    ('IfcFlowSegment', 'IfcPipeSegment'),
                    ('IfcFlowFitting', 'IfcPipeFitting'),
                    ('IfcFlowStorageDevice', 'IfcTank'),
                    ('IfcFlowMovingDevice', 'IfcFlowController'),
                ]
                for check_type, mapped_type in parent_checks:
                    if node.is_a(check_type):
                        return mapped_type
                return ifc_type
            return 'default'
        except Exception:
            return 'default'

    def _get_node_label(self, node) -> str:
        """Generate a short label for the node."""
        try:
            node_type = self._get_node_type(node)
            if node_type == 'IfcSanitaryTerminal':
                # Show the predefined type (SHOWER, SINK, etc.)
                if hasattr(node, 'PredefinedType') and node.PredefinedType:
                    ptype = node.PredefinedType
                    return ptype.replace('_', ' ').capitalize()
                return 'Terminal'
            elif node_type == 'IfcTank':
                return 'Tank'
            elif node_type == 'IfcValve':
                if hasattr(node, 'PredefinedType') and node.PredefinedType:
                    return node.PredefinedType.replace('_', ' ').capitalize()
                return 'Valve'
            elif node_type == 'IfcPipeFitting':
                return 'Fitting'
            elif node_type == 'IfcPipeSegment':
                return 'Pipe'
            else:
                return node_type.replace('Ifc', '')
        except Exception:
            return ''

    def _get_node_color(self, node_type: str) -> str:
        """Get color for a node based on its type."""
        return COMPONENT_COLORS.get(node_type, COMPONENT_COLORS['default'])

    def plot(self,
             figsize: Tuple[int, int] = (12, 8),
             layout: str = 'spring',
             color_by_type: bool = True,
             show_labels: bool = True,
             node_size: int = 500,
             font_size: int = 8,
             title: str = 'Hydraulic System Topology',
             highlight_path: Optional[List[Any]] = None,
             save_path: Optional[str] = None,
             show: bool = True) -> plt.Figure:
        """
        Plot the hydraulic system graph.

        Args:
            figsize: Figure size as (width, height) in inches
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular',
                    'shell', 'spectral', 'hierarchical')
            color_by_type: Color nodes by their IFC component type
            show_labels: Display node labels
            node_size: Size of nodes in the plot
            font_size: Font size for labels
            title: Plot title
            highlight_path: Optional path (list of IFC objects) to highlight
            save_path: Optional file path to save the figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate layout
        pos = self._get_layout(layout)

        # Determine node colors
        if color_by_type:
            node_colors = [self._get_node_color(self.node_types.get(n, 'default'))
                          for n in self.graph.nodes()]
        else:
            node_colors = '#2196F3'

        # Draw edges
        edge_colors = ['#CCCCCC'] * len(self.graph.edges())
        edge_widths = [1.0] * len(self.graph.edges())

        # Highlight path if provided
        if highlight_path is not None:
            highlight_edges = self._get_path_edges(highlight_path)
            for i, edge in enumerate(self.graph.edges()):
                if edge in highlight_edges or (edge[1], edge[0]) in highlight_edges:
                    edge_colors[i] = '#E91E63'  # Pink highlight
                    edge_widths[i] = 3.0

        nx.draw_networkx_edges(self.graph, pos, ax=ax,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.7)

        # Draw nodes by type with distinct shapes
        if color_by_type:
            for comp_type, marker in COMPONENT_SHAPES.items():
                type_nodes = [n for n in self.graph.nodes()
                              if self.node_types.get(n, 'default') == comp_type]
                if not type_nodes:
                    continue
                color = COMPONENT_COLORS.get(comp_type,
                                             COMPONENT_COLORS['default'])
                size = node_size * 1.5 if comp_type in (
                    'IfcSanitaryTerminal', 'IfcTank') else node_size
                nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                       nodelist=type_nodes,
                                       node_color=color,
                                       node_shape=marker,
                                       node_size=size,
                                       alpha=0.9)
            other_nodes = [n for n in self.graph.nodes()
                           if self.node_types.get(n, 'default')
                           not in COMPONENT_SHAPES]
            if other_nodes:
                nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                       nodelist=other_nodes,
                                       node_color=COMPONENT_COLORS['default'],
                                       node_size=node_size,
                                       alpha=0.9)
        else:
            nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                   node_color=node_colors,
                                   node_size=node_size,
                                   alpha=0.9)

        # Draw labels only for key nodes (terminals and tanks)
        if show_labels:
            key_types = {'IfcSanitaryTerminal', 'IfcTank'}
            key_labels = {n: self.node_labels[n] for n in self.graph.nodes()
                          if self.node_types.get(n) in key_types}
            nx.draw_networkx_labels(self.graph, pos, key_labels, ax=ax,
                                   font_size=font_size,
                                   font_color='black',
                                   font_weight='bold')

        # Add legend for component types
        if color_by_type:
            self._add_legend(ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_with_data(self,
                       node_values: Dict[str, float],
                       colormap: str = 'RdYlGn',
                       figsize: Tuple[int, int] = (12, 8),
                       layout: str = 'spring',
                       show_labels: bool = True,
                       node_size: int = 500,
                       font_size: int = 8,
                       title: str = 'Hydraulic System - Data View',
                       colorbar_label: str = 'Value',
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
        """
        Plot graph with nodes colored by data values (e.g., flow rate, pressure).

        Args:
            node_values: Dictionary mapping node IDs to values
            colormap: Matplotlib colormap name
            figsize: Figure size as (width, height)
            layout: Layout algorithm
            show_labels: Display node labels
            node_size: Size of nodes
            font_size: Font size for labels
            title: Plot title
            colorbar_label: Label for the colorbar
            save_path: Optional file path to save the figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        pos = self._get_layout(layout)

        # Map values to nodes
        values = []
        for node in self.graph.nodes():
            values.append(node_values.get(node, 0))

        # Normalize values for colormap
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            vmin, vmax = 0, 1

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, ax=ax,
                              edge_color='#CCCCCC',
                              width=1.0,
                              alpha=0.7)

        # Draw nodes with colormap
        nodes = nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                       node_color=values,
                                       cmap=plt.cm.get_cmap(colormap),
                                       vmin=vmin, vmax=vmax,
                                       node_size=node_size,
                                       alpha=0.9)

        # Add colorbar
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=10)

        # Draw labels
        if show_labels:
            nx.draw_networkx_labels(self.graph, pos, self.node_labels, ax=ax,
                                   font_size=font_size,
                                   font_color='black')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_paths(self,
                   paths: List[List[Any]],
                   figsize: Tuple[int, int] = (14, 10),
                   layout: str = 'hierarchical',
                   show_labels: bool = True,
                   node_size: int = 400,
                   font_size: int = 7,
                   title: str = 'Hydraulic System Topology',
                   save_path: Optional[str] = None,
                   show: bool = True) -> plt.Figure:
        """
        Plot graph with each path highlighted in a different color.

        Args:
            paths: List of paths (each path is a list of IFC objects)
            figsize: Figure size
            layout: Layout algorithm
            show_labels: Display node labels
            node_size: Size of nodes
            font_size: Font size for labels
            title: Plot title
            save_path: Optional file path to save the figure
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        pos = self._get_layout(layout)

        # Generate colors for each path
        path_colors = plt.cm.tab10(range(len(paths)))

        # Draw base edges in grey
        nx.draw_networkx_edges(self.graph, pos, ax=ax,
                              edge_color='#EEEEEE',
                              width=1.0,
                              alpha=0.5)

        # Draw each path with its own color
        legend_handles = []
        for idx, path in enumerate(paths):
            if path is None:
                continue
            path_edges = self._get_path_edges(path)
            color = path_colors[idx % len(path_colors)]

            # Get edges that exist in the graph
            edges_to_draw = []
            for edge in self.graph.edges():
                if edge in path_edges or (edge[1], edge[0]) in path_edges:
                    edges_to_draw.append(edge)

            if edges_to_draw:
                nx.draw_networkx_edges(self.graph, pos, ax=ax,
                                      edgelist=edges_to_draw,
                                      edge_color=[color],
                                      width=2.5,
                                      alpha=0.8)

            # Add to legend
            if path:
                start_label = self._get_node_label(path[0])
                legend_handles.append(mpatches.Patch(color=color,
                                                     label=f'Path {idx+1}: {start_label}'))

        # Draw nodes by type with distinct shapes and sizes
        for comp_type, marker in COMPONENT_SHAPES.items():
            type_nodes = [n for n in self.graph.nodes()
                          if self.node_types.get(n, 'default') == comp_type]
            if not type_nodes:
                continue
            color = COMPONENT_COLORS.get(comp_type, COMPONENT_COLORS['default'])
            # Terminals and tanks get larger nodes
            size = node_size * 1.8 if comp_type in (
                'IfcSanitaryTerminal', 'IfcTank') else node_size
            nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                   nodelist=type_nodes,
                                   node_color=color,
                                   node_shape=marker,
                                   node_size=size,
                                   alpha=0.9)

        # Draw remaining nodes not matching any known type
        other_nodes = [n for n in self.graph.nodes()
                       if self.node_types.get(n, 'default')
                       not in COMPONENT_SHAPES]
        if other_nodes:
            nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                   nodelist=other_nodes,
                                   node_color=COMPONENT_COLORS['default'],
                                   node_size=node_size,
                                   alpha=0.9)

        # Draw labels only for key nodes (terminals and tanks)
        if show_labels:
            key_types = {'IfcSanitaryTerminal', 'IfcTank'}
            key_labels = {n: self.node_labels[n] for n in self.graph.nodes()
                          if self.node_types.get(n) in key_types}
            nx.draw_networkx_labels(self.graph, pos, key_labels, ax=ax,
                                   font_size=font_size + 1,
                                   font_color='black',
                                   font_weight='bold')

        # Add path legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left',
                     fontsize=8, title='Paths')

        # Add component type legend
        self._add_legend(ax, loc='upper right')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def _get_layout(self, layout: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm."""
        if layout == 'hierarchical':
            return self._hierarchical_layout()
        elif layout == 'spring':
            return nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        else:
            return self._hierarchical_layout()

    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Create a hierarchical tree layout suitable for flow networks.

        Uses the tank as the root and builds a proper tree layout where
        each subtree gets proportional horizontal space, avoiding crossings.
        Terminals appear at leaf positions, tank at the root.
        Falls back to using the node with highest degree if no tank is found.
        """
        tanks = [n for n, t in self.node_types.items()
                 if t == 'IfcTank']

        if tanks:
            root = tanks[0]
        elif self.graph.number_of_nodes() > 0:
            # Use the node with the highest degree as root (likely a
            # central junction in the network)
            root = max(self.graph.nodes(), key=lambda n: self.graph.degree(n))
        else:
            return {}

        # Build a BFS tree from the tank (root)
        bfs_tree = nx.bfs_tree(self.graph, root)

        # Calculate the number of leaves in each subtree
        leaf_counts = {}

        def _count_leaves(node):
            children = list(bfs_tree.successors(node))
            if not children:
                leaf_counts[node] = 1
                return 1
            total = sum(_count_leaves(c) for c in children)
            leaf_counts[node] = total
            return total

        _count_leaves(root)

        # Get the depth of each node
        depths = nx.single_source_shortest_path_length(bfs_tree, root)
        max_depth = max(depths.values()) if depths else 0

        # Assign positions: each subtree gets horizontal space proportional
        # to its leaf count, ensuring no edge crossings
        pos = {}

        def _assign_positions(node, x_min, x_max, depth):
            x = (x_min + x_max) / 2.0
            # Tank at top (y=1), terminals at bottom (y=0)
            y = 1.0 - (depth / max(max_depth, 1))
            pos[node] = (x, y)

            children = list(bfs_tree.successors(node))
            if not children:
                return

            total_child_leaves = sum(leaf_counts[c] for c in children)
            current_x = x_min
            for child in children:
                child_width = ((x_max - x_min) *
                               leaf_counts[child] / total_child_leaves)
                _assign_positions(child, current_x,
                                  current_x + child_width, depth + 1)
                current_x += child_width

        _assign_positions(root, 0.0, 1.0, 0)

        # Include any nodes not reached by BFS (disconnected components)
        for node in self.graph.nodes():
            if node not in pos:
                pos[node] = (0.5, 0.5)

        return pos

    def _get_path_edges(self, path: List[Any]) -> set:
        """Convert a path (list of IFC objects) to a set of edge tuples."""
        edges = set()
        if path is None:
            return edges
        for i in range(len(path) - 1):
            id1 = self._get_node_id(path[i])
            id2 = self._get_node_id(path[i + 1])
            edges.add((id1, id2))
        return edges

    def _add_legend(self, ax: plt.Axes, loc: str = 'lower right'):
        """Add component type legend to the plot."""
        # Get unique types in the current graph
        unique_types = set(self.node_types.values())

        handles = []
        for comp_type in sorted(unique_types):
            if comp_type in COMPONENT_COLORS:
                color = COMPONENT_COLORS[comp_type]
                label = comp_type.replace('Ifc', '')
                handles.append(mpatches.Patch(color=color, label=label))

        if handles:
            ax.legend(handles=handles, loc=loc, fontsize=8,
                     title='Components', framealpha=0.9)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': {},
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            'num_components': nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0,
        }

        # Count nodes by type
        for node_type in self.node_types.values():
            if node_type not in stats['node_types']:
                stats['node_types'][node_type] = 0
            stats['node_types'][node_type] += 1

        return stats

    def print_statistics(self):
        """Print graph statistics to console."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("HYDRAULIC SYSTEM GRAPH STATISTICS")
        print("="*50)
        print(f"Total Nodes: {stats['num_nodes']}")
        print(f"Total Edges: {stats['num_edges']}")
        print(f"Connected: {stats['is_connected']}")
        print(f"Components: {stats['num_components']}")
        print("\nNodes by Type:")
        for comp_type, count in sorted(stats['node_types'].items()):
            print(f"  {comp_type}: {count}")
        print("="*50 + "\n")
