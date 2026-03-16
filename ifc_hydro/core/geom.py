"""
Geometry operations module for IFC element geometric calculations.

This module provides methods for geometric operations on IFC elements using
ifcopenshell, including elevation extraction, shape creation, and bounding
box calculations.
"""

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape


class Geom:
    """
    Provides geometric operations for IFC elements.

    This class offers static methods for common geometric operations including
    shape creation, elevation extraction, and bounding box center calculation.
    """

    @staticmethod
    def create_settings(use_world_coords: bool = True) -> ifcopenshell.geom.settings:
        """
        Create geometry settings for shape creation.

        Args:
            use_world_coords (bool): If True, use world coordinates instead of
                local coordinates. Defaults to True.

        Returns:
            ifcopenshell.geom.settings: Configured geometry settings object
        """
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, use_world_coords)
        return settings

    @staticmethod
    def create_shape(element, settings: ifcopenshell.geom.settings = None):
        """
        Create a geometry shape from an IFC element.

        Args:
            element: IFC element object (e.g., IfcProduct)
            settings (ifcopenshell.geom.settings, optional): Geometry settings.
                If None, creates default settings with USE_WORLD_COORDS=True.

        Returns:
            Shape object containing the element's geometry
        """
        if settings is None:
            settings = Geom.create_settings()
        return ifcopenshell.geom.create_shape(settings, element)

    @staticmethod
    def get_top_elevation(element, settings: ifcopenshell.geom.settings = None) -> float:
        """
        Get the top elevation (maximum Z coordinate) of an IFC element.

        Args:
            element: IFC element object (e.g., IfcProduct)
            settings (ifcopenshell.geom.settings, optional): Geometry settings.
                If None, creates default settings with USE_WORLD_COORDS=True.

        Returns:
            float: Top elevation rounded to 3 decimal places
        """
        if settings is None:
            settings = Geom.create_settings()
        shape = ifcopenshell.geom.create_shape(settings, element)
        return round(ifcopenshell.util.shape.get_top_elevation(shape.geometry), 3)

    @staticmethod
    def get_bottom_elevation(element, settings: ifcopenshell.geom.settings = None) -> float:
        """
        Get the bottom elevation (minimum Z coordinate) of an IFC element.

        Args:
            element: IFC element object (e.g., IfcProduct)
            settings (ifcopenshell.geom.settings, optional): Geometry settings.
                If None, creates default settings with USE_WORLD_COORDS=True.

        Returns:
            float: Bottom elevation rounded to 3 decimal places
        """
        if settings is None:
            settings = Geom.create_settings()
        shape = ifcopenshell.geom.create_shape(settings, element)
        return round(ifcopenshell.util.shape.get_bottom_elevation(shape.geometry), 3)

    @staticmethod
    def _get_bbox_from_geometry(geometry) -> tuple:
        """
        Extract bounding box from geometry, handling both standard shapes and Triangulation objects.

        Args:
            geometry: Geometry object from ifcopenshell shape (may be Triangulation or other type)

        Returns:
            tuple: Bounding box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        # Check if geometry is a Triangulation object by looking for verts attribute
        if hasattr(geometry, 'verts'):
            # Triangulation object - compute bbox from vertices
            verts = geometry.verts
            if not verts:
                raise ValueError("Triangulation has no vertices")

            # Vertices are stored as flat list: [x1, y1, z1, x2, y2, z2, ...]
            x_coords = verts[0::3]
            y_coords = verts[1::3]
            z_coords = verts[2::3]

            return (
                (min(x_coords), min(y_coords), min(z_coords)),
                (max(x_coords), max(y_coords), max(z_coords))
            )

        # Try standard get_bbox utility
        bbox = ifcopenshell.util.shape.get_bbox(geometry)

        # Validate the result is in expected format
        if not hasattr(bbox, '__iter__') or not hasattr(bbox, '__getitem__'):
            raise ValueError(f"Unexpected bbox type: {type(bbox).__name__}")

        return bbox

    @staticmethod
    def get_bbox_center(element, settings: ifcopenshell.geom.settings = None) -> tuple:
        """
        Calculate the bounding box center of an IFC element.

        The bounding box center is computed as the midpoint between the minimum
        and maximum coordinates in all three dimensions (X, Y, Z).

        Args:
            element: IFC element object (e.g., IfcProduct)
            settings (ifcopenshell.geom.settings, optional): Geometry settings.
                If None, creates default settings with USE_WORLD_COORDS=True.

        Returns:
            tuple: Center point as (x, y, z) coordinates rounded to 3 decimal places
        """
        if settings is None:
            settings = Geom.create_settings()
        shape = ifcopenshell.geom.create_shape(settings, element)

        # Get bounding box, handling Triangulation objects
        bbox = Geom._get_bbox_from_geometry(shape.geometry)

        # Calculate center from min and max corners
        center_x = round((bbox[0][0] + bbox[1][0]) / 2, 3)
        center_y = round((bbox[0][1] + bbox[1][1]) / 2, 3)
        center_z = round((bbox[0][2] + bbox[1][2]) / 2, 3)

        return (center_x, center_y, center_z)

    @staticmethod
    def get_bbox(element, settings: ifcopenshell.geom.settings = None) -> tuple:
        """
        Get the bounding box of an IFC element.

        Args:
            element: IFC element object (e.g., IfcProduct)
            settings (ifcopenshell.geom.settings, optional): Geometry settings.
                If None, creates default settings with USE_WORLD_COORDS=True.

        Returns:
            tuple: Bounding box as (min_x, min_y, min_z, max_x, max_y, max_z)
                rounded to 3 decimal places
        """
        if settings is None:
            settings = Geom.create_settings()
        shape = ifcopenshell.geom.create_shape(settings, element)

        # Get bounding box, handling Triangulation objects
        bbox = Geom._get_bbox_from_geometry(shape.geometry)

        return (
            round(bbox[0][0], 3),
            round(bbox[0][1], 3),
            round(bbox[0][2], 3),
            round(bbox[1][0], 3),
            round(bbox[1][1], 3),
            round(bbox[1][2], 3)
        )
