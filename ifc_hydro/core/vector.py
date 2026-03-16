"""
Vector operations module for 3D geometric calculations.

This module provides methods for vector operations commonly used in
hydraulic system analysis, including direction vectors, magnitudes,
normalization, and angle calculations.
"""


class Vector:
    """
    Provides 3D vector operations for geometric calculations.

    This class offers static methods for common vector operations including
    direction vector creation, magnitude calculation, normalization,
    dot product, and angle calculations.
    """

    @staticmethod
    def create_direction_vector(from_point: tuple, to_point: tuple) -> tuple:
        """
        Create a direction vector from one point to another.

        Args:
            from_point (tuple): Starting point as (x, y, z)
            to_point (tuple): Ending point as (x, y, z)

        Returns:
            tuple: Direction vector as (dx, dy, dz)
        """
        return (
            round(to_point[0] - from_point[0], 3),
            round(to_point[1] - from_point[1], 3),
            round(to_point[2] - from_point[2], 3)
        )

    @staticmethod
    def magnitude(vector: tuple) -> float:
        """
        Calculate the magnitude (length) of a vector.

        Args:
            vector (tuple): 3D vector as (x, y, z)

        Returns:
            float: Magnitude of the vector
        """
        return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

    @staticmethod
    def normalize(vector: tuple) -> tuple:
        """
        Normalize a vector to unit length.

        Args:
            vector (tuple): 3D vector as (x, y, z)

        Returns:
            tuple: Unit vector in the same direction
        """
        magnitude = Vector.magnitude(vector)
        if magnitude == 0:
            return (0.0, 0.0, 0.0)

        # Round each component to nearest integer (-1, 0, or 1)
        x = round(vector[0]/magnitude, 0)
        y = round(vector[1]/magnitude, 0)
        z = round(vector[2]/magnitude, 0)

        # Convert -0.0 to 0.0 by adding 0.0
        # This preserves -1.0 and 1.0 as they represent actual direction
        return (x + 0.0, y + 0.0, z + 0.0)

    @staticmethod
    def dot_product(vector1: tuple, vector2: tuple) -> float:
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (tuple): First 3D vector as (x, y, z)
            vector2 (tuple): Second 3D vector as (x, y, z)

        Returns:
            float: Dot product of the two vectors
        """
        return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]

    @staticmethod
    def angle_between(vector1: tuple, vector2: tuple) -> float:
        """
        Calculate the angle between two vectors in degrees.

        Args:
            vector1 (tuple): First 3D vector as (x, y, z)
            vector2 (tuple): Second 3D vector as (x, y, z)

        Returns:
            float: Angle between vectors in degrees (0-180)
        """
        import math

        # Normalize vectors (snaps to cardinal directions)
        unit1 = Vector.normalize(vector1)
        unit2 = Vector.normalize(vector2)

        # Calculate actual magnitudes of snapped vectors
        # (may be > 1 when normalize rounds two components to 1)
        mag1 = Vector.magnitude(unit1)
        mag2 = Vector.magnitude(unit2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Calculate dot product and divide by magnitudes
        dot = Vector.dot_product(unit1, unit2)
        cos_angle = dot / (mag1 * mag2)

        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))

        # Calculate angle in radians then convert to degrees
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return round(angle_deg, 2)
