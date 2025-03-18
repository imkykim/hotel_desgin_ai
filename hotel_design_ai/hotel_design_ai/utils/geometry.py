"""
Geometric utility functions for hotel layout design.
"""

from typing import Tuple, List, Dict, Any, Optional, Union, Sequence
import numpy as np


# Type aliases
Point3D = Tuple[float, float, float]
Vector3D = Tuple[float, float, float]
BoundingBox = Tuple[Point3D, Point3D]  # (min_point, max_point)


class GeometricOps:
    """Class for common geometric operations."""

    @staticmethod
    def distance(p1: Point3D, p2: Point3D) -> float:
        """
        Calculate Euclidean distance between two 3D points.

        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)

        Returns:
            float: Euclidean distance
        """
        return np.sqrt(sum((p2[i] - p1[i]) ** 2 for i in range(3)))

    @staticmethod
    def manhattan_distance(p1: Point3D, p2: Point3D) -> float:
        """
        Calculate Manhattan distance between two 3D points.

        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)

        Returns:
            float: Manhattan distance
        """
        return sum(abs(p2[i] - p1[i]) for i in range(3))

    @staticmethod
    def midpoint(p1: Point3D, p2: Point3D) -> Point3D:
        """
        Calculate the midpoint between two 3D points.

        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)

        Returns:
            Point3D: Midpoint
        """
        return tuple((p1[i] + p2[i]) / 2 for i in range(3))

    @staticmethod
    def centroid(points: List[Point3D]) -> Point3D:
        """
        Calculate the centroid of a set of 3D points.

        Args:
            points: List of points (x, y, z)

        Returns:
            Point3D: Centroid
        """
        if not points:
            raise ValueError("Cannot calculate centroid of empty list")

        n = len(points)
        return tuple(sum(p[i] for p in points) / n for i in range(3))


class VectorOps:
    """Class for vector operations."""

    @staticmethod
    def dot_product(v1: Vector3D, v2: Vector3D) -> float:
        """
        Calculate dot product of two 3D vectors.

        Args:
            v1: First vector (x, y, z)
            v2: Second vector (x, y, z)

        Returns:
            float: Dot product
        """
        return sum(v1[i] * v2[i] for i in range(3))

    @staticmethod
    def cross_product(v1: Vector3D, v2: Vector3D) -> Vector3D:
        """
        Calculate cross product of two 3D vectors.

        Args:
            v1: First vector (x, y, z)
            v2: Second vector (x, y, z)

        Returns:
            Vector3D: Cross product
        """
        return (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )

    @staticmethod
    def magnitude(v: Vector3D) -> float:
        """
        Calculate magnitude (length) of a 3D vector.

        Args:
            v: Vector (x, y, z)

        Returns:
            float: Magnitude
        """
        return np.sqrt(sum(v[i] ** 2 for i in range(3)))

    @staticmethod
    def normalize(v: Vector3D) -> Vector3D:
        """
        Normalize a 3D vector to unit length.

        Args:
            v: Vector (x, y, z)

        Returns:
            Vector3D: Normalized vector
        """
        mag = VectorOps.magnitude(v)
        if mag == 0:
            return (0, 0, 0)
        return tuple(v[i] / mag for i in range(3))

    @staticmethod
    def angle_between(v1: Vector3D, v2: Vector3D) -> float:
        """
        Calculate angle between two 3D vectors in radians.

        Args:
            v1: First vector (x, y, z)
            v2: Second vector (x, y, z)

        Returns:
            float: Angle in radians
        """
        dot = VectorOps.dot_product(v1, v2)
        mag1 = VectorOps.magnitude(v1)
        mag2 = VectorOps.magnitude(v2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Ensure dot product is in valid range for arccos
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return np.arccos(cos_angle)


class BoxOps:
    """Class for bounding box operations."""

    @staticmethod
    def are_adjacent(
        box1: BoundingBox, box2: BoundingBox, tolerance: float = 0.001
    ) -> bool:
        """
        Check if two bounding boxes are adjacent (share a face).

        Args:
            box1: First box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
            box2: Second box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
            tolerance: Distance tolerance for adjacency

        Returns:
            bool: True if boxes are adjacent
        """
        (min1_x, min1_y, min1_z), (max1_x, max1_y, max1_z) = box1
        (min2_x, min2_y, min2_z), (max2_x, max2_y, max2_z) = box2

        # Check if boxes overlap
        if (
            max1_x < min2_x
            or min1_x > max2_x
            or max1_y < min2_y
            or min1_y > max2_y
            or max1_z < min2_z
            or min1_z > max2_z
        ):
            # No overlap, check for adjacency

            # X-axis adjacency
            x_adjacent = (
                abs(max1_x - min2_x) < tolerance or abs(max2_x - min1_x) < tolerance
            )
            y_overlap = min1_y <= max2_y and max1_y >= min2_y
            z_overlap = min1_z <= max2_z and max1_z >= min2_z

            if x_adjacent and y_overlap and z_overlap:
                return True

            # Y-axis adjacency
            y_adjacent = (
                abs(max1_y - min2_y) < tolerance or abs(max2_y - min1_y) < tolerance
            )
            x_overlap = min1_x <= max2_x and max1_x >= min2_x

            if y_adjacent and x_overlap and z_overlap:
                return True

            # Z-axis adjacency
            z_adjacent = (
                abs(max1_z - min2_z) < tolerance or abs(max2_z - min1_z) < tolerance
            )

            if z_adjacent and x_overlap and y_overlap:
                return True

        return False

    @staticmethod
    def box_to_corners(box: BoundingBox) -> List[Point3D]:
        """
        Convert a bounding box to its 8 corner points.

        Args:
            box: Box as ((min_x, min_y, min_z), (max_x, max_y, max_z))

        Returns:
            List[Point3D]: List of 8 corner points
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = box

        return [
            (min_x, min_y, min_z),  # 0: bottom, back, left
            (max_x, min_y, min_z),  # 1: bottom, back, right
            (max_x, max_y, min_z),  # 2: bottom, front, right
            (min_x, max_y, min_z),  # 3: bottom, front, left
            (min_x, min_y, max_z),  # 4: top, back, left
            (max_x, min_y, max_z),  # 5: top, back, right
            (max_x, max_y, max_z),  # 6: top, front, right
            (min_x, max_y, max_z),  # 7: top, front, left
        ]

    @staticmethod
    def box_volume(box: BoundingBox) -> float:
        """
        Calculate the volume of a bounding box.

        Args:
            box: Box as ((min_x, min_y, min_z), (max_x, max_y, max_z))

        Returns:
            float: Volume
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = box

        width = max_x - min_x
        length = max_y - min_y
        height = max_z - min_z

        return width * length * height

    @staticmethod
    def box_surface_area(box: BoundingBox) -> float:
        """
        Calculate the surface area of a bounding box.

        Args:
            box: Box as ((min_x, min_y, min_z), (max_x, max_y, max_z))

        Returns:
            float: Surface area
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = box

        width = max_x - min_x
        length = max_y - min_y
        height = max_z - min_z

        return 2 * (width * length + width * height + length * height)

    @staticmethod
    def point_in_box(point: Point3D, box: BoundingBox) -> bool:
        """
        Check if a point is inside a bounding box.

        Args:
            point: Point (x, y, z)
            box: Box as ((min_x, min_y, min_z), (max_x, max_y, max_z))

        Returns:
            bool: True if point is inside box
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = box
        x, y, z = point

        return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z

    @staticmethod
    def box_intersection(box1: BoundingBox, box2: BoundingBox) -> Optional[BoundingBox]:
        """
        Calculate the intersection of two bounding boxes.

        Args:
            box1: First box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
            box2: Second box as ((min_x, min_y, min_z), (max_x, max_y, max_z))

        Returns:
            Optional[BoundingBox]: Intersection box or None if no intersection
        """
        (min1_x, min1_y, min1_z), (max1_x, max1_y, max1_z) = box1
        (min2_x, min2_y, min2_z), (max2_x, max2_y, max2_z) = box2

        # Calculate intersection
        int_min_x = max(min1_x, min2_x)
        int_min_y = max(min1_y, min2_y)
        int_min_z = max(min1_z, min2_z)

        int_max_x = min(max1_x, max2_x)
        int_max_y = min(max1_y, max2_y)
        int_max_z = min(max1_z, max2_z)

        # Check if intersection exists
        if int_min_x > int_max_x or int_min_y > int_max_y or int_min_z > int_max_z:
            return None  # No intersection

        return ((int_min_x, int_min_y, int_min_z), (int_max_x, int_max_y, int_max_z))

    @staticmethod
    def box_union(box1: BoundingBox, box2: BoundingBox) -> BoundingBox:
        """
        Calculate the union (enclosing box) of two bounding boxes.

        Args:
            box1: First box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
            box2: Second box as ((min_x, min_y, min_z), (max_x, max_y, max_z))

        Returns:
            BoundingBox: Union box
        """
        (min1_x, min1_y, min1_z), (max1_x, max1_y, max1_z) = box1
        (min2_x, min2_y, min2_z), (max2_x, max2_y, max2_z) = box2

        # Calculate union
        union_min_x = min(min1_x, min2_x)
        union_min_y = min(min1_y, min2_y)
        union_min_z = min(min1_z, min2_z)

        union_max_x = max(max1_x, max2_x)
        union_max_y = max(max1_y, max2_y)
        union_max_z = max(max1_z, max2_z)

        return (
            (union_min_x, union_min_y, union_min_z),
            (union_max_x, union_max_y, union_max_z),
        )


class GridOps:
    """Class for grid-related operations."""

    @staticmethod
    def snap_to_grid(point: Point3D, grid_size: float) -> Point3D:
        """
        Snap a point to the nearest grid intersection.

        Args:
            point: Point (x, y, z)
            grid_size: Grid cell size

        Returns:
            Point3D: Snapped point
        """
        return tuple(round(p / grid_size) * grid_size for p in point)

    @staticmethod
    def snap_box_to_grid(box: BoundingBox, grid_size: float) -> BoundingBox:
        """
        Snap a bounding box to the grid.

        Args:
            box: Box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
            grid_size: Grid cell size

        Returns:
            BoundingBox: Snapped box
        """
        min_point, max_point = box

        snapped_min = GridOps.snap_to_grid(min_point, grid_size)
        snapped_max = GridOps.snap_to_grid(max_point, grid_size)

        return (snapped_min, snapped_max)


class PathFinding:
    """Class for path finding operations."""

    @staticmethod
    def calculate_corridor_path(
        start: Point3D,
        end: Point3D,
        obstacles: List[BoundingBox],
        corridor_width: float = 1.5,
        clearance: float = 0.5,
        grid_size: float = 0.5,
    ) -> List[Point3D]:
        """
        Calculate a corridor path between two points, avoiding obstacles.
        This uses a simple Manhattan-style route (at most 2 turns).

        Args:
            start: Start point (x, y, z)
            end: End point (x, y, z)
            obstacles: List of obstacle bounding boxes
            corridor_width: Width of the corridor
            clearance: Additional clearance from obstacles
            grid_size: Grid size for snapping

        Returns:
            List[Point3D]: Path points defining the corridor centerline
        """
        # Start and end should be on same floor for now
        if abs(start[2] - end[2]) > 0.1:
            raise ValueError("Start and end points must be on the same floor")

        z = start[2]  # Use same z for entire path

        # Try direct path first
        direct_path = [start, end]
        if not PathFinding._path_intersects_obstacles(
            direct_path, obstacles, corridor_width / 2 + clearance
        ):
            return direct_path

        # Try x-then-y path
        mid_point1 = (end[0], start[1], z)
        xy_path = [start, mid_point1, end]
        if not PathFinding._path_intersects_obstacles(
            xy_path, obstacles, corridor_width / 2 + clearance
        ):
            return xy_path

        # Try y-then-x path
        mid_point2 = (start[0], end[1], z)
        yx_path = [start, mid_point2, end]
        if not PathFinding._path_intersects_obstacles(
            yx_path, obstacles, corridor_width / 2 + clearance
        ):
            return yx_path

        # For now, return the x-then-y path as fallback
        return xy_path

    @staticmethod
    def _path_intersects_obstacles(
        path: List[Point3D], obstacles: List[BoundingBox], clearance: float
    ) -> bool:
        """
        Check if a path intersects with any obstacles, considering clearance.

        Args:
            path: List of path points
            obstacles: List of obstacle bounding boxes
            clearance: Required clearance

        Returns:
            bool: True if path intersects any obstacle
        """
        # For each path segment
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            # For each obstacle
            for obstacle in obstacles:
                # Check for intersection
                if PathFinding._segment_intersects_box(p1, p2, obstacle, clearance):
                    return True

        return False

    @staticmethod
    def _segment_intersects_box(
        p1: Point3D, p2: Point3D, box: BoundingBox, clearance: float
    ) -> bool:
        """
        Check if a line segment intersects with a box (with clearance).

        Args:
            p1, p2: Segment endpoints
            box: Obstacle bounding box
            clearance: Required clearance

        Returns:
            bool: True if segment intersects box
        """
        # Expand box by clearance
        (min_x, min_y, min_z), (max_x, max_y, max_z) = box
        expanded_box = (
            (min_x - clearance, min_y - clearance, min_z - clearance),
            (max_x + clearance, max_y + clearance, max_z + clearance),
        )

        # Specialized check for horizontal/vertical segments (which our Manhattan paths will create)

        # If this is a vertical segment (same x)
        if abs(p1[0] - p2[0]) < 0.001:
            x = p1[0]
            min_y = min(p1[1], p2[1])
            max_y = max(p1[1], p2[1])
            z = p1[2]  # Assume same z

            # Check if x is within expanded box x-range
            if expanded_box[0][0] <= x <= expanded_box[1][0]:
                # Check if y-range overlaps
                if min_y <= expanded_box[1][1] and max_y >= expanded_box[0][1]:
                    # Check if z is within expanded box z-range
                    if expanded_box[0][2] <= z <= expanded_box[1][2]:
                        return True

        # If this is a horizontal segment (same y)
        elif abs(p1[1] - p2[1]) < 0.001:
            min_x = min(p1[0], p2[0])
            max_x = max(p1[0], p2[0])
            y = p1[1]
            z = p1[2]  # Assume same z

            # Check if y is within expanded box y-range
            if expanded_box[0][1] <= y <= expanded_box[1][1]:
                # Check if x-range overlaps
                if min_x <= expanded_box[1][0] and max_x >= expanded_box[0][0]:
                    # Check if z is within expanded box z-range
                    if expanded_box[0][2] <= z <= expanded_box[1][2]:
                        return True

        return False


# Utility functions for direct use (backward compatibility)
distance = GeometricOps.distance
manhattan_distance = GeometricOps.manhattan_distance
midpoint = GeometricOps.midpoint
centroid = GeometricOps.centroid
dot_product = VectorOps.dot_product
cross_product = VectorOps.cross_product
magnitude = VectorOps.magnitude
normalize = VectorOps.normalize
angle_between = VectorOps.angle_between
are_adjacent = BoxOps.are_adjacent
box_to_corners = BoxOps.box_to_corners
box_volume = BoxOps.box_volume
box_surface_area = BoxOps.box_surface_area
point_in_box = BoxOps.point_in_box
box_intersection = BoxOps.box_intersection
box_union = BoxOps.box_union
snap_to_grid = GridOps.snap_to_grid
snap_box_to_grid = GridOps.snap_box_to_grid
calculate_corridor_path = PathFinding.calculate_corridor_path


# Room-specific functions that work directly with room parameters
def room_centroid(
    room_position: Tuple[float, float, float],
    room_dimensions: Tuple[float, float, float],
) -> Point3D:
    """
    Calculate the centroid (center point) of a room.

    Args:
        room_position: (x, y, z) of the room's origin (lower corner)
        room_dimensions: (width, length, height) of the room

    Returns:
        Point3D: Center point of the room
    """
    x, y, z = room_position
    width, length, height = room_dimensions

    return (x + width / 2, y + length / 2, z + height / 2)


def create_room_box(
    room_position: Tuple[float, float, float],
    room_dimensions: Tuple[float, float, float],
) -> BoundingBox:
    """
    Create a bounding box for a room.

    Args:
        room_position: (x, y, z) of the room's origin (lower corner)
        room_dimensions: (width, length, height) of the room

    Returns:
        BoundingBox: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    """
    x, y, z = room_position
    width, length, height = room_dimensions

    return ((x, y, z), (x + width, y + length, z + height))


def room_corners(
    room_position: Tuple[float, float, float],
    room_dimensions: Tuple[float, float, float],
) -> List[Point3D]:
    """
    Calculate all corner points of a room.

    Args:
        room_position: (x, y, z) of the room's origin (lower corner)
        room_dimensions: (width, length, height) of the room

    Returns:
        List[Point3D]: List of 8 corner points
    """
    box = create_room_box(room_position, room_dimensions)
    return box_to_corners(box)


def can_fit_room(
    position: Tuple[float, float, float],
    dimensions: Tuple[float, float, float],
    existing_rooms: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    building_dimensions: Tuple[float, float, float],
    spacing: float = 0.0,
) -> bool:
    """
    Check if a room can fit at a position without colliding with existing rooms.

    Args:
        position: (x, y, z) of the proposed room position
        dimensions: (width, length, height) of the proposed room
        existing_rooms: List of (position, dimensions) tuples for existing rooms
        building_dimensions: (width, length, height) of the building envelope
        spacing: Minimum spacing between rooms

    Returns:
        bool: True if the room can fit, False otherwise
    """
    # Check if room is within building bounds
    x, y, z = position
    width, length, height = dimensions
    building_width, building_length, building_height = building_dimensions

    if (
        x < 0
        or y < 0
        or z < 0
        or x + width > building_width
        or y + length > building_length
        or z + height > building_height
    ):
        return False

    # Create bounding box for the new room
    new_room_box = create_room_box(position, dimensions)

    # Expand box by spacing
    if spacing > 0:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = new_room_box
        expanded_box = (
            (min_x - spacing, min_y - spacing, min_z - spacing),
            (max_x + spacing, max_y + spacing, max_z + spacing),
        )
    else:
        expanded_box = new_room_box

    # Check for collisions with existing rooms
    for room_position, room_dimensions in existing_rooms:
        existing_box = create_room_box(room_position, room_dimensions)

        # Check for intersection
        if box_intersection(expanded_box, existing_box) is not None:
            return False

    return True
