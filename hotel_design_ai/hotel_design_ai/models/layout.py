"""
Layout model for hotel design.
This extends SpatialGrid with additional layout-specific functionality.
"""

from typing import Dict, List, Tuple, Set, Any, Optional
import json
import numpy as np
import uuid
from collections import defaultdict

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room
from hotel_design_ai.utils.geometry import distance, create_room_box, manhattan_distance


class Layout:
    """
    A class representing a hotel layout, extending the SpatialGrid with
    layout-specific functionality and metadata.
    """

    def __init__(
        self,
        spatial_grid: SpatialGrid,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a layout with a spatial grid.

        Args:
            spatial_grid: The underlying spatial grid
            name: Optional name for the layout
            metadata: Optional metadata
        """
        self.spatial_grid = spatial_grid
        self.name = name or f"Layout_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}

        # Cache for frequently accessed data
        self._adjacency_cache = {}
        self._exterior_rooms_cache = None

    @property
    def width(self) -> float:
        """Get the width of the layout"""
        return self.spatial_grid.width

    @property
    def length(self) -> float:
        """Get the length of the layout"""
        return self.spatial_grid.length

    @property
    def height(self) -> float:
        """Get the height of the layout"""
        return self.spatial_grid.height

    @property
    def rooms(self) -> Dict[int, Dict[str, Any]]:
        """Get all rooms in the layout"""
        return self.spatial_grid.rooms

    def get_room(self, room_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a room by ID.

        Args:
            room_id: Room ID

        Returns:
            Optional[Dict]: Room data or None if not found
        """
        return self.rooms.get(room_id)

    def add_room(self, room: Room) -> bool:
        """
        Add a room to the layout.

        Args:
            room: Room object to add

        Returns:
            bool: True if room was added successfully
        """
        if room.position is None:
            return False

        # Clear caches
        self._clear_caches()

        x, y, z = room.position
        return self.spatial_grid.place_room(
            room_id=room.id,
            x=x,
            y=y,
            z=z,
            width=room.width,
            length=room.length,
            height=room.height,
            room_type=room.room_type,
            metadata=room.metadata,
        )

    def remove_room(self, room_id: int) -> bool:
        """
        Remove a room from the layout.

        Args:
            room_id: ID of the room to remove

        Returns:
            bool: True if room was removed successfully
        """
        # Clear caches
        self._clear_caches()

        return self.spatial_grid.remove_room(room_id)

    def _clear_caches(self):
        """Clear all cached data"""
        self._adjacency_cache = {}
        self._exterior_rooms_cache = None

    def move_room(self, room_id: int, new_position: Tuple[float, float, float]) -> bool:
        """
        Move a room to a new position.

        Args:
            room_id: ID of the room to move
            new_position: (x, y, z) new position

        Returns:
            bool: True if room was moved successfully
        """
        room_data = self.get_room(room_id)
        if not room_data:
            return False

        # Remove the room
        self.spatial_grid.remove_room(room_id)

        # Try to place it at the new position
        x, y, z = new_position
        success = self.spatial_grid.place_room(
            room_id=room_id,
            x=x,
            y=y,
            z=z,
            width=room_data["dimensions"][0],
            length=room_data["dimensions"][1],
            height=room_data["dimensions"][2],
            room_type=room_data["type"],
            metadata=room_data.get("metadata", {}),
        )

        # Clear caches
        if success:
            self._clear_caches()

        return success

    def resize_room(
        self, room_id: int, new_dimensions: Tuple[float, float, float]
    ) -> bool:
        """
        Resize a room.

        Args:
            room_id: ID of the room to resize
            new_dimensions: (width, length, height) new dimensions

        Returns:
            bool: True if room was resized successfully
        """
        room_data = self.get_room(room_id)
        if not room_data:
            return False

        # Remove the room
        position = room_data["position"]
        self.spatial_grid.remove_room(room_id)

        # Try to place it with new dimensions
        x, y, z = position
        width, length, height = new_dimensions
        success = self.spatial_grid.place_room(
            room_id=room_id,
            x=x,
            y=y,
            z=z,
            width=width,
            length=length,
            height=height,
            room_type=room_data["type"],
            metadata=room_data.get("metadata", {}),
        )

        # Clear caches
        if success:
            self._clear_caches()

        return success

    def get_adjacency_graph(self) -> Dict[int, List[int]]:
        """
        Get the adjacency graph of rooms.

        Returns:
            Dict[int, List[int]]: Mapping of room ID to list of adjacent room IDs
        """
        # Use cached graph if available
        if self._adjacency_cache:
            return self._adjacency_cache

        # Build adjacency graph
        graph = {}

        for room_id in self.rooms:
            neighbors = self.spatial_grid.get_room_neighbors(room_id)
            graph[room_id] = neighbors

        # Cache the result
        self._adjacency_cache = graph

        return graph

    def are_adjacent(self, room_id1: int, room_id2: int) -> bool:
        """
        Check if two rooms are adjacent.

        Args:
            room_id1: First room ID
            room_id2: Second room ID

        Returns:
            bool: True if rooms are adjacent
        """
        return self.spatial_grid.are_adjacent(room_id1, room_id2)

    def get_exterior_rooms(self) -> List[int]:
        """
        Get all rooms with exterior walls.

        Returns:
            List[int]: List of room IDs with exterior walls
        """
        # Use cached result if available
        if self._exterior_rooms_cache is not None:
            return self._exterior_rooms_cache

        # Get from spatial grid
        exterior_rooms = self.spatial_grid.get_exterior_rooms()

        # Cache the result
        self._exterior_rooms_cache = exterior_rooms

        return exterior_rooms

    def get_rooms_by_type(self, room_type: str) -> List[Dict[str, Any]]:
        """
        Get all rooms of a specific type.

        Args:
            room_type: Type of rooms to get

        Returns:
            List[Dict]: List of room data
        """
        return [
            room_data
            for room_id, room_data in self.rooms.items()
            if room_data["type"] == room_type
        ]

    def get_rooms_by_floor(
        self, floor: int, floor_height: float = 4.0
    ) -> List[Dict[str, Any]]:
        """
        Get all rooms on a specific floor.

        Args:
            floor: Floor number (0 = ground floor, -1 = basement)
            floor_height: Height of each floor in meters

        Returns:
            List[Dict]: List of room data
        """
        min_z = floor * floor_height
        max_z = (floor + 1) * floor_height

        return [
            room_data
            for room_id, room_data in self.rooms.items()
            if min_z <= room_data["position"][2] < max_z
        ]

    def get_rooms_by_department(self, department: str) -> List[Dict[str, Any]]:
        """
        Get all rooms in a specific department.

        Args:
            department: Department name

        Returns:
            List[Dict]: List of room data
        """
        return [
            room_data
            for room_id, room_data in self.rooms.items()
            if room_data.get("metadata", {}).get("department") == department
        ]

    def get_areas_by_type(self) -> Dict[str, float]:
        """
        Calculate total area for each room type.

        Returns:
            Dict[str, float]: Mapping of room type to total area
        """
        areas = defaultdict(float)

        for room_data in self.rooms.values():
            room_type = room_data["type"]
            width, length, _ = room_data["dimensions"]
            area = width * length
            areas[room_type] += area

        return dict(areas)

    def calculate_floor_areas(self, floor_height: float = 4.0) -> Dict[int, float]:
        """
        Calculate total area for each floor.

        Args:
            floor_height: Height of each floor in meters

        Returns:
            Dict[int, float]: Mapping of floor number to total area
        """
        floor_areas = defaultdict(float)

        for room_data in self.rooms.values():
            z = room_data["position"][2]
            floor = int(z / floor_height)

            width, length, _ = room_data["dimensions"]
            area = width * length
            floor_areas[floor] += area

        return dict(floor_areas)

    def find_path(self, start_room_id: int, end_room_id: int) -> Optional[List[int]]:
        """
        Find a path between two rooms using breadth-first search.

        Args:
            start_room_id: Starting room ID
            end_room_id: Ending room ID

        Returns:
            Optional[List[int]]: List of room IDs in the path or None if no path exists
        """
        if start_room_id not in self.rooms or end_room_id not in self.rooms:
            return None

        # Get adjacency graph
        graph = self.get_adjacency_graph()

        # Breadth-first search
        queue = [(start_room_id, [start_room_id])]
        visited = {start_room_id}

        while queue:
            (node, path) = queue.pop(0)

            # Check neighbors
            for neighbor in graph.get(node, []):
                if neighbor == end_room_id:
                    # Found the end room
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found
        return None

    def can_place_room(
        self,
        width: float,
        length: float,
        height: float,
        position: Tuple[float, float, float],
    ) -> bool:
        """
        Check if a room can be placed at the specified position.

        Args:
            width: Room width
            length: Room length
            height: Room height
            position: (x, y, z) position

        Returns:
            bool: True if the room can be placed
        """
        x, y, z = position

        # Convert to grid coordinates
        grid_x = int(x / self.spatial_grid.grid_size)
        grid_y = int(y / self.spatial_grid.grid_size)
        grid_z = int(z / self.spatial_grid.grid_size)
        grid_width = int(width / self.spatial_grid.grid_size)
        grid_length = int(length / self.spatial_grid.grid_size)
        grid_height = int(height / self.spatial_grid.grid_size)

        # Check bounds
        if (
            grid_x < 0
            or grid_y < 0
            or grid_z < 0
            or grid_x + grid_width > self.spatial_grid.width_cells
            or grid_y + grid_length > self.spatial_grid.length_cells
            or grid_z + grid_height > self.spatial_grid.height_cells
        ):
            return False

        # Check for collisions
        target_region = self.spatial_grid.grid[
            grid_x : grid_x + grid_width,
            grid_y : grid_y + grid_length,
            grid_z : grid_z + grid_height,
        ]

        return np.all(target_region == 0)

    def find_valid_position(
        self,
        width: float,
        length: float,
        height: float,
        floor: Optional[int] = None,
        near_room_id: Optional[int] = None,
        structural_grid: Optional[Tuple[float, float]] = None,
        floor_height: float = 4.0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a valid position for a room with the given dimensions.

        Args:
            width: Room width
            length: Room length
            height: Room height
            floor: Optional specific floor to place on
            near_room_id: Optional room ID to place near
            structural_grid: Optional (x_spacing, y_spacing) for grid alignment
            floor_height: Height of each floor in meters

        Returns:
            Optional[Tuple[float, float, float]]: Valid position or None if not found
        """
        # Calculate search space
        if floor is not None:
            # Search on specific floor
            z_min = floor * floor_height
            z_max = z_min + 0.1  # Only search at this z coordinate
        else:
            # Search entire building height
            z_min = 0
            z_max = self.height - height

        # If near_room_id is specified, search near that room first
        if near_room_id is not None and near_room_id in self.rooms:
            near_room = self.rooms[near_room_id]
            near_x, near_y, near_z = near_room["position"]
            near_w, near_l, near_h = near_room["dimensions"]

            # Check positions around the room
            positions = [
                (near_x + near_w, near_y, near_z),  # Right
                (near_x - width, near_y, near_z),  # Left
                (near_x, near_y + near_l, near_z),  # Behind
                (near_x, near_y - length, near_z),  # In front
            ]

            for pos in positions:
                if self.can_place_room(width, length, height, pos):
                    return pos

        # If structural grid is specified, search on grid intersections
        if structural_grid:
            grid_x, grid_y = structural_grid

            # Search on grid intersections
            for z in np.arange(z_min, z_max + 0.1, self.spatial_grid.grid_size):
                for x in np.arange(0, self.width - width + 0.1, grid_x):
                    for y in np.arange(0, self.length - length + 0.1, grid_y):
                        if self.can_place_room(width, length, height, (x, y, z)):
                            return (x, y, z)

        # Fallback: search with grid size intervals
        for z in np.arange(z_min, z_max + 0.1, self.spatial_grid.grid_size):
            for x in np.arange(
                0, self.width - width + 0.1, self.spatial_grid.grid_size
            ):
                for y in np.arange(
                    0, self.length - length + 0.1, self.spatial_grid.grid_size
                ):
                    if self.can_place_room(width, length, height, (x, y, z)):
                        return (x, y, z)

        # No valid position found
        return None

    def get_distance_between_rooms(
        self, room_id1: int, room_id2: int
    ) -> Optional[float]:
        """
        Calculate the distance between the centers of two rooms.

        Args:
            room_id1: First room ID
            room_id2: Second room ID

        Returns:
            Optional[float]: Distance in meters or None if either room is not found
        """
        room1 = self.get_room(room_id1)
        room2 = self.get_room(room_id2)

        if not room1 or not room2:
            return None

        # Calculate room centers
        x1, y1, z1 = room1["position"]
        w1, l1, h1 = room1["dimensions"]
        center1 = (x1 + w1 / 2, y1 + l1 / 2, z1 + h1 / 2)

        x2, y2, z2 = room2["position"]
        w2, l2, h2 = room2["dimensions"]
        center2 = (x2 + w2 / 2, y2 + l2 / 2, z2 + h2 / 2)

        # Calculate distance
        return distance(center1, center2)

    def calculate_space_utilization(self) -> float:
        """
        Calculate space utilization ratio (occupied volume / total volume).

        Returns:
            float: Space utilization ratio (0.0 to 1.0)
        """
        return self.spatial_grid.calculate_space_utilization()

    def clone(self) -> "Layout":
        """
        Create a deep copy of the layout.

        Returns:
            Layout: Cloned layout
        """
        # Create a copy of the spatial grid
        cloned_grid = SpatialGrid(
            width=self.width,
            length=self.length,
            height=self.height,
            grid_size=self.spatial_grid.grid_size,
        )

        # Copy grid data and rooms
        cloned_grid.grid = np.copy(self.spatial_grid.grid)
        cloned_grid.rooms = {
            room_id: room_data.copy() for room_id, room_data in self.rooms.items()
        }

        # Create a new layout with the cloned grid
        return Layout(
            spatial_grid=cloned_grid,
            name=f"{self.name}_clone",
            metadata=self.metadata.copy() if self.metadata else {},
        )

    def swap_rooms(self, room_id1: int, room_id2: int) -> bool:
        """
        Swap the positions of two rooms.

        Args:
            room_id1: First room ID
            room_id2: Second room ID

        Returns:
            bool: True if rooms were swapped successfully
        """
        room1 = self.get_room(room_id1)
        room2 = self.get_room(room_id2)

        if not room1 or not room2:
            return False

        # Record original positions and dimensions
        pos1 = room1["position"]
        dim1 = room1["dimensions"]
        type1 = room1["type"]
        metadata1 = room1.get("metadata", {})

        pos2 = room2["position"]
        dim2 = room2["dimensions"]
        type2 = room2["type"]
        metadata2 = room2.get("metadata", {})

        # Remove both rooms
        self.spatial_grid.remove_room(room_id1)
        self.spatial_grid.remove_room(room_id2)

        # Try to place room1 at position2
        success1 = self.spatial_grid.place_room(
            room_id=room_id1,
            x=pos2[0],
            y=pos2[1],
            z=pos2[2],
            width=dim1[0],
            length=dim1[1],
            height=dim1[2],
            room_type=type1,
            metadata=metadata1,
        )

        # Try to place room2 at position1
        success2 = self.spatial_grid.place_room(
            room_id=room_id2,
            x=pos1[0],
            y=pos1[1],
            z=pos1[2],
            width=dim2[0],
            length=dim2[1],
            height=dim2[2],
            room_type=type2,
            metadata=metadata2,
        )

        # If both placements succeed, clear caches and return True
        if success1 and success2:
            self._clear_caches()
            return True

        # If either placement fails, restore original positions
        if not success1 or not success2:
            # Clean up any successful placements
            if success1:
                self.spatial_grid.remove_room(room_id1)
            if success2:
                self.spatial_grid.remove_room(room_id2)

            # Restore original positions
            self.spatial_grid.place_room(
                room_id=room_id1,
                x=pos1[0],
                y=pos1[1],
                z=pos1[2],
                width=dim1[0],
                length=dim1[1],
                height=dim1[2],
                room_type=type1,
                metadata=metadata1,
            )

            self.spatial_grid.place_room(
                room_id=room_id2,
                x=pos2[0],
                y=pos2[1],
                z=pos2[2],
                width=dim2[0],
                length=dim2[1],
                height=dim2[2],
                room_type=type2,
                metadata=metadata2,
            )

            return False

    def create_room_objects(self) -> List[Room]:
        """
        Create Room objects from the layout's spatial grid.

        Returns:
            List[Room]: List of Room objects
        """
        rooms = []

        for room_id, room_data in self.rooms.items():
            x, y, z = room_data["position"]
            width, length, height = room_data["dimensions"]
            room_type = room_data["type"]
            metadata = room_data.get("metadata", {})
            name = metadata.get("name", f"{room_type}_{room_id}")

            room = Room(
                width=width,
                length=length,
                height=height,
                room_type=room_type,
                name=name,
                metadata=metadata,
                id=room_id,
            )

            # Set position
            room.position = (x, y, z)

            rooms.append(room)

        return rooms

    def get_room_boxes(
        self,
    ) -> Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Get bounding boxes for all rooms.

        Returns:
            Dict[int, Tuple]: Mapping of room ID to bounding box ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        return {
            room_id: create_room_box(room_data["position"], room_data["dimensions"])
            for room_id, room_data in self.rooms.items()
        }

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the layout to a dictionary.

        Returns:
            Dict: Serialized layout
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "spatial_grid": self.spatial_grid.to_dict(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Layout":
        """
        Create a Layout from serialized data.

        Args:
            data: Serialized layout data

        Returns:
            Layout: Deserialized layout
        """
        spatial_grid = SpatialGrid.from_dict(data["spatial_grid"])

        return cls(
            spatial_grid=spatial_grid,
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, filepath: str):
        """
        Save the layout to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "Layout":
        """
        Load a layout from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Layout: Loaded layout
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.deserialize(data)

    def __str__(self) -> str:
        return f"Layout(name={self.name}, rooms={len(self.rooms)})"
