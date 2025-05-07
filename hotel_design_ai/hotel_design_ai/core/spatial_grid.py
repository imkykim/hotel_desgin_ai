import numpy as np
from typing import Tuple, List, Dict, Set, Optional


class SpatialGrid:
    """
    3D spatial representation system for hotel layouts.
    Uses a voxel-based grid to represent spaces.
    Enhanced to support basement floors properly.
    """

    def __init__(
        self,
        width: float,
        length: float,
        height: float,
        grid_size: float = 1.0,
        min_floor: int = -2,
        floor_height: float = 5.0,
    ):
        """
        Initialize a 3D spatial grid with support for basement floors.

        Args:
            width: Total width of the bounding box in meters
            length: Total length of the bounding box in meters
            height: Total height of the bounding box in meters
            grid_size: Size of each grid cell in meters
            min_floor: Lowest basement floor (negative number)
            floor_height: Height of each floor in meters
        """
        self.width = width
        self.length = length
        self.height = height
        self.grid_size = grid_size
        self.min_floor = min_floor
        self.floor_height = floor_height

        # Calculate total height including basement
        total_height = height
        if min_floor < 0:
            total_height += abs(min_floor) * floor_height

        # Calculate grid dimensions
        self.width_cells = int(width / grid_size)
        self.length_cells = int(length / grid_size)
        self.height_cells = int(total_height / grid_size)

        # Calculate z-offset for converting between real coordinates and grid indices
        self.z_offset = 0
        if min_floor < 0:
            self.z_offset = int(abs(min_floor) * floor_height / grid_size)

        # Initialize 3D grid (0 = empty space, positive int = room ID)
        self.grid = np.zeros(
            (self.width_cells, self.length_cells, self.height_cells), dtype=int
        )

        # Room metadata
        self.rooms = {}  # room_id -> metadata

    def _convert_to_grid_indices(
        self, x: float, y: float, z: float
    ) -> Tuple[int, int, int]:
        """
        Convert real-world coordinates to grid indices, handling negative z-coordinates.

        Args:
            x, y, z: Real-world coordinates

        Returns:
            Tuple of grid indices (gx, gy, gz)
        """
        gx = int(x / self.grid_size)
        gy = int(y / self.grid_size)
        gz = int(z / self.grid_size) + self.z_offset  # Apply offset for basement

        return gx, gy, gz

    def _is_valid_placement(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        grid_width: int,
        grid_length: int,
        grid_height: int,
    ) -> bool:
        """Check if a room can be placed at the specified location"""
        # Check bounds
        if (
            grid_x < 0
            or grid_y < 0
            or grid_z < 0  # Now grid_z should always be non-negative
            or grid_x + grid_width > self.width_cells
            or grid_y + grid_length > self.length_cells
            or grid_z + grid_height > self.height_cells
        ):
            return False

        # Check for collisions with existing rooms
        try:
            # Get the region of the grid where the room would be placed
            target_region = self.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # If any cell in this region is non-zero, there's a collision
            return np.all(target_region == 0)
        except IndexError:
            # If there's an index error, the room would be out of bounds
            return False

    # Modify this section in SpatialGrid.place_room method
    def place_room(
        self,
        room_id: int,
        x: float,
        y: float,
        z: float,
        width: float,
        length: float,
        height: float,
        room_type: str = "generic",
        metadata: Dict = None,
        allow_overlap: List[str] = None,
        force_placement: bool = False,  # Add force_placement parameter
    ):
        """
        Place a room in the grid at the specified position.
        Enhanced to handle overlaps between parking and vertical circulation.

        Args:
            room_id: Unique identifier for the room
            x, y, z: Coordinates of the room's origin (bottom-left corner)
            width, length, height: Dimensions of the room
            room_type: Type of room (e.g., "guest_room", "lobby", etc.)
            metadata: Additional room information
            allow_overlap: List of room types this room can overlap with
            force_placement: If True, force placement even if it would overlap

        Returns:
            bool: True if placement was successful, False otherwise
        """
        if height > 50:
            force_placement = True

        # Default allowed overlap types
        if allow_overlap is None:
            # Allow bidirectional overlap between parking and vertical circulation
            if room_type == "vertical_circulation":
                allow_overlap = ["parking"]
            elif room_type == "parking":
                allow_overlap = ["vertical_circulation"]
            else:
                allow_overlap = []

        # Convert to grid coordinates using the helper method
        grid_x, grid_y, grid_z = self._convert_to_grid_indices(x, y, z)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

        # Check if placement is valid or force placement
        if not force_placement and not self._is_valid_placement_with_overlaps(
            grid_x,
            grid_y,
            grid_z,
            grid_width,
            grid_length,
            grid_height,
            room_type,
            allow_overlap,
        ):
            return False

        # Remember existing room IDs in the target region to handle overlaps
        existing_room_ids = set()
        try:
            # Get the region where the room would be placed
            target_region = self.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # Collect existing room IDs
            unique_ids = np.unique(target_region)
            for uid in unique_ids:
                if uid != 0:  # Skip empty cells
                    existing_room_ids.add(int(uid))
        except IndexError:
            return False  # Grid indices out of bounds

        # If force_placement, we need to update overlapping rooms
        if force_placement and existing_room_ids:
            # For each existing room, add this room to its overlaps
            for existing_id in existing_room_ids:
                if existing_id in self.rooms:
                    # Skip if overlap is not allowed
                    if not allow_overlap and room_type not in self.rooms.get(
                        existing_id, {}
                    ).get("allowed_overlap_types", []):
                        continue

                    # Add to overlaps
                    if "overlaps_with" not in self.rooms[existing_id]:
                        self.rooms[existing_id]["overlaps_with"] = [room_id]
                    else:
                        self.rooms[existing_id]["overlaps_with"].append(room_id)

        # Place the room in the grid
        self.grid[
            grid_x : grid_x + grid_width,
            grid_y : grid_y + grid_length,
            grid_z : grid_z + grid_height,
        ] = room_id

        # Store room metadata
        self.rooms[room_id] = {
            "id": room_id,
            "type": room_type,
            "position": (x, y, z),
            "dimensions": (width, length, height),
            "grid_position": (grid_x, grid_y, grid_z),
            "grid_dimensions": (grid_width, grid_length, grid_height),
            "allowed_overlap_types": allow_overlap,  # Store allowed overlap types
            **(metadata or {}),
        }

        # For rooms that can overlap, store the overlapping relationship
        if existing_room_ids:
            # Create a new metadata field or add to existing
            if "overlaps_with" not in self.rooms[room_id]:
                self.rooms[room_id]["overlaps_with"] = list(existing_room_ids)
            else:
                self.rooms[room_id]["overlaps_with"].extend(list(existing_room_ids))

        return True

    def _is_valid_placement_with_overlaps(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        grid_width: int,
        grid_length: int,
        grid_height: int,
        room_type: str,
        allow_overlap: List[str],
    ) -> bool:
        """
        Enhanced method to check if a room can be placed at the specified location,
        with special handling for allowed overlaps.

        Args:
            grid_x, grid_y, grid_z: Grid coordinates
            grid_width, grid_length, grid_height: Grid dimensions
            room_type: Type of room being placed
            allow_overlap: List of room types this room can overlap with

        Returns:
            bool: True if placement is valid, False otherwise
        """
        # Check bounds
        if (
            grid_x < 0
            or grid_y < 0
            or grid_z < 0  # Now grid_z should always be non-negative
            or grid_x + grid_width > self.width_cells
            or grid_y + grid_length > self.length_cells
            or grid_z + grid_height > self.height_cells
        ):
            return False

        # Check for collisions with existing rooms
        try:
            # Get the region of the grid where the room would be placed
            target_region = self.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # If any cell is non-zero, check if the existing room type is in allow_overlap
            existing_room_ids = set(np.unique(target_region)) - {0}

            # If no existing rooms, placement is valid
            if not existing_room_ids:
                return True

            # If there are existing rooms and overlaps are allowed
            if allow_overlap:
                # Check if all existing rooms are of allowed types
                for room_id in existing_room_ids:
                    if room_id in self.rooms:
                        existing_room_type = self.rooms[room_id]["type"]
                        if existing_room_type not in allow_overlap:
                            return False  # Overlap not allowed with this room type
                    else:
                        return False  # Unknown room ID, assume overlap not allowed

                # All overlaps are with allowed room types
                return True

            # No overlaps allowed
            return False

        except IndexError:
            # If there's an index error, the room would be out of bounds
            return False

    def remove_room(self, room_id: int) -> bool:
        """
        Remove a room from the grid.

        Args:
            room_id: ID of the room to remove

        Returns:
            bool: True if room was found and removed, False otherwise
        """
        if room_id not in self.rooms:
            return False

        # Get room grid coordinates
        grid_x, grid_y, grid_z = self.rooms[room_id]["grid_position"]
        grid_width, grid_length, grid_height = self.rooms[room_id]["grid_dimensions"]

        # Clear room from grid
        self.grid[
            grid_x : grid_x + grid_width,
            grid_y : grid_y + grid_length,
            grid_z : grid_z + grid_height,
        ] = 0

        # Remove from room metadata
        del self.rooms[room_id]

        return True

    # Rest of the methods remain largely unchanged
    # Just ensure any method that converts between real coordinates and grid coordinates
    # uses the _convert_to_grid_indices method or applies the offset correctly

    def are_adjacent(self, room_id1: int, room_id2: int) -> bool:
        """Check if two rooms are adjacent to each other"""
        if room_id1 not in self.rooms or room_id2 not in self.rooms:
            return False

        # Create a temporary grid with only these two rooms
        temp_grid = np.zeros_like(self.grid)

        # Fill room 1
        room1 = self.rooms[room_id1]
        x1, y1, z1 = room1["grid_position"]
        w1, l1, h1 = room1["grid_dimensions"]
        temp_grid[x1 : x1 + w1, y1 : y1 + l1, z1 : z1 + h1] = 1

        # Fill room 2
        room2 = self.rooms[room_id2]
        x2, y2, z2 = room2["grid_position"]
        w2, l2, h2 = room2["grid_dimensions"]
        temp_grid[x2 : x2 + w2, y2 : y2 + l2, z2 : z2 + h2] = 2

        # Expand room 1 by 1 cell in all directions
        expanded = np.zeros_like(temp_grid)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    # Skip diagonals, only consider face adjacency
                    if abs(dx) + abs(dy) + abs(dz) != 1:
                        continue

                    # Compute shifted indices
                    x_min = max(0, x1 + dx)
                    x_max = min(self.width_cells, x1 + w1 + dx)
                    y_min = max(0, y1 + dy)
                    y_max = min(self.length_cells, y1 + l1 + dy)
                    z_min = max(0, z1 + dz)
                    z_max = min(self.height_cells, z1 + h1 + dz)

                    # Set expanded cells
                    expanded[x_min:x_max, y_min:y_max, z_min:z_max] = 1

        # Check if expanded room 1 overlaps with room 2
        return np.any((expanded == 1) & (temp_grid == 2))

    def get_room_neighbors(self, room_id: int) -> List[int]:
        """Get all rooms adjacent to the specified room"""
        if room_id not in self.rooms:
            return []

        neighbors = []
        for other_id in self.rooms:
            if other_id != room_id and self.are_adjacent(room_id, other_id):
                neighbors.append(other_id)

        return neighbors

    def get_exterior_rooms(self) -> List[int]:
        """Get rooms that are on the exterior of the building"""
        exterior_rooms = []

        # Check each face of the bounding box
        for room_id in self.rooms:
            room = self.rooms[room_id]
            x, y, z = room["grid_position"]
            w, l, h = room["grid_dimensions"]

            # Check if room touches any exterior face
            if (
                x == 0
                or y == 0
                or z
                == 0  # This is now the absolute bottom of the grid, including basement
                or x + w == self.width_cells
                or y + l == self.length_cells
                or z + h == self.height_cells
            ):
                exterior_rooms.append(room_id)

        return exterior_rooms

    def calculate_space_utilization(self) -> float:
        """Calculate what percentage of the volume is utilized"""
        occupied_cells = np.count_nonzero(self.grid)
        total_cells = self.width_cells * self.length_cells * self.height_cells
        return occupied_cells / total_cells if total_cells > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert the spatial grid to a serializable dictionary"""
        return {
            "dimensions": {
                "width": self.width,
                "length": self.length,
                "height": self.height,
                "grid_size": self.grid_size,
                "min_floor": self.min_floor,
                "floor_height": self.floor_height,
            },
            "rooms": self.rooms,
            "grid": self.grid.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SpatialGrid":
        """Create a SpatialGrid from a dictionary"""
        dimensions = data["dimensions"]
        grid = cls(
            width=dimensions["width"],
            length=dimensions["length"],
            height=dimensions["height"],
            grid_size=dimensions["grid_size"],
            min_floor=dimensions.get("min_floor", -2),
            floor_height=dimensions.get("floor_height", 5.0),
        )

        grid.grid = np.array(data["grid"])
        grid.rooms = data["rooms"]

        return grid
