"""
Base engine for layout generation with common functionality used by both
rule-based and reinforcement learning engines.
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
from abc import ABC, abstractmethod

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


class BaseEngine(ABC):
    """
    Base class for layout generation engines with common functionality
    shared between rule-based and RL-based approaches.
    """

    def __init__(
        self,
        bounding_box: Tuple[float, float, float],
        grid_size: float = 1.0,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        building_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the base engine.

        Args:
            bounding_box: (width, length, height) of buildable area in meters
            grid_size: Size of spatial grid cells in meters
            structural_grid: (x_spacing, y_spacing) of structural grid in meters
            building_config: Building configuration parameters
        """
        self.width, self.length, self.height = bounding_box
        self.grid_size = grid_size
        self.structural_grid = structural_grid

        # Store building configuration
        self.building_config = building_config or {
            "floor_height": 5.0,
            "min_floor": -1,
            "max_floor": 3,
        }

        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(
            width=self.width,
            length=self.length,
            height=self.height,
            grid_size=self.grid_size,
            min_floor=self.building_config.get("min_floor", -1),
            floor_height=self.building_config.get("floor_height", 5.0),
        )

        # Floor range
        self.min_floor = self.building_config.get("min_floor", -1)
        self.max_floor = self.building_config.get("max_floor", 3)
        self.floor_height = self.building_config.get("floor_height", 5.0)

        # Fixed elements (set by user)
        self.fixed_elements = {}  # room_id -> position

        # Standard floor zones (for hotel tower portion)
        self.standard_floor_zones = []
        self.standard_floor_mask = None

    def update_fixed_elements(
        self, fixed_elements: Dict[int, Tuple[float, float, float]]
    ):
        """
        Update the fixed elements (user placed/modified rooms).

        Args:
            fixed_elements: Dictionary mapping room IDs to positions
        """
        self.fixed_elements = fixed_elements.copy()

    def clear_non_fixed_elements(self):
        """Remove all non-fixed elements from the spatial grid"""
        all_room_ids = list(self.spatial_grid.rooms.keys())
        for room_id in all_room_ids:
            if room_id not in self.fixed_elements:
                self.spatial_grid.remove_room(room_id)

    def set_standard_floor_zones(self, floor_zones):
        """
        Set standard floor zones for the hotel tower portion.

        Args:
            floor_zones: List of dicts with x, y, width, height defining standard floor areas
        """
        self.standard_floor_zones = floor_zones

        # Create a mask for standard floor areas
        self.standard_floor_mask = np.zeros(
            (int(self.width), int(self.length)), dtype=bool
        )

        # Mark standard floor zones in the mask
        for zone in floor_zones:
            x1 = int(zone["x"] / self.grid_size)
            y1 = int(zone["y"] / self.grid_size)
            x2 = int((zone["x"] + zone["width"]) / self.grid_size)
            y2 = int((zone["y"] + zone["height"]) / self.grid_size)

            # Ensure within bounds
            x1 = max(0, min(x1, self.spatial_grid.width_cells - 1))
            y1 = max(0, min(y1, self.spatial_grid.length_cells - 1))
            x2 = max(0, min(x2, self.spatial_grid.width_cells - 1))
            y2 = max(0, min(y2, self.spatial_grid.length_cells - 1))

            self.standard_floor_mask[x1:x2, y1:y2] = True

        print(f"Set {len(floor_zones)} standard floor zones")

    def _place_in_standard_floor_zone(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Place a room within the standard floor zones.

        Args:
            room: The room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placement was successful
        """
        if not hasattr(self, "standard_floor_mask"):
            # If no standard floor zones defined, use regular placement
            return self.place_room_by_constraints(room, placed_rooms_by_type)

        # Get floor and z coordinate
        floor = room.floor if hasattr(room, "floor") and room.floor is not None else 1
        z = floor * self.floor_height

        # Try positions within standard floor zones
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                # Check if position is within standard floor zone
                grid_x = int(x / self.grid_size)
                grid_y = int(y / self.grid_size)
                room_width_cells = int(room.width / self.grid_size)
                room_length_cells = int(room.length / self.grid_size)

                # Skip if any part of room would be outside standard zone
                if not np.all(
                    self.standard_floor_mask[
                        grid_x : grid_x + room_width_cells,
                        grid_y : grid_y + room_length_cells,
                    ]
                ):
                    continue

                # Check if position is valid
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    success = self.spatial_grid.place_room(
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
                    if success:
                        return True

        # If no position found in standard zones
        return False

    def _is_valid_position(
        self, x: float, y: float, z: float, width: float, length: float, height: float
    ) -> bool:
        """
        Check if a position is valid for room placement.

        Args:
            x, y, z: Coordinates of the position
            width, length, height: Dimensions of the room

        Returns:
            bool: True if the position is valid
        """
        # Check building bounds
        if x < 0 or y < 0 or x + width > self.width or y + length > self.length:
            return False

        # Check floor range
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)
        floor_height = self.building_config.get("floor_height", 5.0)

        min_z = min_floor * floor_height
        max_z = (max_floor + 1) * floor_height

        if z < min_z or z + height > max_z:
            return False

        # Convert to grid coordinates using SpatialGrid's helper method if available
        if hasattr(self.spatial_grid, "_convert_to_grid_indices"):
            grid_x, grid_y, grid_z = self.spatial_grid._convert_to_grid_indices(x, y, z)
        else:
            # Fallback to manual conversion
            grid_x = int(x / self.grid_size)
            grid_y = int(y / self.grid_size)
            grid_z = int(z / self.grid_size)

            # Apply z-offset manually if SpatialGrid has one
            if hasattr(self.spatial_grid, "z_offset"):
                grid_z += self.spatial_grid.z_offset

        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

        # Check grid bounds
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
        try:
            # Get the region where the room would be placed
            target_region = self.spatial_grid.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # If any cell is non-zero, there's a collision
            return np.all(target_region == 0)
        except IndexError:
            return False  # Grid indices out of bounds

    def _get_preferred_floors(self, room: Room) -> List[int]:
        """
        Get preferred floors for a room, searching multiple sources.

        Args:
            room: The room to get preferred floors for

        Returns:
            List[int]: List of preferred floor numbers
        """
        preferred_floors = []

        # 1. Check for preferred_floors attribute
        if hasattr(room, "preferred_floors") and room.preferred_floors is not None:
            if isinstance(room.preferred_floors, list):
                preferred_floors = room.preferred_floors
            else:
                preferred_floors = [room.preferred_floors]

        # 2. Or check metadata for preferred_floors
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            floors = room.metadata["preferred_floors"]
            if isinstance(floors, list):
                preferred_floors = floors
            else:
                preferred_floors = [floors]

        # 3. Or check for a floor attribute
        elif hasattr(room, "floor") and room.floor is not None:
            if isinstance(room.floor, list):
                preferred_floors = room.floor
            else:
                preferred_floors = [room.floor]

        # 4. Or check floor in metadata
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            floor = room.metadata["floor"]
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor]

        # 5. If still no preferred floors, use defaults
        if not preferred_floors:
            # Default to ground floor or min habitable floor
            preferred_floors = [max(0, self.min_floor)]

        # Ensure all floor values are integers
        return [int(floor) for floor in preferred_floors if floor is not None]

    @abstractmethod
    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on specified constraints.
        This is an abstract method that must be implemented by subclasses.

        Args:
            rooms: List of Room objects to place

        Returns:
            SpatialGrid: The generated layout
        """
        pass

    @abstractmethod
    def place_room_by_constraints(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Place a room according to architectural constraints.
        This is an abstract method that must be implemented by subclasses.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        pass
