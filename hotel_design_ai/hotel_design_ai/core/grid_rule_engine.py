"""
Rule-based layout generation engine that uses architectural principles to create hotel layouts.
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
import numpy as np
import random

# Add these imports at the top
import math
from hotel_design_ai.config.config_loader_grid import calculate_grid_aligned_dimensions

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room
from hotel_design_ai.core.base_engine import BaseEngine
from hotel_design_ai.core.constraints import create_default_constraints


class RuleEngine(BaseEngine):
    """
    Rule-based layout generation engine that uses architectural principles
    to create hotel layouts.
    """

    def __init__(
        self,
        bounding_box: Tuple[float, float, float],
        grid_size: float = 1.0,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        building_config: Optional[Dict[str, Any]] = None,
        grid_fraction: float = 0.5,
    ):
        """
        Initialize the rule engine.

        Args:
            bounding_box: (width, length, height) of buildable area in meters
            grid_size: Size of spatial grid cells in meters
            structural_grid: (x_spacing, y_spacing) of structural grid in meters
            building_config: Building configuration parameters
        """
        # Initialize the base engine
        super().__init__(bounding_box, grid_size, structural_grid, building_config)

        # Initialize room placement priorities from configuration instead of hardcoding
        self._init_placement_priorities()

        # Initialize adjacency preferences from constraints
        self._init_adjacency_preferences()

        # Initialize floor preferences dynamically
        self._init_floor_preferences()

        # Initialize exterior preferences
        self._init_exterior_preferences()

        self.grid_fraction = grid_fraction
        self.snap_size_x = self.structural_grid[0] * grid_fraction
        self.snap_size_y = self.structural_grid[1] * grid_fraction

    def _init_placement_priorities(self):
        """Initialize room placement priorities based on architectural importance"""
        # Default priorities - can be extended with configuration
        self.placement_priorities = {
            "entrance": 10,
            "lobby": 9,
            "vertical_circulation": 8,
            "restaurant": 7,
            "kitchen": 7,
            "meeting_room": 6,
            "guest_room": 5,
            "service_area": 4,
            "back_of_house": 3,
        }

        # Add any additional room types from configuration
        if (
            hasattr(self, "building_config")
            and "room_priorities" in self.building_config
        ):
            self.placement_priorities.update(self.building_config["room_priorities"])

    def _init_adjacency_preferences(self):
        """Initialize adjacency preferences from constraints"""
        try:
            # Try to get adjacency constraints from the config system
            from hotel_design_ai.config.config_loader import get_constraints_by_type

            adjacency_constraints = get_constraints_by_type("adjacency")

            # Build adjacency preferences dictionary
            self.adjacency_preferences = {}

            for constraint in adjacency_constraints:
                if "room_type1" in constraint and "room_type2" in constraint:
                    if constraint["room_type1"] not in self.adjacency_preferences:
                        self.adjacency_preferences[constraint["room_type1"]] = []

                    self.adjacency_preferences[constraint["room_type1"]].append(
                        constraint["room_type2"]
                    )
        except (ImportError, FileNotFoundError):
            # Fall back to default adjacency preferences
            self.adjacency_preferences = {
                "entrance": ["lobby", "vertical_circulation"],
                "lobby": ["entrance", "restaurant", "vertical_circulation"],
                "vertical_circulation": ["lobby", "guest_room"],
                "restaurant": ["lobby", "kitchen"],
                "kitchen": ["restaurant", "service_area"],
                "meeting_room": ["lobby", "vertical_circulation"],
                "guest_room": ["vertical_circulation"],
                "service_area": ["vertical_circulation", "back_of_house"],
                "back_of_house": ["service_area", "kitchen"],
            }

    def _init_floor_preferences(self):
        """
        Initialize floor preferences dynamically based on building_config and constraints.
        This makes the system adaptable to changes in floor requirements.
        """
        try:
            # Try to get floor constraints from the config system
            from hotel_design_ai.config.config_loader import get_constraints_by_type

            floor_constraints = get_constraints_by_type("floor")

            # Build floor preferences dictionary
            self.floor_preferences = {}

            for constraint in floor_constraints:
                if "room_type" in constraint and "floor" in constraint:
                    self.floor_preferences[constraint["room_type"]] = constraint[
                        "floor"
                    ]

        except (ImportError, FileNotFoundError):
            # Fall back to dynamically generated defaults based on building_config

            # Get floor range
            min_floor = self.min_floor
            max_floor = self.max_floor

            # Dynamic assignments based on floor range
            # Public areas on the "lowest habitable floor" (usually 0)
            public_floor = max(0, min_floor)

            # Upper areas on the "middle floor" if multiple available
            upper_floor = min(1, max_floor) if max_floor > 0 else max(0, min_floor)

            # Service areas on the "lowest floor" (basement if available)
            service_floor = min_floor

            self.floor_preferences = {
                # Public areas
                "entrance": public_floor,
                "lobby": public_floor,
                "restaurant": public_floor,
                "kitchen": public_floor,
                "meeting_room": public_floor,
                # Guest areas - None means distribute according to algorithm
                "guest_room": None,
                # Service/BOH areas
                "service_area": service_floor,
                "back_of_house": service_floor,
                "mechanical": service_floor,
                "parking": service_floor,
            }

        # Add any room types that appear in the building_config but not in the constraints
        if "preferred_floors" in self.building_config:
            preferred_floors = self.building_config["preferred_floors"]
            for room_type, floor in preferred_floors.items():
                if room_type not in self.floor_preferences:
                    self.floor_preferences[room_type] = floor

    def _init_exterior_preferences(self):
        """Initialize exterior access preferences from constraints"""
        try:
            # Try to get exterior constraints from the config system
            from hotel_design_ai.config.config_loader import get_constraints_by_type

            exterior_constraints = get_constraints_by_type("exterior")

            # Build exterior preferences dictionary
            self.exterior_preferences = {}

            for constraint in exterior_constraints:
                if "room_type" in constraint and "weight" in constraint:
                    # Convert weight to preference level: 2=required, 1=preferred, 0=no preference
                    weight = constraint["weight"]
                    preference = 2 if weight >= 2.0 else 1 if weight >= 0.5 else 0
                    self.exterior_preferences[constraint["room_type"]] = preference

        except (ImportError, FileNotFoundError):
            # Fall back to default exterior preferences
            self.exterior_preferences = {
                "entrance": 2,  # Required
                "lobby": 1,  # Preferred
                "restaurant": 1,  # Preferred
                "guest_room": 1,  # Preferred
                "vertical_circulation": 0,  # No preference
                "service_area": 0,  # No preference
                "back_of_house": 0,  # No preference
                "meeting_room": 1,  # Preferred
                "office": 1,  # Preferred
                "pool": 1,  # Preferred
                "retail": 1,  # Preferred
            }

    def adjust_room_dimensions_to_grid(self, room, grid_fraction=0.5):
        """
        Adjust room dimensions to align with the structural grid.
        Prioritizes full grid alignment over fractional grid.

        Args:
            room: Room object with area requirements
            grid_fraction: Fraction of grid to use (0.5 = half grid)

        Returns:
            Tuple[float, float]: Adjusted (width, length) in meters
        """
        # Use the utility function from grid_aligned_dimensions.py
        return calculate_grid_aligned_dimensions(
            area=room.width * room.length,
            grid_x=self.structural_grid[0],
            grid_y=self.structural_grid[1],
            min_width=getattr(room, "width", None),
            grid_fraction=grid_fraction,
        )

    def snap_position_to_grid(self, x, y, z, grid_fraction=0.5):
        """
        Snap a position to the nearest valid grid point.

        Args:
            x, y, z: Position coordinates
            grid_fraction: Fraction of grid to use (0.5 = half grid)

        Returns:
            Tuple[float, float, float]: Snapped position
        """
        # Get snap sizes based on grid fraction
        snap_x = self.structural_grid[0] * grid_fraction
        snap_y = self.structural_grid[1] * grid_fraction

        # First try full grid if close enough
        full_x = round(x / self.structural_grid[0]) * self.structural_grid[0]
        full_y = round(y / self.structural_grid[1]) * self.structural_grid[1]

        if abs(full_x - x) <= 0.2 * self.structural_grid[0]:
            x = full_x
        else:
            x = round(x / snap_x) * snap_x

        if abs(full_y - y) <= 0.2 * self.structural_grid[1]:
            y = full_y
        else:
            y = round(y / snap_y) * snap_y

        return x, y, z

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on architectural constraints.
        Modified to prioritize fixed position rooms and place them exactly.

        Args:
            rooms: List of Room objects to place

        Returns:
            SpatialGrid: The generated layout
        """
        # Reset the spatial grid if needed
        if hasattr(self.spatial_grid, "rooms") and self.spatial_grid.rooms:
            # Create a fresh spatial grid
            self.spatial_grid = SpatialGrid(
                width=self.width,
                length=self.length,
                height=self.height,
                grid_size=self.grid_size,
                min_floor=self.min_floor,
                floor_height=self.floor_height,
            )

        # ENHANCEMENT: First, separate rooms with fixed positions from others
        fixed_rooms = []
        unfixed_rooms = []

        for room in rooms:
            if hasattr(room, "position") and room.position is not None:
                fixed_rooms.append(room)
            else:
                unfixed_rooms.append(room)

        print(f"\nFound {len(fixed_rooms)} rooms with fixed positions")
        for i, room in enumerate(fixed_rooms):
            print(
                f"  Fixed Room #{i+1}: id={room.id}, type={room.room_type}, name={room.name}, pos={room.position}"
            )

        # ENHANCEMENT: Place fixed rooms first, regardless of priority
        print("\n=== Placing fixed position rooms first ===")
        placed_rooms_by_type = {}
        failed_fixed_rooms = []

        for room in fixed_rooms:
            print(
                f"\nPlacing fixed room: id={room.id}, type={room.room_type}, name={room.name}, pos={room.position}"
            )
            # Use force placement to ensure fixed positions are respected
            success = self.place_fixed_room_exactly(room, placed_rooms_by_type)

            if success:
                if room.room_type not in placed_rooms_by_type:
                    placed_rooms_by_type[room.room_type] = []
                placed_rooms_by_type[room.room_type].append(room.id)
                print(
                    f"  ✓ Successfully placed fixed room id={room.id} at {room.position}"
                )
            else:
                failed_fixed_rooms.append(room)
                print(f"  ✗ Failed to place fixed room id={room.id} at {room.position}")

        # Now sort remaining rooms by architectural priority
        print("\n=== Placing remaining rooms by priority ===")
        sorted_unfixed_rooms = sorted(
            unfixed_rooms,
            key=lambda r: self.placement_priorities.get(r.room_type, 0),
            reverse=True,
        )

        # Track rooms that failed placement
        failed_rooms = []

        # Place remaining rooms in priority order
        for room in sorted_unfixed_rooms:
            success = self.place_room_by_constraints(room, placed_rooms_by_type)

            if success:
                if room.room_type not in placed_rooms_by_type:
                    placed_rooms_by_type[room.room_type] = []
                placed_rooms_by_type[room.room_type].append(room.id)
            else:
                failed_rooms.append(room)
                print(
                    f"Failed to place room: {room.name} (id={room.id}, type={room.room_type})"
                )

        # Report placement statistics
        total_rooms = len(rooms)
        placed_rooms = total_rooms - len(failed_rooms) - len(failed_fixed_rooms)

        print(f"\nRoom placement statistics:")
        print(f"  Total rooms: {total_rooms}")
        print(
            f"  Placed successfully: {placed_rooms} ({placed_rooms / total_rooms * 100:.1f}%)"
        )
        print(f"  Failed fixed rooms: {len(failed_fixed_rooms)}")
        print(f"  Failed regular rooms: {len(failed_rooms)}")

        for room_type in placed_rooms_by_type:
            print(f"  {room_type}: {len(placed_rooms_by_type[room_type])} placed")

        return self.spatial_grid

    def place_fixed_room_exactly(self, room, placed_rooms_by_type):
        """
        Place a room with a fixed position exactly where specified, using force if necessary.

        Args:
            room: Room object with fixed position
            placed_rooms_by_type: Dictionary tracking placed rooms by type

        Returns:
            bool: True if successful, False otherwise
        """
        if not hasattr(room, "position") or room.position is None:
            print(f"Error: Room {room.id} has no fixed position")
            return False

        x, y, z = room.position
        print(
            f"Placing fixed room {room.id} ({room.room_type}, {room.name}) at EXACT position: {room.position}"
        )

        # Check if position is within building bounds
        if (
            x < 0
            or y < 0
            or x + room.width > self.width
            or y + room.length > self.length
            or z < self.min_floor * self.floor_height
            or z > self.max_floor * self.floor_height
        ):
            print(
                f"WARNING: Fixed position {room.position} for room {room.id} is outside building bounds!"
            )
            print(
                f"Building dimensions: {self.width}m × {self.length}m × {self.height}m"
            )
            print(f"Floor range: {self.min_floor} to {self.max_floor}")

        # Try first without force placement
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

        # If normal placement fails, use force placement
        if not success:
            print(
                f"Standard placement failed for fixed room {room.id}, using force placement"
            )
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
                force_placement=True,  # FORCE placement for fixed rooms
            )

        # STRICT: If still not successful, do NOT allow fallback placement
        if not success:
            print(
                f"ERROR: Could not place FIXED room {room.id} at {room.position} (even with force). This room will NOT be placed."
            )
            return False

        print(
            f"Successfully placed fixed room {room.id} at exact position {room.position}"
        )
        return True

    def place_room_by_constraints(self, room, placed_rooms_by_type):
        """
        Place a room according to architectural constraints.
        Updated to use place_fixed_room_exactly for rooms with fixed positions.
        """
        # If the room has a fixed position, use the specialized method
        if hasattr(room, "position") and room.position is not None:
            # STRICT: Only try the exact fixed position, never fallback
            return self.place_fixed_room_exactly(room, placed_rooms_by_type)

        # The rest of the method remains unchanged for rooms without fixed positions
        original_width = room.width
        original_length = room.length

        # Apply grid alignment to dimensions
        width, length = self.adjust_room_dimensions_to_grid(room)

        # Temporarily update room dimensions for placement
        room.width = width
        room.length = length

        # Get preferred floors
        preferred_floors = self._get_preferred_floors(room)

        # Handle special room types
        if self._needs_special_handling(room.room_type):
            success = self._handle_special_room_type(
                room, placed_rooms_by_type, preferred_floors
            )

            # Restore original dimensions regardless of outcome
            room.width = original_width
            room.length = original_length

            return success

        for floor in preferred_floors:
            # Calculate z coordinate for this floor
            z = floor * self.floor_height

            # Try adjacency-based placement
            if room.room_type in self.adjacency_preferences:
                # (existing adjacency code)
                # But snap the final position to grid
                position = self._find_position_adjacent_to(
                    room, room.room_type, placed_rooms_by_type, z
                )
                if position:
                    x, y, z_pos = position
                    # Snap position to grid
                    x, y, z_pos = self.snap_position_to_grid(x, y, z_pos)

                    # Check if snapped position is still valid
                    if self._is_valid_position(x, y, z_pos, width, length, room.height):
                        success = self.spatial_grid.place_room(
                            room_id=room.id,
                            x=x,
                            y=y,
                            z=z_pos,
                            width=width,
                            length=length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=room.metadata,
                        )
                        if success:
                            # Restore original dimensions
                            room.width = original_width
                            room.length = original_length
                            return True

            # Check if room needs exterior access
            exterior_pref = self.exterior_preferences.get(room.room_type, 0)
            if exterior_pref > 0:
                # Try perimeter position
                position = self._find_perimeter_position(room, floor)
                if position:
                    x, y, z_pos = position
                    success = self.spatial_grid.place_room(
                        room_id=room.id,
                        x=x,
                        y=y,
                        z=z_pos,
                        width=room.width,
                        length=room.length,
                        height=room.height,
                        room_type=room.room_type,
                        metadata=room.metadata,
                    )
                    if success:
                        return True

                # If exterior is required but no position found, try next floor
                if exterior_pref == 2:
                    continue

            # Try any position on this floor
            position = self._find_position_on_floor(room, floor)
            if position:
                x, y, z_pos = position
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=z_pos,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )
                if success:
                    return True

        # If all preferred floors failed, try alternative floors
        alternate_floors = self._get_alternate_floors(room.room_type, preferred_floors)
        for alt_floor in alternate_floors:
            # Skip floors already tried in preferred_floors
            if alt_floor in preferred_floors:
                continue

            z_pos = alt_floor * self.floor_height
            position = self._find_position_on_floor(room, alt_floor)
            if position:
                x, y, z_pos = position
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=z_pos,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )
                if success:
                    return True

        # Restore original dimensions before returning
        room.width = original_width
        room.length = original_length
        return False

    def _needs_special_handling(self, room_type: str) -> bool:
        """
        Determine if a room type needs special handling.
        This makes it easy to extend with new room types that need special handling.

        Args:
            room_type: The type of room

        Returns:
            bool: True if the room type needs special handling
        """
        special_types = ["vertical_circulation", "entrance", "parking"]

        # Check building_config for additional special room types
        if (
            hasattr(self, "building_config")
            and "special_room_types" in self.building_config
        ):
            special_types.extend(self.building_config["special_room_types"])

        return room_type in special_types

    def _handle_special_room_type(self, room, placed_rooms_by_type, preferred_floors):
        """
        Handle placement for room types that need special processing.
        Modified to respect pre-set positions.
        """
        # Check if this room already has a position
        if hasattr(room, "position") and room.position is not None:
            x, y, z = room.position
            print(
                f"Using fixed position for special room {room.name} (id={room.id}): {room.position}"
            )
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
                # Add to placed rooms tracking if needed
                if room.room_type not in placed_rooms_by_type:
                    placed_rooms_by_type[room.room_type] = []
                if room.id not in placed_rooms_by_type[room.room_type]:
                    placed_rooms_by_type[room.room_type].append(room.id)
                return True

        # Original special handling logic
        if room.room_type == "vertical_circulation":
            return self._place_vertical_circulation(room)
        elif room.room_type == "entrance":
            return self._place_entrance(room, preferred_floors)
        elif room.room_type == "parking":
            return self._place_parking(room, preferred_floors, placed_rooms_by_type)

        # If we reach here, there's a configuration error - we said the room type
        # needs special handling but didn't provide a method
        print(
            f"Warning: Room type '{room.room_type}' is marked for special handling but no handler exists"
        )

        # Fall back to standard placement
        position = self._find_position_on_floor(
            room, preferred_floors[0] if preferred_floors else 0
        )
        if position:
            x, y, z_pos = position
            success = self.spatial_grid.place_room(
                room_id=room.id,
                x=x,
                y=y,
                z=z_pos,
                width=room.width,
                length=room.length,
                height=room.height,
                room_type=room.room_type,
                metadata=room.metadata,
            )
            if success:
                return True

        return False

    def _place_entrance(self, room: Room, preferred_floors: List[int]) -> bool:
        """Place entrance on perimeter with preference for front facade."""
        # Use the first preferred floor
        floor = preferred_floors[0] if preferred_floors else 0
        z = floor * self.floor_height

        # Try front edge first (y = 0)
        center_x = (self.width - room.width) / 2
        center_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]

        # Try center front position
        if self._is_valid_position(
            center_x, 0, z, room.width, room.length, room.height
        ):
            return self.spatial_grid.place_room(
                room_id=room.id,
                x=center_x,
                y=0,
                z=z,
                width=room.width,
                length=room.length,
                height=room.height,
                room_type=room.room_type,
                metadata=room.metadata,
            )

        # Try other front positions working outward from center
        for offset in range(1, 20):
            # Try left of center
            x_left = center_x - offset * self.structural_grid[0]
            if x_left >= 0 and self._is_valid_position(
                x_left, 0, z, room.width, room.length, room.height
            ):
                return self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x_left,
                    y=0,
                    z=z,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )

            # Try right of center
            x_right = center_x + offset * self.structural_grid[0]
            if x_right + room.width <= self.width and self._is_valid_position(
                x_right, 0, z, room.width, room.length, room.height
            ):
                return self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x_right,
                    y=0,
                    z=z,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )

        # Try side and back edges if front fails
        position = self._find_perimeter_position(room, floor)
        if position:
            x, y, z_pos = position
            return self.spatial_grid.place_room(
                room_id=room.id,
                x=x,
                y=y,
                z=z_pos,
                width=room.width,
                length=room.length,
                height=room.height,
                room_type=room.room_type,
                metadata=room.metadata,
            )

        return False

    def _place_vertical_circulation(self, room, preferred_floors=None):
        """
        Special placement for vertical circulation elements with no corridors.
        Modified to respect pre-set positions.
        """
        existing_cores = 0
        for room_id, room_data in self.spatial_grid.rooms.items():
            if room_data["type"] == "vertical_circulation":
                existing_cores += 1

        # ADDED: Skip if we already have a vertical circulation core
        if existing_cores > 0:
            print(
                f"Already have {existing_cores} vertical circulation core(s), skipping additional core placement"
            )
            return True  # Return True to indicate "success" and prevent additional placement attempts

        # Check if room already has a position
        if hasattr(room, "position") and room.position is not None:
            # Already handled in _handle_special_room_type
            return False  # Should never reach here if correct

        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        # Calculate full height from min to max floor
        total_floors = self.max_floor - self.min_floor + 1
        total_height = total_floors * self.floor_height

        # Starting z-coordinate from the lowest floor
        start_z = self.min_floor * self.floor_height

        # Update metadata
        if not room.metadata:
            room.metadata = {}
        room.metadata["is_core"] = True
        room.metadata["spans_floors"] = list(range(self.min_floor, self.max_floor + 1))

        # Try positions near the center first aligned to full grid
        center_x = ((self.width / 2) // self.structural_grid[0]) * self.structural_grid[
            0
        ]
        center_y = (
            (self.length / 2) // self.structural_grid[1]
        ) * self.structural_grid[1]

        # Generate positions in a spiral pattern starting from center
        positions = [(center_x, center_y)]

        # Add grid-aligned positions in a spiral
        grid_x, grid_y = self.structural_grid
        max_radius = min(self.width, self.length) / 4

        for radius in range(int(grid_x), int(max_radius) + 1, int(grid_x)):
            for angle in range(0, 360, 45):
                angle_rad = angle * np.pi / 180
                x = center_x + radius * np.cos(angle_rad)
                y = center_y + radius * np.sin(angle_rad)
                x = round(x / grid_x) * grid_x
                y = round(y / grid_y) * grid_y
                x = max(0, min(x, self.width - width))
                y = max(0, min(y, self.length - length))
                positions.append((x, y))

        # Remove duplicates
        positions = list(set(positions))

        # Try each position
        for x, y in positions:
            # Check if this position works for a vertical circulation spanning all floors
            if self._is_valid_position(
                x, y, start_z, room.width, room.length, total_height
            ):
                # Place the room spanning all floors
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=start_z,
                    width=room.width,
                    length=room.length,
                    height=total_height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                    allow_overlap=["parking"],  # Allow overlap with parking
                )

                if success:
                    print(
                        f"Successfully placed vertical circulation at ({x}, {y}, {start_z})"
                    )
                    return True

        # If all attempts fail, try to force placement as a last resort
        print("Using force placement for vertical circulation")
        x = grid_x
        y = grid_y

        # Force placement even if it has to overlap with existing rooms
        success = self.spatial_grid.place_room(
            room_id=room.id,
            x=x,
            y=y,
            z=start_z,
            width=room.width,
            length=room.length,
            height=total_height,
            room_type=room.room_type,
            metadata=room.metadata,
            allow_overlap=["parking"],  # Allow overlap with parking
            force_placement=True,  # Force placement
        )

        if success:
            print(f"Forced placement of vertical circulation at ({x}, {y}, {start_z})")
            return True

        # If all fails, report failure
        print("Failed to place vertical circulation")
        return False

    def _place_parking(
        self,
        room: Room,
        preferred_floors: List[int],
        placed_rooms_by_type: Dict[str, List[int]],
    ) -> bool:
        """
        Special placement logic for parking areas that can overlap with vertical circulation.

        Args:
            room: Parking room to place
            preferred_floors: List of preferred floor numbers
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully, False otherwise
        """
        # Use the first preferred floor
        floor = preferred_floors[0] if preferred_floors else self.min_floor
        z = floor * self.floor_height

        print(f"Placing parking area {room.id} at z={z}")

        # Try standard placement first
        position = self._find_position_on_floor(room, floor)
        if position:
            x, y, z_pos = position
            success = self.spatial_grid.place_room(
                room_id=room.id,
                x=x,
                y=y,
                z=z_pos,
                width=room.width,
                length=room.length,
                height=room.height,
                room_type=room.room_type,
                metadata=room.metadata,
                allow_overlap=[
                    "vertical_circulation"
                ],  # Allow overlap with vertical circulation
            )
            if success:
                print(f"Successfully placed parking at ({x}, {y}, {z_pos})")
                return True

        # If standard placement fails, try to place it even if it overlaps with vertical circulation
        # First, try positions near existing vertical circulation
        if "vertical_circulation" in placed_rooms_by_type:
            for circ_id in placed_rooms_by_type["vertical_circulation"]:
                if circ_id in self.spatial_grid.rooms:
                    circ_room = self.spatial_grid.rooms[circ_id]
                    circ_x, circ_y, circ_z = circ_room["position"]

                    # Try placing around the circulation
                    for offset_x in range(
                        0, int(self.width), int(self.structural_grid[0])
                    ):
                        for offset_y in range(
                            0, int(self.length), int(self.structural_grid[1])
                        ):
                            x = offset_x
                            y = offset_y

                            # Ensure the parking area is fully within bounds
                            if (
                                x + room.width > self.width
                                or y + room.length > self.length
                            ):
                                continue

                            # Try placing with overlap allowed
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
                                allow_overlap=["vertical_circulation"],  # Allow overlap
                            )

                            if success:
                                print(
                                    f"Placed parking at ({x}, {y}, {z}) with circulation overlap"
                                )
                                return True

        # If all else fails, try a forced placement
        print("Using force placement for parking as a last resort")
        x = self.structural_grid[0]
        y = self.structural_grid[1]

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
            allow_overlap=["vertical_circulation"],
            force_placement=True,  # Force placement
        )

        if success:
            print(f"Forced placement of parking at ({x}, {y}, {z})")
            return True

        print("Failed to place parking area")
        return False

    def _find_position_adjacent_to(
        self,
        room: Room,
        adjacent_type: str,
        placed_rooms_by_type: Dict[str, List[int]],
        z: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find position adjacent to a room of the specified type with grid alignment.

        Args:
            room: Room to place
            adjacent_type: Type of room to be adjacent to
            placed_rooms_by_type: Dictionary of already placed rooms by type
            z: Z-coordinate (floor level)

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        if adjacent_type not in placed_rooms_by_type:
            return None

        # Get grid sizes
        grid_x, grid_y = self.structural_grid
        half_grid_x = grid_x * 0.5
        half_grid_y = grid_y * 0.5

        # Try each room of the adjacent type
        for adj_room_id in placed_rooms_by_type[adjacent_type]:
            adj_room = self.spatial_grid.rooms[adj_room_id]
            adj_x, adj_y, adj_z = adj_room["position"]
            adj_w, adj_l, adj_h = adj_room["dimensions"]

            # Only consider if on the same floor
            if abs(adj_z - z) > 0.1:
                continue

            # Try positions in each direction
            positions = []

            # --- Right side (aligned to structural grid) ---
            # Snap the adjacent room's right edge to nearest grid line
            right_edge = adj_x + adj_w
            right_edge_snapped = round(right_edge / grid_x) * grid_x

            # Try different y positions along this edge, prioritizing full grid alignment
            for y_offset in range(0, int(adj_l), int(grid_y)):
                y = adj_y + y_offset
                # Snap to full grid
                y_snapped = round(y / grid_y) * grid_y
                positions.append((right_edge_snapped, y_snapped, z))

                # Only add half-grid positions if we're allowing them and they're different
                if self.grid_fraction <= 0.5:
                    # Snap to half grid
                    y_half_snapped = round(y / half_grid_y) * half_grid_y
                    if (
                        abs(y_half_snapped - y_snapped) > 0.1
                    ):  # Only if different from full grid
                        positions.append((right_edge_snapped, y_half_snapped, z))

            # --- Left side (aligned to structural grid) ---
            left_edge = adj_x - room.width
            left_edge_snapped = round(left_edge / grid_x) * grid_x

            for y_offset in range(0, int(adj_l), int(grid_y)):
                y = adj_y + y_offset
                # Snap to full grid
                y_snapped = round(y / grid_y) * grid_y
                positions.append((left_edge_snapped, y_snapped, z))

                # Half grid if enabled
                if self.grid_fraction <= 0.5:
                    y_half_snapped = round(y / half_grid_y) * half_grid_y
                    if abs(y_half_snapped - y_snapped) > 0.1:
                        positions.append((left_edge_snapped, y_half_snapped, z))

            # --- Bottom side (aligned to structural grid) ---
            bottom_edge = adj_y + adj_l
            bottom_edge_snapped = round(bottom_edge / grid_y) * grid_y

            for x_offset in range(0, int(adj_w), int(grid_x)):
                x = adj_x + x_offset
                # Snap to full grid
                x_snapped = round(x / grid_x) * grid_x
                positions.append((x_snapped, bottom_edge_snapped, z))

                # Half grid if enabled
                if self.grid_fraction <= 0.5:
                    x_half_snapped = round(x / half_grid_x) * half_grid_x
                    if abs(x_half_snapped - x_snapped) > 0.1:
                        positions.append((x_half_snapped, bottom_edge_snapped, z))

            # --- Top side (aligned to structural grid) ---
            top_edge = adj_y - room.length
            top_edge_snapped = round(top_edge / grid_y) * grid_y

            for x_offset in range(0, int(adj_w), int(grid_x)):
                x = adj_x + x_offset
                # Snap to full grid
                x_snapped = round(x / grid_x) * grid_x
                positions.append((x_snapped, top_edge_snapped, z))

                # Half grid if enabled
                if self.grid_fraction <= 0.5:
                    x_half_snapped = round(x / half_grid_x) * half_grid_x
                    if abs(x_half_snapped - x_snapped) > 0.1:
                        positions.append((x_half_snapped, top_edge_snapped, z))

            # Sort positions to try full grid positions first
            def is_full_grid(pos):
                px, py, _ = pos
                return (px % grid_x < 0.1 or px % grid_x > grid_x - 0.1) and (
                    py % grid_y < 0.1 or py % grid_y > grid_y - 0.1
                )

            positions.sort(key=lambda pos: 0 if is_full_grid(pos) else 1)

            # Try each position
            for pos in positions:
                x, y, z_pos = pos

                # Check bounds
                if (
                    x < 0
                    or y < 0
                    or x + room.width > self.width
                    or y + room.length > self.length
                ):
                    continue

                # Check if position is valid
                if self._is_valid_position(
                    x, y, z_pos, room.width, room.length, room.height
                ):
                    return pos

        return None

    def _find_perimeter_position(
        self, room: Room, floor: int
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find position on building perimeter for rooms needing exterior access with grid alignment.

        Args:
            room: Room to place
            floor: Floor to place on

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        z = floor * self.floor_height

        # Get grid sizes
        grid_x, grid_y = self.structural_grid
        half_grid_x = grid_x * 0.5
        half_grid_y = grid_y * 0.5

        # Create perimeter positions
        perimeter_positions = []

        # --- Full grid positions first ---

        # Front edge (y=0)
        for x in range(0, int(self.width - room.width) + 1, int(grid_x)):
            perimeter_positions.append((x, 0, z))

        # Back edge
        back_y = self.length - room.length
        back_y_snapped = round(back_y / grid_y) * grid_y
        for x in range(0, int(self.width - room.width) + 1, int(grid_x)):
            perimeter_positions.append((x, back_y_snapped, z))

        # Left edge
        for y in range(0, int(self.length - room.length) + 1, int(grid_y)):
            perimeter_positions.append((0, y, z))

        # Right edge
        right_x = self.width - room.width
        right_x_snapped = round(right_x / grid_x) * grid_x
        for y in range(0, int(self.length - room.length) + 1, int(grid_y)):
            perimeter_positions.append((right_x_snapped, y, z))

        # --- Half grid positions next, if enabled ---
        if self.grid_fraction <= 0.5:
            half_grid_positions = []

            # Front edge - half grid positions
            for x in range(0, int(self.width - room.width) + 1, int(half_grid_x)):
                if x % grid_x >= 0.1 * grid_x:  # Skip positions close to full grid
                    half_grid_positions.append((x, 0, z))

            # Back edge - half grid positions
            for x in range(0, int(self.width - room.width) + 1, int(half_grid_x)):
                if x % grid_x >= 0.1 * grid_x:
                    half_grid_positions.append((x, back_y_snapped, z))

            # Left edge - half grid positions
            for y in range(0, int(self.length - room.length) + 1, int(half_grid_y)):
                if y % grid_y >= 0.1 * grid_y:
                    half_grid_positions.append((0, y, z))

            # Right edge - half grid positions
            for y in range(0, int(self.length - room.length) + 1, int(half_grid_y)):
                if y % grid_y >= 0.1 * grid_y:
                    half_grid_positions.append((right_x_snapped, y, z))

            # Add after full grid positions
            perimeter_positions.extend(half_grid_positions)

        # Shuffle positions to avoid patterns in each group
        # But keep full grid positions first
        full_grid_positions = [
            pos
            for pos in perimeter_positions
            if (pos[0] % grid_x < 0.1 or pos[0] % grid_x > grid_x - 0.1)
            and (pos[1] % grid_y < 0.1 or pos[1] % grid_y > grid_y - 0.1)
        ]

        half_grid_positions = [
            pos for pos in perimeter_positions if pos not in full_grid_positions
        ]

        import random

        random.shuffle(full_grid_positions)
        random.shuffle(half_grid_positions)

        # Try full grid positions first, then half grid
        sorted_positions = full_grid_positions + half_grid_positions

        # Check each position
        for pos in sorted_positions:
            x, y, z_pos = pos

            # Check bounds
            if (
                x < 0
                or y < 0
                or x + room.width > self.width
                or y + room.length > self.length
            ):
                continue

            # Check if position is valid
            if self._is_valid_position(
                x, y, z_pos, room.width, room.length, room.height
            ):
                return pos

        return None

    def _find_position_on_floor(self, room, floor):
        """Find any valid position on a specific floor with grid alignment."""
        z = floor * self.floor_height

        # First try full grid positions
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # If full grid fails, try half-grid positions
        half_grid_x = self.structural_grid[0] * 0.5
        half_grid_y = self.structural_grid[1] * 0.5

        for x in range(0, int(self.width - room.width) + 1, int(half_grid_x)):
            for y in range(0, int(self.length - room.length) + 1, int(half_grid_y)):
                # Skip positions already tried with full grid
                if (
                    x % self.structural_grid[0] == 0
                    and y % self.structural_grid[1] == 0
                ):
                    continue

                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        return None

    def _get_alternate_floors(
        self, room_type: str, used_floors: List[int]
    ) -> List[int]:
        """
        Get alternative floors for a room type if the preferred floors are full.
        Enhanced to avoid floors already tried.

        Args:
            room_type: Type of room
            used_floors: Floors already tried

        Returns:
            List of alternative floor numbers
        """
        # Define floor ranges for different room categories based on actual building config
        habitable_floors = [
            f for f in range(max(0, self.min_floor), self.max_floor + 1)
        ]
        upper_floors = [f for f in range(1, self.max_floor + 1)]
        basement_floors = [f for f in range(self.min_floor, min(0, self.max_floor + 1))]

        # Get all floors to try
        all_floors = list(range(self.min_floor, self.max_floor + 1))

        # First remove the floors we've already tried
        candidate_floors = [f for f in all_floors if f not in used_floors]

        # If all floors have been tried, return empty list
        if not candidate_floors:
            return []

        # Now prioritize floors based on room type
        if room_type == "guest_room":
            # Guest rooms preferably on upper floors, but could go on ground floor if needed
            return sorted(
                candidate_floors,
                key=lambda f: (-1 if f in upper_floors else 0 if f == 0 else 1),
            )

        elif room_type in ["office", "staff_area"]:
            # Office spaces preferably on upper floors or ground
            return sorted(
                candidate_floors,
                key=lambda f: (-1 if f in upper_floors else 0 if f == 0 else 1),
            )

        elif room_type in ["mechanical", "maintenance", "back_of_house", "parking"]:
            # Service spaces preferably in basement, then upper floors
            return sorted(
                candidate_floors, key=lambda f: (-1 if f in basement_floors else 0)
            )

        elif room_type in ["meeting_room", "food_service", "restaurant", "retail"]:
            # Public areas preferably on ground or 1st floor
            priority_floors = [0, 1] if 1 in all_floors else [0]
            return sorted(
                candidate_floors, key=lambda f: (-1 if f in priority_floors else 0)
            )

        else:
            # For other types, no particular ordering
            return candidate_floors
