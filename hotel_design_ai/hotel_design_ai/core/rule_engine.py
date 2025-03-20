"""
Rule-based layout generation engine that uses architectural principles to create hotel layouts.
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
import numpy as np
import random
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room
from hotel_design_ai.core.constraints import create_default_constraints


class RuleEngine:
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
    ):
        """
        Initialize the rule engine.

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

        # Initialize room placement priorities from configuration instead of hardcoding
        self._init_placement_priorities()

        # Initialize adjacency preferences from constraints
        self._init_adjacency_preferences()

        # Initialize floor preferences dynamically
        self._init_floor_preferences()

        # Initialize exterior preferences
        self._init_exterior_preferences()

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

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on architectural constraints.

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

        # Sort rooms by architectural priority
        sorted_rooms = sorted(
            rooms,
            key=lambda r: self.placement_priorities.get(r.room_type, 0),
            reverse=True,
        )

        # Track placed rooms by type
        placed_rooms_by_type = {}

        # Track rooms that failed placement
        failed_rooms = []

        # Place rooms in priority order
        for room in sorted_rooms:
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
        placed_rooms = total_rooms - len(failed_rooms)

        print(f"\nRoom placement statistics:")
        print(f"  Total rooms: {total_rooms}")
        print(
            f"  Placed successfully: {placed_rooms} ({placed_rooms / total_rooms * 100:.1f}%)"
        )
        print(
            f"  Failed to place: {len(failed_rooms)} ({len(failed_rooms) / total_rooms * 100:.1f}%)"
        )

        for room_type in placed_rooms_by_type:
            print(f"  {room_type}: {len(placed_rooms_by_type[room_type])} placed")

        return self.spatial_grid

    def place_room_by_constraints(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Place a room according to architectural constraints.
        Enhanced to support a list of preferred floors.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        # First try to extract preferred floors from the room's attributes
        # Check different sources in order of precedence

        # 1. Check for preferred_floors attribute
        preferred_floors = None
        if hasattr(room, "preferred_floors") and room.preferred_floors is not None:
            # Use the preferred_floors attribute directly if available
            preferred_floors = room.preferred_floors

        # 2. Check for preferred_floors in metadata
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            preferred_floors = room.metadata["preferred_floors"]

        # 3. Check for a single floor attribute on the room
        elif hasattr(room, "floor") and room.floor is not None:
            # If it's a single floor value, convert to a list
            preferred_floors = [room.floor]

        # 4. Check for a single floor in metadata
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            preferred_floors = [room.metadata["floor"]]

        # 5. Use default floor preferences by room type
        if preferred_floors is None:
            default_floor = self.floor_preferences.get(room.room_type)
            if default_floor is not None:
                # Convert single floor to list
                preferred_floors = [default_floor]
            else:
                # Handle special case for guest rooms distribution
                if room.room_type == "guest_room":
                    # Distribute guest rooms across upper floors starting from 1 or min habitable floor
                    start_floor = (
                        max(1, self.min_floor) if self.max_floor > 0 else self.min_floor
                    )
                    floor = start_floor + (
                        len(placed_rooms_by_type.get("guest_room", []))
                        % max(1, self.max_floor - start_floor + 1)
                    )
                    preferred_floors = [floor]
                else:
                    # Default to ground floor or min habitable floor
                    preferred_floors = [max(0, self.min_floor)]

        # Try each preferred floor in order
        for floor_value in preferred_floors:
            # Ensure floor_value is an integer
            try:
                floor_value = int(floor_value)
                # Calculate z coordinate for this floor
                z = floor_value * self.floor_height

                # Handle special room types
                if self._needs_special_handling(room.room_type):
                    success = self._handle_special_room_type(
                        room, placed_rooms_by_type, z
                    )
                    if success:
                        return True
                    else:
                        # Continue to next floor if this floor failed for special handling
                        continue
            except (ValueError, TypeError):
                # Skip this floor value if it can't be converted to an integer
                print(
                    f"Warning: Invalid floor value {floor_value} for room {room.name}, skipping"
                )
                continue
            # Try to place based on adjacency requirements
            if room.room_type in self.adjacency_preferences:
                adjacent_types = self.adjacency_preferences[room.room_type]
                for adj_type in adjacent_types:
                    if adj_type in placed_rooms_by_type:
                        position = self._find_position_adjacent_to(
                            room, adj_type, placed_rooms_by_type, z
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

        # If all attempts fail, report failure
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

    def _handle_special_room_type(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]], z: float
    ) -> bool:
        """
        Handle placement for room types that need special processing.
        This makes it easy to add new special room types.

        Args:
            room: The room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type
            z: Z-coordinate for placement

        Returns:
            bool: True if room was placed successfully
        """
        if room.room_type == "vertical_circulation":
            return self._place_vertical_circulation(room)
        elif room.room_type == "entrance":
            return self._place_entrance(room, z)
        elif room.room_type == "parking":
            return self._place_parking(room, z, placed_rooms_by_type)

        # If we reach here, there's a configuration error - we said the room type
        # needs special handling but didn't provide a method
        print(
            f"Warning: Room type '{room.room_type}' is marked for special handling but no handler exists"
        )

        # Fall back to standard placement
        position = self._find_position_on_floor(room, int(z / self.floor_height))
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

    def _place_entrance(self, room: Room, z: float) -> bool:
        """Place entrance on perimeter with preference for front facade."""
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
        position = self._find_perimeter_position(room, int(z / self.floor_height))
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

    def _place_vertical_circulation(self, room: Room) -> bool:
        """
        Special placement logic for vertical circulation elements (elevators, stairs)
        that need to span multiple floors from minimum floor to maximum floor.

        Args:
            room: Vertical circulation room to place

        Returns:
            bool: True if placed successfully, False otherwise
        """
        # Calculate full height from min to max floor
        total_floors = self.max_floor - self.min_floor + 1
        total_height = total_floors * self.floor_height

        # Starting z-coordinate from the lowest floor (basement)
        start_z = self.min_floor * self.floor_height

        # Update the metadata to show this is a core element
        if not room.metadata:
            room.metadata = {}
        room.metadata["is_core"] = True
        room.metadata["spans_floors"] = list(range(self.min_floor, self.max_floor + 1))

        # Print debug info
        print(
            f"Placing vertical circulation {room.id} with height {total_height} from z={start_z}"
        )

        # Find a valid (x,y) position that works for the entire vertical span
        # Start with a grid search aligned to structural grid
        grid_x = self.structural_grid[0]
        grid_y = self.structural_grid[1]

        # Try center positions first for better architecture
        center_x = self.width / 2
        center_y = self.length / 2

        # Generate positions in a spiral pattern starting from center
        positions = []
        max_radius = max(self.width, self.length) / 2

        for radius in range(0, int(max_radius) + 1, int(min(grid_x, grid_y))):
            # Add positions in a rough "circle" at this radius
            for angle in range(0, 360, 45):  # 45 degree increments
                # Convert polar to cartesian coordinates
                angle_rad = angle * 3.14159 / 180
                x = center_x + radius * np.cos(angle_rad)
                y = center_y + radius * np.sin(angle_rad)

                # Snap to grid
                x = round(x / grid_x) * grid_x
                y = round(y / grid_y) * grid_y

                # Ensure within bounds
                x = max(0, min(x, self.width - room.width))
                y = max(0, min(y, self.length - room.length))

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
        self, room: Room, z: float, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Special placement logic for parking areas that can overlap with vertical circulation.

        Args:
            room: Parking room to place
            z: Z-coordinate (height) for the room
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully, False otherwise
        """
        print(f"Placing parking area {room.id} at z={z}")

        # Try standard placement first
        position = self._find_position_on_floor(room, int(z / self.floor_height))
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
        """Find position adjacent to a room of the specified type."""
        if adjacent_type not in placed_rooms_by_type:
            return None

        # Try each room of the adjacent type
        for adj_room_id in placed_rooms_by_type[adjacent_type]:
            adj_room = self.spatial_grid.rooms[adj_room_id]
            adj_x, adj_y, adj_z = adj_room["position"]
            adj_w, adj_l, adj_h = adj_room["dimensions"]

            # Only consider if on the same floor
            if abs(adj_z - z) > 0.1:
                continue

            # Try positions in each direction
            positions = [
                (adj_x + adj_w, adj_y, z),  # Right
                (adj_x - room.width, adj_y, z),  # Left
                (adj_x, adj_y + adj_l, z),  # Behind
                (adj_x, adj_y - room.length, z),  # In front
            ]

            for pos in positions:
                x, y, z_pos = pos
                if (
                    x >= 0
                    and y >= 0
                    and x + room.width <= self.width
                    and y + room.length <= self.length
                    and self._is_valid_position(
                        x, y, z_pos, room.width, room.length, room.height
                    )
                ):
                    return pos

        return None

    def _find_perimeter_position(
        self, room: Room, floor: int
    ) -> Optional[Tuple[float, float, float]]:
        """Find position on building perimeter for rooms needing exterior access."""
        z = floor * self.floor_height

        # Create perimeter positions
        perimeter_positions = []

        # Front edge (y=0)
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            perimeter_positions.append((x, 0, z))

        # Back edge
        back_y = self.length - room.length
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            perimeter_positions.append((x, back_y, z))

        # Left edge
        for y in range(
            0, int(self.length - room.length) + 1, int(self.structural_grid[1])
        ):
            perimeter_positions.append((0, y, z))

        # Right edge
        right_x = self.width - room.width
        for y in range(
            0, int(self.length - room.length) + 1, int(self.structural_grid[1])
        ):
            perimeter_positions.append((right_x, y, z))

        # Shuffle positions to avoid predictable patterns
        random.shuffle(perimeter_positions)

        # Check each position
        for pos in perimeter_positions:
            x, y, z_pos = pos
            if self._is_valid_position(
                x, y, z_pos, room.width, room.length, room.height
            ):
                return pos

        return None

    def _find_position_on_floor(
        self, room: Room, floor: int
    ) -> Optional[Tuple[float, float, float]]:
        """Find any valid position on a specific floor."""
        z = floor * self.floor_height

        # Try structural grid positions
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

        # If structural grid fails, try finer grid
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
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

    def _is_valid_position(
        self, x: float, y: float, z: float, width: float, length: float, height: float
    ) -> bool:
        """Check if a position is valid for room placement."""
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
