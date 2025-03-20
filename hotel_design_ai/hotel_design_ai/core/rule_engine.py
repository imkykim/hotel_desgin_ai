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
        )

        # Floor range
        self.min_floor = self.building_config.get("min_floor", -1)
        self.max_floor = self.building_config.get("max_floor", 3)
        self.floor_height = self.building_config.get("floor_height", 5.0)

        # Room placement priorities (architectural knowledge)
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

        # Adjacency preferences (architectural knowledge)
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

        # Exterior access preferences (0 = no preference, 1 = preferred, 2 = required)
        self.exterior_preferences = {
            "entrance": 2,  # Required
            "lobby": 1,  # Preferred
            "restaurant": 1,  # Preferred
            "guest_room": 1,  # Preferred
            "vertical_circulation": 0,  # No preference
            "service_area": 0,  # No preference
            "back_of_house": 0,  # No preference
        }

        # Floor preferences by room type
        self.floor_preferences = {
            "entrance": 0,
            "lobby": 0,
            "restaurant": 0,
            "kitchen": 0,
            "meeting_room": 0,
            "guest_room": None,  # Distribute across floors 1-3
            "service_area": -1,
            "back_of_house": -1,
            "mechanical": -1,
            "parking": -1,
        }

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on architectural constraints.

        Args:
            rooms: List of Room objects to place

        Returns:
            SpatialGrid: The generated layout
        """
        # Clear the spatial grid
        self.spatial_grid = SpatialGrid(
            width=self.width,
            length=self.length,
            height=self.height,
            grid_size=self.grid_size,
        )

        # Sort rooms by architectural priority
        sorted_rooms = sorted(
            rooms,
            key=lambda r: self.placement_priorities.get(r.room_type, 0),
            reverse=True,
        )

        # Track placed rooms by type
        placed_rooms_by_type = {}

        # Place rooms in priority order
        for room in sorted_rooms:
            success = self.place_room_by_constraints(room, placed_rooms_by_type)

            if success:
                if room.room_type not in placed_rooms_by_type:
                    placed_rooms_by_type[room.room_type] = []
                placed_rooms_by_type[room.room_type].append(room.id)

        return self.spatial_grid

    def place_room_by_constraints(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Place a room according to architectural constraints.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        # First determine target floor
        floor = getattr(room, "floor", None)
        if floor is None:
            # Use default floor preference by room type
            floor = self.floor_preferences.get(room.room_type)

            # If no preference, use appropriate default
            if floor is None:
                if room.room_type == "guest_room":
                    # Distribute guest rooms across upper floors
                    floor = 1 + (
                        len(placed_rooms_by_type.get("guest_room", [])) % self.max_floor
                    )
                else:
                    # Default to ground floor
                    floor = 0

        z = floor * self.floor_height

        # Special handling for vertical circulation
        if room.room_type == "vertical_circulation":
            return self._place_vertical_circulation(room)

        # Special handling for entrance (must be on perimeter)
        if room.room_type == "entrance":
            return self._place_entrance(room, z)

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

            # If exterior is required but no position found, consider this a failure
            if exterior_pref == 2:
                return False

        # Fall back to any position on the target floor
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

        # If preferred floor failed, try alternate floors
        alternate_floors = self._get_alternate_floors(room.room_type, floor)
        for alt_floor in alternate_floors:
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
            min_floor: Minimum floor (usually basement)
            max_floor: Maximum floor
            floor_height: Height of each floor

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
                )

                if success:
                    print(
                        f"Successfully placed vertical circulation at ({x}, {y}, {start_z})"
                    )
                    return True

        # If all attempts fail, try a fixed position as a last resort
        print("Falling back to fixed position for vertical circulation")
        x = grid_x
        y = grid_y

        # Force placement even if it has to overlap
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
            force_placement=True,
        )

        if success:
            print(f"Forced placement of vertical circulation at ({x}, {y}, {start_z})")
            return True

        # If all fails, report failure
        print("Failed to place vertical circulation")
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

    def _get_alternate_floors(self, room_type: str, current_floor: int) -> List[int]:
        """Get alternative floors for a room type if the preferred floor is full."""
        if room_type in ["guest_room"]:
            # Guest rooms: try upper floors
            return [f for f in range(1, self.max_floor + 1) if f != current_floor]
        elif room_type in ["mechanical", "maintenance", "back_of_house"]:
            # Service spaces: try basement, then upper floors
            return [self.min_floor] + [
                f for f in range(0, self.max_floor + 1) if f != current_floor
            ]
        elif room_type in ["entrance", "lobby", "restaurant"]:
            # Public spaces: primarily ground floor only
            return [0]
        else:
            # Others: ground, then upper floors, then basement
            return [0] + [f for f in range(1, self.max_floor + 1)] + [self.min_floor]

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

        # Convert to grid coordinates
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

        # Handle negative z for basement
        if z < 0 and grid_z >= 0:
            grid_z = -1  # Force negative grid coordinate

        # Check grid bounds
        if (
            grid_x < 0
            or grid_y < 0
            or grid_x + grid_width > self.spatial_grid.width_cells
            or grid_y + grid_length > self.spatial_grid.length_cells
            or grid_z + grid_height > self.spatial_grid.height_cells
        ):
            return False

        # Special check for basement
        if grid_z < 0 and abs(grid_z) > self.spatial_grid.height_cells:
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

    def _is_valid_for_vertical_circulation(
        self, x: float, y: float, z: float, width: float, length: float, height: float
    ) -> bool:
        """Check if a position is valid for vertical circulation (can overlap with parking)."""
        # Check building bounds and floor range
        if x < 0 or y < 0 or x + width > self.width or y + length > self.length:
            return False

        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)
        floor_height = self.building_config.get("floor_height", 5.0)

        min_z = min_floor * floor_height
        max_z = (max_floor + 1) * floor_height

        if z < min_z or z + height > max_z:
            return False

        # Convert to grid coordinates
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

        # Check grid bounds
        if (
            grid_x < 0
            or grid_y < 0
            or grid_x + grid_width > self.spatial_grid.width_cells
            or grid_y + grid_length > self.spatial_grid.length_cells
            or grid_z + grid_height > self.spatial_grid.height_cells
        ):
            return False

        # Check for collisions with non-parking rooms
        try:
            # Get region
            region = self.spatial_grid.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # Get unique room IDs in the region
            room_ids = set(np.unique(region)) - {0}

            # If no rooms, position is valid
            if not room_ids:
                return True

            # Check if all rooms are parking
            for room_id in room_ids:
                if room_id in self.spatial_grid.rooms:
                    if self.spatial_grid.rooms[room_id]["type"] != "parking":
                        return False
                else:
                    return False

            # All overlaps are with parking areas
            return True

        except IndexError:
            return False  # Grid indices out of bounds
