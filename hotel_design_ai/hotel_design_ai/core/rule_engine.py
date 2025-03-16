from typing import Dict, List, Tuple, Optional, Set, Any
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

        # Room placement priorities (architectural knowledge)
        self.placement_priorities = {
            "entrance": 10,
            "lobby": 9,
            "vertical_circulation": 8,
            "restaurant": 7,
            "meeting_rooms": 6,
            "guest_room": 5,
            "service_area": 4,
            "back_of_house": 3,
        }

        # Adjacency preferences (architectural knowledge)
        self.adjacency_preferences = {
            "entrance": ["lobby", "vertical_circulation"],
            "lobby": ["entrance", "restaurant", "vertical_circulation", "meeting_room"],
            "vertical_circulation": ["lobby", "guest_room", "service_area"],
            "restaurant": ["lobby", "kitchen"],
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
            "meeting_room": 0,  # No preference
        }

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on architectural rules.

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

        # Phase 1: Place critical elements (hard constraints)
        # First, identify and place critical rooms like entrance and vertical circulation
        critical_room_types = ["entrance", "vertical_circulation"]
        critical_rooms = [r for r in rooms if r.room_type in critical_room_types]
        other_rooms = [r for r in rooms if r.room_type not in critical_room_types]

        # Sort critical rooms (entrance first)
        critical_rooms.sort(key=lambda r: 0 if r.room_type == "entrance" else 1)

        # Track placed rooms by type for adjacency checks
        placed_rooms_by_type = {}

        # Place critical rooms first
        for room in critical_rooms:
            if room.room_type == "entrance":
                position = self._place_entrance(room)
            elif room.room_type == "vertical_circulation":
                position = self._place_vertical_circulation_across_floors(
                    room, placed_rooms_by_type
                )
            else:
                position = None

            if position:
                x, y, z = position
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
                    # Add to placed rooms tracking
                    if room.room_type not in placed_rooms_by_type:
                        placed_rooms_by_type[room.room_type] = []
                    placed_rooms_by_type[room.room_type].append(room.id)

        # Phase 2: Place remaining rooms with soft constraints
        # Sort remaining rooms by architectural priority
        sorted_rooms = sorted(
            other_rooms,
            key=lambda r: self.placement_priorities.get(r.room_type, 0),
            reverse=True,
        )

        # Place rooms according to priorities
        for room in sorted_rooms:
            best_position = self._find_best_position(room, placed_rooms_by_type)

            if best_position:
                x, y, z = best_position
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
                    # Add to placed rooms tracking
                    if room.room_type not in placed_rooms_by_type:
                        placed_rooms_by_type[room.room_type] = []
                    placed_rooms_by_type[room.room_type].append(room.id)

        return self.spatial_grid

    def _find_best_position(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find the best position for a room based on architectural rules.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Best (x, y, z) position or None if no valid position
        """
        # Special handling based on room type
        if room.room_type == "lobby":
            return self._place_lobby(room, placed_rooms_by_type)
        elif room.room_type == "guest_room":
            return self._place_guest_room(room, placed_rooms_by_type)
        else:
            # Default placement logic for other room types
            return self._place_general_room(room, placed_rooms_by_type)

    def _place_entrance(self, room: Room) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for entrance"""
        # By architectural convention, entrances are typically:
        # 1. On the ground floor (z=0)
        # 2. At the front of the building (commonly y=0 or y=max)
        # 3. Centered along the x-axis

        # Center along x-axis
        center_x = (self.width - room.width) / 2
        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]

        # Try front of building first (y=0)
        if self._is_valid_position(grid_x, 0, 0, room.width, room.length, room.height):
            return (grid_x, 0, 0)

        # Try back of building if front doesn't work
        back_y = self.length - room.length
        if self._is_valid_position(
            grid_x, back_y, 0, room.width, room.length, room.height
        ):
            return (grid_x, back_y, 0)

        # Try left side
        if self._is_valid_position(
            0, center_x, 0, room.width, room.length, room.height
        ):
            return (0, center_x, 0)

        # Try right side
        right_x = self.width - room.width
        if self._is_valid_position(
            right_x, center_x, 0, room.width, room.length, room.height
        ):
            return (right_x, center_x, 0)

        # If all preferred positions fail, try a more exhaustive search
        return self._find_valid_position_on_grid(room, z=0)

    def _place_vertical_circulation_across_floors(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find position and place vertical circulation on all floors"""
        # Find position that works on all floors
        position = self._find_vertical_circulation_position(room, placed_rooms_by_type)

        if not position:
            return None

        x, y, _ = position  # We only need x,y; z will vary by floor

        # Get floor range from building configuration
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)
        floor_height = self.building_config.get("floor_height", 5.0)

        # Place vertical circulation on ALL floors including ground floor
        for floor in range(min_floor, max_floor + 1):
            z = floor * floor_height

            # Create a unique ID for each floor's circulation element
            circ_id = room.id + (floor * 1000)  # Use offset to create unique IDs

            # Make sure we set the room_type to 'vertical_circulation' for all elements
            self.spatial_grid.place_room(
                room_id=circ_id,
                x=x,
                y=y,
                z=z,
                width=room.width,
                length=room.length,
                height=room.height,
                room_type="vertical_circulation",  # Ensure correct room type
                metadata={"floor": floor, "is_vertical_circulation": True},
            )

        # Return the ground floor position (for reference)
        return (x, y, 0)

    def _can_place_across_floors(
        self,
        x: float,
        y: float,
        width: float,
        length: float,
        height: float,
        min_floor: int,
        max_floor: int,
        floor_height: float,
    ) -> bool:
        """
        Check if a position works for placing vertical circulation on all floors.

        Args:
            x, y: Position coordinates
            width, length, height: Dimensions
            min_floor, max_floor: Floor range
            floor_height: Height of each floor

        Returns:
            bool: True if the position works for all floors
        """
        # Check position on each floor
        for floor in range(min_floor, max_floor + 1):
            z = floor * floor_height
            if not self._is_valid_position(x, y, z, width, length, height):
                return False

        return True

    def _find_vertical_circulation_position(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find a position that works across all floors"""
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)
        floor_height = self.building_config.get("floor_height", 5.0)

        # Try central position
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        grid_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]

        if self._can_place_across_floors(
            grid_x,
            grid_y,
            room.width,
            room.length,
            room.height,
            min_floor,
            max_floor,
            floor_height,
        ):
            return (grid_x, grid_y, 0)

        # Additional position checks...
        # [Rest of the positioning logic]
        # If lobby is placed, try positions adjacent to it
        if "lobby" in placed_rooms_by_type and placed_rooms_by_type["lobby"]:
            lobby_id = placed_rooms_by_type["lobby"][0]
            lobby = self.spatial_grid.rooms[lobby_id]

            lobby_x, lobby_y, _ = lobby["position"]
            lobby_w, lobby_l, _ = lobby["dimensions"]

            # Try positions around the lobby
            positions = [
                (lobby_x + lobby_w, lobby_y, 0),  # Right
                (lobby_x, lobby_y + lobby_l, 0),  # Behind
                (lobby_x - room.width, lobby_y, 0),  # Left
            ]

            for pos in positions:
                x, y, z = pos
                if self._can_place_across_floors(
                    x,
                    y,
                    room.width,
                    room.length,
                    room.height,
                    min_floor,
                    max_floor,
                    floor_height,
                ):
                    return pos

        # Try corner positions as fallback
        corner_positions = [
            (0, 0, 0),  # Bottom-left
            (self.width - room.width, 0, 0),  # Bottom-right
            (0, self.length - room.length, 0),  # Top-left
            (self.width - room.width, self.length - room.length, 0),  # Top-right
        ]

        for pos in corner_positions:
            x, y, z = pos
            if self._can_place_across_floors(
                x,
                y,
                room.width,
                room.length,
                room.height,
                min_floor,
                max_floor,
                floor_height,
            ):
                return pos

        # As a last resort, try grid search
        return self._find_vertical_circulation_position_on_grid(
            room, min_floor, max_floor, floor_height
        )
        return None

    def _find_vertical_circulation_position_on_grid(
        self, room: Room, min_floor: int, max_floor: int, floor_height: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a position for vertical circulation that works across all floors using grid search.

        Args:
            room: Room to place
            min_floor, max_floor: Floor range
            floor_height: Height of each floor

        Returns:
            Optional[Tuple[float, float, float]]: Valid position or None
        """
        # Try positions aligned with structural grid
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                if self._can_place_across_floors(
                    x,
                    y,
                    room.width,
                    room.length,
                    room.height,
                    min_floor,
                    max_floor,
                    floor_height,
                ):
                    return (x, y, 0)  # Return ground floor position

        # If structural grid fails, try arbitrary positions
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                if self._can_place_across_floors(
                    x,
                    y,
                    room.width,
                    room.length,
                    room.height,
                    min_floor,
                    max_floor,
                    floor_height,
                ):
                    return (x, y, 0)

        # No valid position found
        return None

    def _place_lobby(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for lobby"""
        # Lobbies should be:
        # 1. Adjacent to entrance
        # 2. On the ground floor
        # 3. Central to the building

        # Check if entrance has been placed
        if "entrance" in placed_rooms_by_type and placed_rooms_by_type["entrance"]:
            entrance_id = placed_rooms_by_type["entrance"][0]
            entrance = self.spatial_grid.rooms[entrance_id]

            # Try positions adjacent to the entrance
            entrance_x, entrance_y, _ = entrance["position"]
            entrance_w, entrance_l, _ = entrance["dimensions"]

            # Try positioning after the entrance (further into the building)
            lobby_y = entrance_y + entrance_l
            if self._is_valid_position(
                entrance_x, lobby_y, 0, room.width, room.length, room.height
            ):
                return (entrance_x, lobby_y, 0)

            # Try positioning beside the entrance
            lobby_x = entrance_x + entrance_w
            if self._is_valid_position(
                lobby_x, entrance_y, 0, room.width, room.length, room.height
            ):
                return (lobby_x, entrance_y, 0)

        # If no entrance or no valid position near entrance, place centrally
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2

        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        grid_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]

        if self._is_valid_position(
            grid_x, grid_y, 0, room.width, room.length, room.height
        ):
            return (grid_x, grid_y, 0)

        # Fallback to grid search
        return self._find_valid_position_on_grid(room, z=0)

    def _place_guest_room(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for guest rooms"""
        # Guest rooms are typically:
        # 1. On upper floors
        # 2. Arranged in efficient layouts along corridors
        # 3. Have exterior access for natural light

        # Determine which floor to place on (start from floor 1, not ground floor)
        floor_height = self.building_config.get("floor_height", 5.0)
        current_floor = 1  # Start from first floor

        # Calculate z coordinate for this floor
        z = current_floor * floor_height

        # Try to place along the perimeter for exterior access
        perimeter_positions = []

        # Front edge
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

        # Shuffle to avoid always filling in the same pattern
        random.shuffle(perimeter_positions)

        for pos in perimeter_positions:
            x, y, z = pos
            if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                return pos

        # If perimeter is full, try interior positions
        return self._find_valid_position_on_grid(room, z=z)

    def _place_general_room(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Default placement logic for other room types"""
        # Check if room has preferred floor
        preferred_floor = None
        if hasattr(room, "floor") and room.floor is not None:
            preferred_floor = room.floor
        else:
            # Assign floors based on room type
            room_type_floors = {
                "entrance": 0,
                "lobby": 0,
                "restaurant": 0,
                "kitchen": 0,
                "meeting_room": 0,
                "ballroom": 0,
                "guest_room": None,  # Will be distributed across floors 1-3
                "office": 1,
                "staff_area": 1,
                "back_of_house": -1,  # Basement
                "mechanical": -1,
                "maintenance": -1,
                "parking": -1,
                "storage": -1,
                "service_area": -1,
            }
            preferred_floor = room_type_floors.get(room.room_type)

            # Special handling for guest rooms - distribute across floors 1, 2, and 3
            if room.room_type == "guest_room":
                # Get count of existing guest rooms by floor
                floor_counts = {1: 0, 2: 0, 3: 0}
                for room_id in placed_rooms_by_type.get("guest_room", []):
                    existing_room = self.spatial_grid.rooms[room_id]
                    floor_height = self.building_config.get("floor_height", 5.0)
                    room_floor = int(existing_room["position"][2] / floor_height)
                    if room_floor in floor_counts:
                        floor_counts[room_floor] += 1

                # Find the floor with the fewest guest rooms
                if floor_counts:
                    min_count = min(floor_counts.values())
                    candidate_floors = [
                        f for f, count in floor_counts.items() if count == min_count
                    ]
                    preferred_floor = candidate_floors[0] if candidate_floors else 1
                else:
                    # Start with floor 1 if no guest rooms yet
                    preferred_floor = 1

        # If we have a preferred floor, try to place there first
        if preferred_floor is not None:
            floor_height = self.building_config.get("floor_height", 5.0)
            z = preferred_floor * floor_height
            position = self._find_valid_position_on_grid(room, z=z)
            if position:
                return position

            # For guest rooms, if preferred floor is full, try other floors
            if room.room_type == "guest_room":
                for floor in [1, 2, 3]:  # Try all guest room floors
                    if floor != preferred_floor:  # Skip the floor we already tried
                        z = floor * floor_height
                        position = self._find_valid_position_on_grid(room, z=z)
                        if position:
                            return position

        # Continue with the original logic for adjacency, etc.
        # Check for adjacency preferences
        adjacent_to = self.adjacency_preferences.get(room.room_type, [])

        # Try to place adjacent to preferred room types
        for preferred_type in adjacent_to:
            if preferred_type in placed_rooms_by_type:
                # Get all rooms of this type
                for room_id in placed_rooms_by_type[preferred_type]:
                    existing_room = self.spatial_grid.rooms[room_id]

                    x, y, z = existing_room["position"]
                    w, l, h = existing_room["dimensions"]

                    # Try positions around this room
                    positions = [
                        (x + w, y, z),  # Right
                        (x, y + l, z),  # Behind
                        (x - room.width, y, z),  # Left
                        (x, y - room.length, z),  # In front
                    ]

                    for pos in positions:
                        test_x, test_y, test_z = pos
                        if self._is_valid_position(
                            test_x, test_y, test_z, room.width, room.length, room.height
                        ):
                            return pos

        # Check if exterior access is needed
        exterior_pref = self.exterior_preferences.get(room.room_type, 0)

        if exterior_pref > 0:
            # Calculate floor to place on
            floor_height = self.building_config.get("floor_height", 5.0)
            current_floor = 0  # Default to ground floor

            # Calculate z coordinate for this floor
            z = current_floor * floor_height

            # Similar perimeter logic as for guest rooms
            perimeter_positions = []

            # Front edge
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

            random.shuffle(perimeter_positions)

            for pos in perimeter_positions:
                x, y, z = pos
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return pos

            # If exterior is required but no position found, return None
            if exterior_pref == 2:
                return None

        # Default to grid-based placement on appropriate floor
        if hasattr(room, "floor") and room.floor is not None:
            floor_height = self.building_config.get("floor_height", 5.0)
            z = room.floor * floor_height
            return self._find_valid_position_on_grid(room, z=z)
        else:
            return self._find_valid_position_on_grid(room)

    def _is_valid_position(
        self, x: float, y: float, z: float, width: float, length: float, height: float
    ) -> bool:
        """Check if a position is valid for room placement"""
        # Convert to grid coordinates
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

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

        # Check for collisions with existing rooms
        target_region = self.spatial_grid.grid[
            grid_x : grid_x + grid_width,
            grid_y : grid_y + grid_length,
            grid_z : grid_z + grid_height,
        ]

        return np.all(target_region == 0)

    def _find_valid_position_on_grid(
        self, room: Room, z: float = 0
    ) -> Optional[Tuple[float, float, float]]:
        """Find a valid position on the structural grid"""
        grid_step_x = self.structural_grid[0]
        grid_step_y = self.structural_grid[1]

        # Try all positions aligned with structural grid
        for x in range(0, int(self.width - room.width) + 1, int(grid_step_x)):
            for y in range(0, int(self.length - room.length) + 1, int(grid_step_y)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # If structural grid positions don't work, try arbitrary positions
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # If z=0 doesn't work and this is not a ground-floor-specific room type,
        # try other floors
        if z == 0 and room.room_type not in ["entrance", "lobby", "restaurant"]:
            # Get floor range from building configuration
            min_floor = self.building_config.get("min_floor", -1)
            max_floor = self.building_config.get("max_floor", 3)
            floor_height = self.building_config.get("floor_height", 5.0)

            for floor in range(min_floor, max_floor + 1):
                if floor != 0:  # Already tried ground floor
                    new_z = floor * floor_height
                    result = self._find_valid_position_on_grid(room, z=new_z)
                    if result:
                        return result

        # No valid position found
        return None
