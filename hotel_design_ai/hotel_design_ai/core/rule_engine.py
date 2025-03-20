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

        # Floor range
        self.min_floor = self.building_config.get("min_floor", -1)
        self.max_floor = self.building_config.get("max_floor", 3)
        self.floor_height = self.building_config.get("floor_height", 5.0)

        # Track room count by floor for balanced distribution
        self.rooms_per_floor = {
            floor: 0 for floor in range(self.min_floor, self.max_floor + 1)
        }

        # Room placement priorities (architectural knowledge)
        self.placement_priorities = {
            "entrance": 10,
            "lobby": 9,
            "vertical_circulation": 8,
            "restaurant": 7,
            "meeting_room": 6,
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

        # Floor preferences by room type
        self.floor_preferences = {
            "entrance": 0,
            "lobby": 0,
            "restaurant": 0,
            "kitchen": 0,
            "ballroom": 0,
            "meeting_room": 0,
            "pre_function": 0,
            "retail": 0,
            "guest_room": None,  # Distribute across floors 1-3
            "office": 1,
            "staff_area": 1,
            "service_area": -1,
            "back_of_house": -1,
            "maintenance": -1,
            "mechanical": -1,
            "parking": -1,
        }

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on architectural rules with enhanced placement.

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

        # Group rooms by floor for systematic placement
        rooms_by_floor = {}
        floor_height = self.building_config.get("floor_height", 5.0)
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)

        # Initialize all floors
        for floor in range(min_floor, max_floor + 1):
            rooms_by_floor[floor] = []

        # Group rooms by floor
        for room in rooms:
            if hasattr(room, "floor") and room.floor is not None:
                floor = room.floor
                if floor not in rooms_by_floor:
                    rooms_by_floor[floor] = []
                rooms_by_floor[floor].append(room)
            else:
                # Default floor assignment for rooms without floor
                if room.room_type in ["entrance", "lobby", "restaurant"]:
                    rooms_by_floor[0].append(room)  # Ground floor
                elif room.room_type in ["guest_room"]:
                    # Distribute guest rooms evenly across floors 1-3
                    guest_floors = list(range(1, max_floor + 1))
                    target_floor = guest_floors[
                        len(rooms_by_floor.get(1, [])) % len(guest_floors)
                    ]
                    rooms_by_floor[target_floor].append(room)
                elif room.room_type in ["mechanical", "parking", "maintenance"]:
                    rooms_by_floor[min_floor].append(room)  # Basement
                else:
                    # Default to ground floor
                    rooms_by_floor[0].append(room)

        # Count rooms per floor
        room_counts = {floor: len(rooms) for floor, rooms in rooms_by_floor.items()}
        guest_count = sum(1 for r in rooms if r.room_type == "guest_room")

        # Phase 1: Place critical elements first
        critical_room_types = ["entrance", "vertical_circulation", "lobby"]

        # Find critical rooms across all floors
        critical_rooms = []
        other_rooms = []

        for floor, floor_rooms in rooms_by_floor.items():
            for room in floor_rooms:
                if room.room_type in critical_room_types:
                    critical_rooms.append(room)
                else:
                    other_rooms.append(room)

        # Sort critical rooms (entrance first, then lobby, then vertical circulation)
        critical_rooms.sort(
            key=lambda r: (
                critical_room_types.index(r.room_type)
                if r.room_type in critical_room_types
                else len(critical_room_types)
            )
        )

        # Place critical rooms
        placed_critical_rooms = []
        for room in critical_rooms:
            placed = self.place_with_fallback(room)
            if placed:
                placed_critical_rooms.append(room.id)

        # Phase 2: Place remaining rooms floor by floor
        successfully_placed = len(placed_critical_rooms)
        failed_to_place = []

        # Start with basement and move up to ensure basement rooms get placed first
        for floor in range(min_floor, max_floor + 1):
            floor_z = floor * floor_height
            floor_rooms = rooms_by_floor.get(floor, [])

            if not floor_rooms:
                continue

            # Filter out already placed critical rooms
            floor_rooms = [r for r in floor_rooms if r.id not in placed_critical_rooms]

            # Sort remaining rooms for this floor by architectural priority
            sorted_rooms = sorted(
                floor_rooms,
                key=lambda r: self.placement_priorities.get(r.room_type, 0),
                reverse=True,
            )

            # Place each non-critical room on this floor
            floor_success = 0
            floor_fail = 0

            for room in sorted_rooms:
                placed = self.place_with_fallback(room)
                if placed:
                    floor_success += 1
                    successfully_placed += 1
                else:
                    floor_fail += 1
                    failed_to_place.append(f"{room.room_type} (id:{room.id})")

        # Calculate actual floor distribution after placement
        floor_distribution = {}
        floor_areas = {}

        for room_id, room_data in self.spatial_grid.rooms.items():
            # Access the z-coordinate from the position tuple
            position = room_data.get("position", (0, 0, 0))
            z = position[2]  # Get z from position tuple

            floor = int(z / floor_height)
            width, length, _ = room_data.get("dimensions", (0, 0, 0))
            area = width * length

            if floor not in floor_distribution:
                floor_distribution[floor] = 0
                floor_areas[floor] = 0

            floor_distribution[floor] += 1
            floor_areas[floor] += area

        # Calculate space utilization
        space_util = self.spatial_grid.calculate_space_utilization() * 100

        return self.spatial_grid

    def place_with_fallback(self, room: Room) -> bool:
        """
        Place a room on its assigned floor, with fallback to ANY floor if necessary.
        Special handling for vertical circulation elements.

        Args:
            room: Room object to place

        Returns:
            bool: True if placed successfully, False otherwise
        """
        # Get building config
        floor_height = self.building_config.get("floor_height", 5.0)
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)

        # Special handling for vertical circulation elements
        if room.room_type == "vertical_circulation":
            return self._place_vertical_circulation(
                room, min_floor, max_floor, floor_height
            )

        # Determine assigned floor (if any)
        assigned_floor = getattr(room, "floor", None)

        # Step 1: Try the assigned floor first (if specified)
        if assigned_floor is not None:
            # Calculate z-coordinate from floor number, handling negative floors correctly
            z_pos = assigned_floor * floor_height
            # Find position on assigned floor
            position = self._find_best_position(room, {}, assigned_floor)

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
                    return True

        # Step 2: If assigned floor failed or no floor assigned, try ALL floors
        # Try each floor in order of preference
        floor_preference = []

        # Customize floor order based on room type
        if room.room_type in [
            "mechanical",
            "parking",
            "maintenance",
            "storage",
            "back_of_house",
        ]:
            # Basement-friendly rooms: try basement first, then upper floors
            floor_preference = (
                list(range(min_floor, 0)) + [0] + list(range(1, max_floor + 1))
            )
        elif room.room_type in ["guest_room", "office", "staff_area"]:
            # Private rooms: try upper floors first, then ground, avoid basement
            floor_preference = list(range(1, max_floor + 1)) + [0]
        elif room.room_type in ["entrance", "lobby", "restaurant", "retail"]:
            # Public-facing: only try ground floor
            floor_preference = [0]
        else:
            # General case: try ground first, then upper, then basement
            floor_preference = (
                [0] + list(range(1, max_floor + 1)) + list(range(min_floor, 0))
            )

        # Try each floor
        for floor in floor_preference:
            position = self._find_best_position(room, {}, floor)

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
                    return True

        return False

    def _find_best_position(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]], target_floor: int
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find the best position for a room based on architectural rules.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type
            target_floor: Target floor to place the room on

        Returns:
            Optional[Tuple[float, float, float]]: Best (x, y, z) position or None if no valid position
        """
        # Convert floor to z coordinate - ensure negative floors become negative z coordinates
        target_z = target_floor * self.floor_height

        # Special handling based on room type
        if room.room_type == "lobby":
            return self._place_lobby(room, placed_rooms_by_type, target_z)
        elif room.room_type == "guest_room":
            return self._place_guest_room(room, placed_rooms_by_type, target_z)
        else:
            # Default placement logic for other room types
            return self._place_general_room(room, placed_rooms_by_type, target_z)

    def _is_valid_position(
        self, x: float, y: float, z: float, width: float, length: float, height: float
    ) -> bool:
        """
        Check if a position is valid for room placement

        Args:
            x, y, z: Position coordinates
            width, length, height: Room dimensions

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Check if position is within building bounds for x and y
        if x < 0 or y < 0 or x + width > self.width or y + length > self.length:
            return False

        # Calculate appropriate floor bounds
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)
        floor_height = self.building_config.get("floor_height", 5.0)

        # Calculate z bounds based on floor range, properly handling negative floors
        min_z = min_floor * floor_height
        max_z = (max_floor + 1) * floor_height  # +1 for the ceiling of the top floor

        # Check if z is within allowed floor range
        if z < min_z or z + height > max_z:
            return False

        # Convert to grid coordinates (for collision check)
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)

        # Handle negative z coordinates for basement properly
        if z < 0:
            # Make sure grid_z is negative too
            grid_z = int(z / self.grid_size)  # This might already be negative
            if grid_z >= 0:  # Extra safety check
                grid_z = -1  # Force it to be in basement grid cells

        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

        # Check bounds again after converting to grid coordinates
        if (
            grid_x < 0
            or grid_y < 0
            or grid_x + grid_width > self.spatial_grid.width_cells
            or grid_y + grid_length > self.spatial_grid.length_cells
            or grid_z + grid_height > self.spatial_grid.height_cells
        ):
            return False

        # Handle special case for basement: ensure grid_z is negative and within bounds
        if grid_z < 0 and abs(grid_z) > self.spatial_grid.height_cells:
            return False

        # Check for collisions with existing rooms
        try:
            # Get the region of the grid where the room would be placed
            target_region = self.spatial_grid.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # If any cell in this region is non-zero, there's a collision
            return np.all(target_region == 0)
        except IndexError:
            # If there's an index error, the room would be out of bounds
            return False

    def _find_valid_position_on_floor(
        self, room: Room, z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Find a valid position on a specific floor"""
        grid_step_x = self.structural_grid[0]
        grid_step_y = self.structural_grid[1]

        # Try positions aligned with structural grid
        # Start from the center and spiral outward for better space utilization
        center_x = self.width / 2
        center_y = self.length / 2

        # Generate positions in a spiral pattern starting from center
        positions = []
        max_radius = max(self.width, self.length) / 2

        for radius in range(0, int(max_radius) + 1, int(min(grid_step_x, grid_step_y))):
            # Add positions in a rough "circle" at this radius
            for angle in range(0, 360, 45):  # 45 degree increments
                # Convert polar to cartesian coordinates
                angle_rad = angle * 3.14159 / 180
                x = center_x + radius * np.cos(angle_rad)
                y = center_y + radius * np.sin(angle_rad)

                # Snap to grid
                x = round(x / grid_step_x) * grid_step_x
                y = round(y / grid_step_y) * grid_step_y

                # Ensure within bounds
                x = max(0, min(x, self.width - room.width))
                y = max(0, min(y, self.length - room.length))

                positions.append((x, y))

        # Remove duplicates
        positions = list(set(positions))

        # Try each position
        for x, y in positions:
            if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                return (x, y, z)

        # If structural grid positions don't work, fall back to a finer grid
        # Try positions at a finer grain
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # No valid position found on this floor
        return None

    def _place_vertical_circulation(
        self, room: Room, min_floor: int, max_floor: int, floor_height: float
    ) -> bool:
        """
        Special placement logic for vertical circulation elements (elevators, stairs)
        that need to span multiple floors from basement to top floor.

        Args:
            room: Vertical circulation room to place
            min_floor: Minimum floor (usually basement)
            max_floor: Maximum floor
            floor_height: Height of each floor

        Returns:
            bool: True if placed successfully, False otherwise
        """
        # Calculate full height from min to max floor
        total_floors = max_floor - min_floor + 1
        total_height = total_floors * floor_height

        # Starting z-coordinate from the lowest floor (usually basement)
        start_z = min_floor * floor_height

        # Update the metadata to show this is a core element
        if not room.metadata:
            room.metadata = {}
        room.metadata["is_core"] = True
        room.metadata["spans_floors"] = list(range(min_floor, max_floor + 1))

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
            if self._is_valid_for_vertical_circulation(
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

        # If all attempts fail, try to place it without overlapping with parking
        print("Falling back to non-overlapping vertical circulation placement")
        for x, y in positions:
            # Check if position is valid without allowing overlaps
            valid = self._is_valid_position(
                x, y, start_z, room.width, room.length, total_height
            )
            if valid:
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
                        f"Successfully placed non-overlapping vertical circulation at ({x}, {y}, {start_z})"
                    )
                    return True

        # If completely failed, place at a fixed location as a last resort
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

    def _is_valid_for_vertical_circulation(
        self, x: float, y: float, z: float, width: float, length: float, height: float
    ) -> bool:
        """
        Check if a position is valid for vertical circulation, with special rules:
        - Can overlap with parking areas
        - Cannot overlap with other rooms

        Args:
            x, y, z: Position coordinates
            width, length, height: Room dimensions

        Returns:
            bool: True if position is valid for vertical circulation
        """
        # Check building bounds
        if x < 0 or y < 0 or x + width > self.width or y + length > self.length:
            return False

        # Check if the position is within the valid height range
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
            or grid_z < 0
            or grid_x + grid_width > self.spatial_grid.width_cells
            or grid_y + grid_length > self.spatial_grid.length_cells
            or grid_z + grid_height > self.spatial_grid.height_cells
        ):
            return False

        # Check for collisions with non-parking rooms
        # Get the grid region where we would place the vertical circulation
        try:
            region = self.spatial_grid.grid[
                grid_x : grid_x + grid_width,
                grid_y : grid_y + grid_length,
                grid_z : grid_z + grid_height,
            ]

            # Find any non-zero values (existing rooms)
            room_ids = set(np.unique(region)) - {0}

            # If there are no rooms, position is valid
            if not room_ids:
                return True

            # Check if all overlapping rooms are parking areas
            for room_id in room_ids:
                if room_id in self.spatial_grid.rooms:
                    room_type = self.spatial_grid.rooms[room_id]["type"]
                    # Only allow overlap with parking areas
                    if room_type != "parking":
                        return False

            # If we get here, all overlaps are with parking areas, which is allowed
            return True

        except IndexError:
            # Grid index out of bounds
            return False

    def _place_lobby(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]], target_z: float
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
                entrance_x, lobby_y, target_z, room.width, room.length, room.height
            ):
                return (entrance_x, lobby_y, target_z)

            # Try positioning beside the entrance
            lobby_x = entrance_x + entrance_w
            if self._is_valid_position(
                lobby_x, entrance_y, target_z, room.width, room.length, room.height
            ):
                return (lobby_x, entrance_y, target_z)

        # If no entrance or no valid position near entrance, place centrally
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2

        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        grid_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]

        if self._is_valid_position(
            grid_x, grid_y, target_z, room.width, room.length, room.height
        ):
            return (grid_x, grid_y, target_z)

        # Fallback to grid search on this floor
        return self._find_valid_position_on_floor(room, target_z)

    def _place_guest_room(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]], target_z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for guest rooms on the specified floor"""
        # Try to place along the perimeter for exterior access
        perimeter_positions = []

        # Front edge
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            perimeter_positions.append((x, 0, target_z))

        # Back edge
        back_y = self.length - room.length
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            perimeter_positions.append((x, back_y, target_z))

        # Left edge
        for y in range(
            0, int(self.length - room.length) + 1, int(self.structural_grid[1])
        ):
            perimeter_positions.append((0, y, target_z))

        # Right edge
        right_x = self.width - room.width
        for y in range(
            0, int(self.length - room.length) + 1, int(self.structural_grid[1])
        ):
            perimeter_positions.append((right_x, y, target_z))

        # Shuffle to avoid always filling in the same pattern
        random.shuffle(perimeter_positions)

        for pos in perimeter_positions:
            x, y, z = pos
            if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                return pos

        # If perimeter is full, try any valid position on this floor
        return self._find_valid_position_on_floor(room, target_z)

    def _place_general_room(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]], target_z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Default placement logic for other room types"""
        # Check for adjacency preferences
        adjacent_to = self.adjacency_preferences.get(room.room_type, [])

        # Try to place adjacent to preferred room types
        for preferred_type in adjacent_to:
            if preferred_type in placed_rooms_by_type:
                # Get all rooms of this type
                for room_id in placed_rooms_by_type[preferred_type]:
                    existing_room = self.spatial_grid.rooms[room_id]

                    # Check if this room is on the same floor
                    existing_z = existing_room["position"][2]
                    existing_floor = int(existing_z / self.floor_height)
                    target_floor = int(target_z / self.floor_height)

                    if existing_floor == target_floor:
                        x, y, z = existing_room["position"]
                        w, l, h = existing_room["dimensions"]

                        # Try positions around this room
                        positions = [
                            (x + w, y, target_z),  # Right
                            (x, y + l, target_z),  # Behind
                            (x - room.width, y, target_z),  # Left
                            (x, y - room.length, target_z),  # In front
                        ]

                        for pos in positions:
                            test_x, test_y, test_z = pos
                            if self._is_valid_position(
                                test_x,
                                test_y,
                                test_z,
                                room.width,
                                room.length,
                                room.height,
                            ):
                                return pos

        # Check if exterior access is needed
        exterior_pref = self.exterior_preferences.get(room.room_type, 0)

        if exterior_pref > 0:
            # Try perimeter positions
            perimeter_positions = []

            # Front edge
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, 0, target_z))

            # Back edge
            back_y = self.length - room.length
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, back_y, target_z))

            # Left edge
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((0, y, target_z))

            # Right edge
            right_x = self.width - room.width
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((right_x, y, target_z))

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

        # Default to grid-based placement on specified floor
        return self._find_valid_position_on_floor(room, target_z)
