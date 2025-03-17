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

        """
    Modified generate_layout method that uses the enhanced placement strategies
    """

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

        # Show building configuration
        print(f"\nBuilding configuration:")
        print(
            f"  Width: {self.width:.1f}m, Length: {self.length:.1f}m, Height: {self.height:.1f}m"
        )
        print(
            f"  Floor Range: {self.building_config.get('min_floor', -1)} to {self.building_config.get('max_floor', 3)} (Height per floor: {self.building_config.get('floor_height', 5.0):.1f}m)"
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

        print("\nRoom distribution before placement:")
        for floor in sorted(room_counts.keys()):
            floor_name = "Basement" if floor < 0 else f"Floor {floor}"
            print(f"  {floor_name}: {room_counts[floor]} rooms")

        guest_count = sum(1 for r in rooms if r.room_type == "guest_room")
        print(f"  Guest rooms: {guest_count} rooms (to be distributed)")

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
            print(f"Placing critical room: {room.room_type}")
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

            print(f"\nPlacing {len(floor_rooms)} rooms on floor {floor}...")

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

            print(f"  Floor {floor}: {floor_success} placed, {floor_fail} failed")

        # Calculate actual floor distribution after placement
        floor_distribution = {}
        floor_areas = {}

        """
        Fixed version of the key access in the generate_layout method
        """

        # In generate_layout, replace the part that's causing the error with this:

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

        print(f"\nLayout generation complete:")
        print(
            f"  Successfully placed {successfully_placed} of {len(rooms)} rooms ({successfully_placed/len(rooms)*100:.1f}%)"
        )
        print(f"  Space utilization: {space_util:.1f}%")

        # Print actual floor distribution
        print("\nActual floor distribution after placement:")
        for floor in sorted(floor_distribution.keys()):
            floor_name = "Basement" if floor < 0 else f"Floor {floor}"
            print(
                f"  {floor_name}: {floor_distribution[floor]} rooms ({floor_areas[floor]:.1f} m²)"
            )

        if failed_to_place:
            print(f"\n  Failed to place {len(failed_to_place)} rooms:")
            room_type_counts = {}
            for item in failed_to_place:
                room_type = item.split(" ")[0]
                if room_type not in room_type_counts:
                    room_type_counts[room_type] = 0
                room_type_counts[room_type] += 1

            for room_type, count in sorted(
                room_type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    - {room_type}: {count} rooms")

        # Total area placed
        total_area_placed = sum(floor_areas.values())
        print(f"\nTotal area placed in layout: {total_area_placed:.1f} m²")

        return self.spatial_grid

    def _print_floor_utilization(self):
        """Print a summary of how many rooms were placed on each floor"""
        print("\nFloor utilization summary:")
        total_rooms = sum(self.rooms_per_floor.values())

        # Calculate floor areas as well
        floor_areas = {}
        total_area = 0

        # Calculate areas by floor
        for room_id, room_data in self.spatial_grid.rooms.items():
            z = room_data["position"][2]
            floor = int(z / self.floor_height)

            width, length, _ = room_data["dimensions"]
            area = width * length

            if floor not in floor_areas:
                floor_areas[floor] = 0

            floor_areas[floor] += area
            total_area += area

        # Print detailed floor stats
        for floor in sorted(self.rooms_per_floor.keys()):
            count = self.rooms_per_floor[floor]
            area = floor_areas.get(floor, 0)
            room_percentage = (count / total_rooms * 100) if total_rooms > 0 else 0
            area_percentage = (area / total_area * 100) if total_area > 0 else 0

            floor_name = "Basement" if floor < 0 else f"Floor {floor}"
            print(
                f"  {floor_name}: {count} rooms ({room_percentage:.1f}%), {area:.1f}m² ({area_percentage:.1f}%)"
            )

        # Print total area
        print(f"  Total area: {total_area:.1f}m²")

    def _get_preferred_floor(self, room: Room) -> Optional[int]:
        """Get the preferred floor for a room based on room type"""
        # First check if room has a specified floor
        if hasattr(room, "floor") and room.floor is not None:
            return room.floor

        # Otherwise use the type-based preference
        return self.floor_preferences.get(room.room_type)

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
        # Convert floor to z coordinate
        target_z = target_floor * self.floor_height

        # Special handling based on room type
        if room.room_type == "lobby":
            return self._place_lobby(room, placed_rooms_by_type, target_z)
        elif room.room_type == "guest_room":
            return self._place_guest_room(room, placed_rooms_by_type, target_z)
        else:
            # Default placement logic for other room types
            return self._place_general_room(room, placed_rooms_by_type, target_z)

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
        """
        Find a position for vertical circulation that works across all floors.

        Args:
            room: Room representing vertical circulation
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Position for ground floor element
        """
        # Find a good central position first
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2

        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        grid_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]

        # Try center position - check if it works for all floors
        if self._can_place_across_floors(
            grid_x,
            grid_y,
            room.width,
            room.length,
            room.height,
            self.min_floor,
            self.max_floor,
            self.floor_height,
        ):
            return (grid_x, grid_y, 0)  # Return ground floor position

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
                    self.min_floor,
                    self.max_floor,
                    self.floor_height,
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
                self.min_floor,
                self.max_floor,
                self.floor_height,
            ):
                return pos

        # As a last resort, try grid search
        return self._find_vertical_circulation_position_on_grid(
            room, self.min_floor, self.max_floor, self.floor_height
        )

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
        # Guest rooms are typically:
        # 1. On upper floors
        # 2. Arranged in efficient layouts along corridors
        # 3. Have exterior access for natural light

        # Get the floor number for logging purposes
        target_floor = int(target_z / self.floor_height)
        print(f"    Finding position for guest room on floor {target_floor}...")

        # Check if this room prefers to be near vertical circulation
        near_circulation = False
        if "vertical_circulation" in placed_rooms_by_type:
            near_circulation = True

        if near_circulation:
            # Get vertical circulation elements on this floor
            circ_positions = []

            for circ_id in placed_rooms_by_type["vertical_circulation"]:
                # Skip if the circulation element no longer exists (e.g., was removed)
                if circ_id not in self.spatial_grid.rooms:
                    continue

                circ_room = self.spatial_grid.rooms[circ_id]
                circ_z = circ_room["position"][2]
                circ_floor = int(circ_z / self.floor_height)

                # Only consider circulation on the same floor
                if circ_floor == target_floor:
                    circ_positions.append(circ_room["position"])

            # If we found vertical circulation on this floor
            if circ_positions:
                print(
                    f"    Found {len(circ_positions)} vertical circulation elements on floor {target_floor}"
                )
                # Try positions near circulation elements
                for circ_pos in circ_positions:
                    circ_x, circ_y, _ = circ_pos

                    # Try to place along a corridor extending from circulation
                    corridor_positions = []

                    # Horizontal corridor
                    for x in range(
                        0,
                        int(self.width - room.width) + 1,
                        int(self.structural_grid[0]),
                    ):
                        corridor_positions.append((x, circ_y, target_z))

                    # Vertical corridor
                    for y in range(
                        0,
                        int(self.length - room.length) + 1,
                        int(self.structural_grid[1]),
                    ):
                        corridor_positions.append((circ_x, y, target_z))

                    # Try corridor positions
                    for pos in corridor_positions:
                        x, y, z = pos
                        if self._is_valid_position(
                            x, y, z, room.width, room.length, room.height
                        ):
                            print(
                                f"    Found corridor position at ({x:.1f}, {y:.1f}, {z:.1f})"
                            )
                            return pos

        # Try to place along the perimeter for exterior access
        print(f"    Trying perimeter positions for exterior access...")
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
                print(f"    Found perimeter position at ({x:.1f}, {y:.1f}, {z:.1f})")
                return pos

        # If perimeter is full, try any valid position on this floor
        print(f"    Trying any valid position on floor {target_floor}...")
        position = self._find_valid_position_on_floor(room, target_z)
        if position:
            x, y, z = position
            print(f"    Found interior position at ({x:.1f}, {y:.1f}, {z:.1f})")
        else:
            print(f"    FAILED to find any valid position on floor {target_floor}")
        return position

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

    """
    Update the _is_valid_position method to use the existing SpatialGrid methods
    instead of the new check_collision method
    """

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
        # Check if position is within building bounds
        # NOTE: Allow negative z for basement floors
        if x < 0 or y < 0 or x + width > self.width or y + length > self.length:
            return False

        # Calculate appropriate floor bounds
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)
        floor_height = self.building_config.get("floor_height", 5.0)

        # Calculate z bounds based on floor range
        min_z = min_floor * floor_height
        max_z = (max_floor + 1) * floor_height  # +1 for the ceiling of the top floor

        # Check if z is within allowed floor range
        if z < min_z or z + height > max_z:
            return False

        # Convert to grid coordinates (for collision check)
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)

        # Check bounds again after converting to grid coordinates
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

        """
        Simplified version of place_with_fallback to avoid dependency on missing methods
        """

    def place_with_fallback(self, room: Room) -> bool:
        """
        Place a room on its assigned floor, with fallback to ANY floor if necessary.

        Args:
            room: Room object to place

        Returns:
            bool: True if placed successfully, False otherwise
        """
        # Get building config
        floor_height = self.building_config.get("floor_height", 5.0)
        min_floor = self.building_config.get("min_floor", -1)
        max_floor = self.building_config.get("max_floor", 3)

        # Determine assigned floor (if any)
        assigned_floor = getattr(room, "floor", None)

        # Step 1: Try the assigned floor first (if specified)
        if assigned_floor is not None:
            z_pos = assigned_floor * floor_height
            # Use the correct signature - include the target_floor parameter
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
                    if assigned_floor != int(z / floor_height):
                        print(
                            f"  Note: {room.room_type} assigned to floor {assigned_floor} "
                            f"placed on floor {int(z / floor_height)} instead"
                        )
                    return True

        # Step 2: If assigned floor failed or no floor assigned, try ALL floors
        print(
            f"  Could not place {room.room_type} on assigned floor {assigned_floor}, trying ALL floors..."
        )

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
            # Use the correct signature - include the target_floor parameter
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
                    placed_floor = int(z / floor_height)
                    if assigned_floor is not None and assigned_floor != placed_floor:
                        print(
                            f"  Success! {room.room_type} reassigned from floor {assigned_floor} "
                            f"to floor {placed_floor}"
                        )
                    else:
                        print(
                            f"  Success! {room.room_type} placed on floor {placed_floor}"
                        )
                    return True

        # If still not placed, provide simple feedback
        print(f"  Failed to place {room.room_type} on any floor")

        # If room is large, suggest splitting it
        if room.width * room.length > 200:
            print(
                f"  Large room ({room.width:.1f}m × {room.length:.1f}m = {room.width * room.length:.1f} m²)"
            )
            print(
                f"  Consider splitting into smaller rooms in your program requirements"
            )

        return False

    def _find_valid_position_on_floor(
        self, room: Room, z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Find a valid position on a specific floor"""
        floor = int(z / self.floor_height)
        grid_step_x = self.structural_grid[0]
        grid_step_y = self.structural_grid[1]

        # Try structural grid positions first
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
        print(
            f"    No position found on structural grid for floor {floor}, trying finer grid..."
        )

        # Try positions at a finer grain
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # No valid position found on this floor
        print(
            f"    WARNING: Could not find any valid position on floor {floor} for {room.room_type}"
        )
        return None

    """
    Enhanced room placement algorithm for the rule_engine.py file
    to improve placement success rates and better handle difficult rooms
    """

    def _find_valid_position_on_grid(
        self, room: Room, z: float = 0
    ) -> Optional[Tuple[float, float, float]]:
        """Find a valid position on the structural grid with enhanced placement logic"""
        grid_step_x = self.structural_grid[0]
        grid_step_y = self.structural_grid[1]

        # Try original orientation first
        position = self._try_positions_on_floor(room, z, grid_step_x, grid_step_y)
        if position:
            return position

        # Try rotating the room if it's not square
        if abs(room.width - room.length) > 0.1:
            # Swap width and length
            orig_width, orig_length = room.width, room.length
            room.width, room.length = orig_length, orig_width

            # Try the rotated orientation
            position = self._try_positions_on_floor(room, z, grid_step_x, grid_step_y)

            # Restore original dimensions
            room.width, room.length = orig_width, orig_length

            if position:
                # Log that rotation worked
                print(f"    Succeeded in placing {room.room_type} by rotating it")
                return position

        # If the room is very large (> 300 m²), try subdividing it
        area = room.width * room.length
        if area > 300 and (room.width > 15 or room.length > 15):
            print(
                f"    Large room ({room.room_type}, {area:.1f} m²) - trying flexible dimensions"
            )

            # Try different aspect ratios while maintaining area
            aspect_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

            # Store original dimensions
            orig_width, orig_length = room.width, room.length

            for ratio in aspect_ratios:
                # Calculate new dimensions based on aspect ratio
                new_width = (area * ratio) ** 0.5
                new_length = area / new_width

                # Skip if new dimensions are too small
                min_width = getattr(room, "min_width", 3.0)
                min_length = getattr(room, "min_length", 3.0)

                if new_width < min_width or new_length < min_length:
                    continue

                # Update room dimensions temporarily
                room.width, room.length = new_width, new_length

                # Try this shape
                position = self._try_positions_on_floor(
                    room, z, grid_step_x, grid_step_y
                )

                # Also try rotated
                if not position and abs(new_width - new_length) > 0.1:
                    room.width, room.length = new_length, new_width
                    position = self._try_positions_on_floor(
                        room, z, grid_step_x, grid_step_y
                    )
                    room.width, room.length = new_width, new_length

                # Restore original dimensions
                room.width, room.length = orig_width, orig_length

                if position:
                    print(
                        f"    Succeeded with adjusted dimensions ({new_width:.1f}m × {new_length:.1f}m)"
                    )
                    return position

        # If z=0 doesn't work and this is not a ground-floor-specific room type,
        # try other floors
        if z == 0 and room.room_type not in ["entrance", "lobby"]:
            # Get floor range from building configuration
            min_floor = self.building_config.get("min_floor", -1)
            max_floor = self.building_config.get("max_floor", 3)
            floor_height = self.building_config.get("floor_height", 5.0)

            # Try all other floors
            for floor in range(max_floor, min_floor - 1, -1):
                if floor != 0:  # Already tried ground floor
                    new_z = floor * floor_height
                    position = self._try_positions_on_floor(
                        room, new_z, grid_step_x, grid_step_y
                    )
                    if position:
                        print(
                            f"    Placed {room.room_type} on floor {floor} instead of ground floor"
                        )
                        return position

        # If structural grid positions don't work, try finer grid
        position = self._try_positions_with_finer_grid(room, z)
        if position:
            return position

        # If the room has fixed floor constraints, print more detailed warning
        if hasattr(room, "floor") and room.floor is not None:
            floor_num = room.floor
            print(
                f"    WARNING: Could not place {room.room_type} ({room.width:.1f}m × {room.length:.1f}m, {room.width * room.length:.1f} m²) fixed on floor {floor_num}"
            )
        else:
            print(
                f"    WARNING: Could not place {room.room_type} ({room.width:.1f}m × {room.length:.1f}m, {room.width * room.length:.1f} m²) on any floor"
            )

        return None

    """
    Add the missing debug_placement_failure method to the RuleEngine class
    """

    def debug_placement_failure(self, room, z_floor):
        """Print detailed debug info for placement failures"""
        floor_height = self.building_config.get("floor_height", 5.0)
        floor_num = int(z_floor / floor_height)
        room_type = room.room_type
        room_size = room.width * room.length

        print(f"\n=== PLACEMENT FAILURE DEBUG ===")
        print(f"Failed to place {room_type} (id:{room.id}) on floor {floor_num}")
        print(
            f"Room dimensions: {room.width:.1f}m × {room.length:.1f}m = {room_size:.1f} m²"
        )

        # Check building bounds
        if room.width > self.width or room.length > self.length:
            print(f"ISSUE: Room too large for building!")
            print(f"Building size: {self.width}m × {self.length}m")
            return

        # Simplify the debug output since we don't have all the needed methods
        print(f"Building dimensions: {self.width}m × {self.length}m × {self.height}m")
        print(f"Floor {floor_num} coordinates: z = {z_floor}")

        # Additional checks for specific floor issues
        if floor_num < 0:  # Basement floors
            print(f"Basement floor check:")
            test_pos = (0, 0, z_floor)
            is_valid = self._is_valid_position(
                test_pos[0], test_pos[1], test_pos[2], 1, 1, 1
            )
            print(
                f"  Test position at ({test_pos[0]}, {test_pos[1]}, {test_pos[2]}): {'VALID' if is_valid else 'INVALID'}"
            )

        print("=== END DEBUG ===\n")

    def _try_positions_on_floor(
        self, room: Room, z: float, grid_step_x: float, grid_step_y: float
    ) -> Optional[Tuple[float, float, float]]:
        """Try positions aligned with structural grid on a specific floor using multiple strategies"""
        # Try multiple placement strategies in order

        # 1. Try from corners (often more efficient for space utilization)
        corner_positions = [
            (0, 0, z),  # Bottom-left
            (self.width - room.width, 0, z),  # Bottom-right
            (0, self.length - room.length, z),  # Top-left
            (self.width - room.width, self.length - room.length, z),  # Top-right
        ]

        for pos in corner_positions:
            x, y, z_pos = pos
            if self._is_valid_position(
                x, y, z_pos, room.width, room.length, room.height
            ):
                return pos

        # 2. Try along edges (good for rooms needing exterior access)
        edge_positions = []

        # Bottom edge
        for x in range(0, int(self.width - room.width) + 1, int(grid_step_x)):
            edge_positions.append((x, 0, z))

        # Top edge
        y_top = self.length - room.length
        for x in range(0, int(self.width - room.width) + 1, int(grid_step_x)):
            edge_positions.append((x, y_top, z))

        # Left edge
        for y in range(0, int(self.length - room.length) + 1, int(grid_step_y)):
            edge_positions.append((0, y, z))

        # Right edge
        x_right = self.width - room.width
        for y in range(0, int(self.length - room.length) + 1, int(grid_step_y)):
            edge_positions.append((x_right, y, z))

        for pos in edge_positions:
            x, y, z_pos = pos
            if self._is_valid_position(
                x, y, z_pos, room.width, room.length, room.height
            ):
                return pos

        # 3. Try center outward spiral pattern
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2

        # Round to grid
        center_grid_x = round(center_x / grid_step_x) * grid_step_x
        center_grid_y = round(center_y / grid_step_y) * grid_step_y

        # Add center position
        center_pos = (center_grid_x, center_grid_y, z)
        if self._is_valid_position(
            center_pos[0],
            center_pos[1],
            center_pos[2],
            room.width,
            room.length,
            room.height,
        ):
            return center_pos

        # Create spiral pattern from center
        spiral_positions = []
        max_steps = (
            max(int(self.width / grid_step_x), int(self.length / grid_step_y)) // 2
        )

        for step in range(1, max_steps + 1):
            # Top and bottom rows
            for x in range(-step, step + 1):
                x_pos = center_grid_x + x * grid_step_x
                y_pos1 = center_grid_y - step * grid_step_y  # Top row
                y_pos2 = center_grid_y + step * grid_step_y  # Bottom row
                spiral_positions.append((x_pos, y_pos1, z))
                spiral_positions.append((x_pos, y_pos2, z))

            # Left and right columns (excluding corners already added)
            for y in range(-step + 1, step):
                x_pos1 = center_grid_x - step * grid_step_x  # Left column
                x_pos2 = center_grid_x + step * grid_step_x  # Right column
                y_pos = center_grid_y + y * grid_step_y
                spiral_positions.append((x_pos1, y_pos, z))
                spiral_positions.append((x_pos2, y_pos, z))

        # Filter positions to ensure they're inside building bounds
        valid_spiral_positions = []
        for pos in spiral_positions:
            x, y, z_pos = pos
            if (
                0 <= x <= self.width - room.width
                and 0 <= y <= self.length - room.length
            ):
                valid_spiral_positions.append(pos)

        # Try all valid positions
        for pos in valid_spiral_positions:
            x, y, z_pos = pos
            if self._is_valid_position(
                x, y, z_pos, room.width, room.length, room.height
            ):
                return pos

        # 4. Full grid search as last resort
        for x in range(0, int(self.width - room.width) + 1, int(grid_step_x)):
            for y in range(0, int(self.length - room.length) + 1, int(grid_step_y)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        return None

    def _try_positions_with_finer_grid(
        self, room: Room, z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Try positions with a finer grid when structural grid fails"""
        floor_num = int(z / self.building_config.get("floor_height", 5.0))
        print(
            f"    No position found on structural grid for floor {floor_num}, trying finer grid..."
        )

        # Try positions at half the structural grid
        half_grid_x = self.structural_grid[0] / 2
        half_grid_y = self.structural_grid[1] / 2

        for x in range(0, int(self.width - room.width) + 1, int(half_grid_x)):
            for y in range(0, int(self.length - room.length) + 1, int(half_grid_y)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # Try at quarter grid size
        quarter_grid_x = self.structural_grid[0] / 4
        quarter_grid_y = self.structural_grid[1] / 4

        for x in range(0, int(self.width - room.width) + 1, int(quarter_grid_x)):
            for y in range(0, int(self.length - room.length) + 1, int(quarter_grid_y)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        # Try at grid_size intervals as last resort
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    return (x, y, z)

        return None
