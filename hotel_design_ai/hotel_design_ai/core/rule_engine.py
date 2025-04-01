"""
Rule-based layout generation engine that uses architectural principles to create hotel layouts.
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from collections import defaultdict
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

    def place_room_by_constraints(self, room, placed_rooms_by_type):
        """
        Place a room according to architectural constraints with priority on direct adjacency.
        Modified to create tight packing with no corridors.
        """
        if hasattr(room, "position") and room.position is not None:
            x, y, z = room.position
            print(
                f"Using fixed position for {room.name} (id={room.id}): {room.position}"
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
                # Add to placed rooms tracking
                if room.room_type not in placed_rooms_by_type:
                    placed_rooms_by_type[room.room_type] = []
                placed_rooms_by_type[room.room_type].append(room.id)
                return True
            else:
                print(
                    f"Warning: Could not place {room.name} at fixed position {room.position}"
                )
        # First adjust dimensions to align with grid
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

        # Try each preferred floor
        for floor in preferred_floors:
            # Calculate z coordinate for this floor
            z = floor * self.floor_height

            # MODIFIED: Prioritize adjacency-based placement even more strongly
            # Try to find position adjacent to preferred room types with maximized wall sharing
            if room.room_type in self.adjacency_preferences:
                position = self._find_position_with_max_adjacency(
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

            # If no preferred adjacencies or adjacency placement failed,
            # try to place adjacent to ANY existing room to maintain compactness
            position = self._find_position_adjacent_to_any_room(
                room, z, placed_rooms_by_type
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
                    room.width = original_width
                    room.length = original_length
                    return True

            # Check if room needs exterior access (lower priority than adjacency in this model)
            exterior_pref = self.exterior_preferences.get(room.room_type, 0)
            if exterior_pref > 0:
                # Try perimeter position but maintain compactness
                position = self._find_compact_perimeter_position(
                    room, floor, placed_rooms_by_type
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
                        room.width = original_width
                        room.length = original_length
                        return True

            # Try any position on this floor that's close to existing rooms
            position = self._find_compact_position_on_floor(
                room, floor, placed_rooms_by_type
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
                    room.width = original_width
                    room.length = original_length
                    return True

        # Restore original dimensions before returning
        room.width = original_width
        room.length = original_length
        return False

    def _find_position_with_max_adjacency(
        self,
        room: Room,
        room_type: str,
        placed_rooms_by_type: Dict[str, List[int]],
        z: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find position with the maximum shared wall surface with preferred adjacent rooms.

        Args:
            room: Room to place
            room_type: Type of room being placed
            placed_rooms_by_type: Dictionary of already placed rooms by type
            z: Z-coordinate (floor level)

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        best_position = None
        max_adjacency_score = -1

        # Get preferred adjacencies for this room type
        preferred_types = self.adjacency_preferences.get(room_type, [])

        # Get all placed rooms on this floor of preferred types
        candidate_rooms = []
        for adj_type in preferred_types:
            if adj_type in placed_rooms_by_type:
                for room_id in placed_rooms_by_type[adj_type]:
                    if room_id in self.spatial_grid.rooms:
                        room_data = self.spatial_grid.rooms[room_id]
                        room_z = room_data["position"][2]

                        # Check if on the same floor
                        if abs(room_z - z) < 0.1:
                            candidate_rooms.append(room_data)

        # If no candidates, try any room type for compactness
        if not candidate_rooms:
            for type_name, room_ids in placed_rooms_by_type.items():
                for room_id in room_ids:
                    if room_id in self.spatial_grid.rooms:
                        room_data = self.spatial_grid.rooms[room_id]
                        room_z = room_data["position"][2]

                        # Check if on the same floor
                        if abs(room_z - z) < 0.1:
                            candidate_rooms.append(room_data)

        # Try positions adjacent to each candidate room
        for adj_room in candidate_rooms:
            adj_x, adj_y, adj_z = adj_room["position"]
            adj_w, adj_l, adj_h = adj_room["dimensions"]

            # Try positions directly adjacent (sharing a wall)
            positions = [
                (adj_x + adj_w, adj_y, z),  # Right
                (adj_x - room.width, adj_y, z),  # Left
                (adj_x, adj_y + adj_l, z),  # Top
                (adj_x, adj_y - room.length, z),  # Bottom
            ]

            for x, y, pos_z in positions:
                # Skip if out of bounds
                if (
                    x < 0
                    or y < 0
                    or x + room.width > self.width
                    or y + room.length > self.length
                ):
                    continue

                # Calculate adjacency score (shared wall length)
                adjacency_score = self._calculate_adjacency_score(
                    x, y, room.width, room.length, placed_rooms_by_type
                )

                # Check position validity
                if adjacency_score > max_adjacency_score and self._is_valid_position(
                    x, y, pos_z, room.width, room.length, room.height
                ):
                    max_adjacency_score = adjacency_score
                    best_position = (x, y, pos_z)

        return best_position

    def _calculate_adjacency_score(
        self,
        x: float,
        y: float,
        width: float,
        length: float,
        placed_rooms_by_type: Dict[str, List[int]],
    ) -> float:
        """
        Calculate an adjacency score based on shared wall length.

        Args:
            x, y: Position to evaluate
            width, length: Room dimensions
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            float: Score representing total shared wall length
        """
        shared_wall_length = 0.0
        room_edges = [
            ((x, y), (x + width, y)),  # Bottom edge
            ((x + width, y), (x + width, y + length)),  # Right edge
            ((x, y + length), (x + width, y + length)),  # Top edge
            ((x, y), (x, y + length)),  # Left edge
        ]

        # Check all placed rooms for shared walls
        for room_type, room_ids in placed_rooms_by_type.items():
            for room_id in room_ids:
                if room_id not in self.spatial_grid.rooms:
                    continue

                room_data = self.spatial_grid.rooms[room_id]
                rx, ry, _ = room_data["position"]
                rw, rl, _ = room_data["dimensions"]

                # Calculate existing room edges
                room_edges_existing = [
                    ((rx, ry), (rx + rw, ry)),  # Bottom edge
                    ((rx + rw, ry), (rx + rw, ry + rl)),  # Right edge
                    ((rx, ry + rl), (rx + rw, ry + rl)),  # Top edge
                    ((rx, ry), (rx, ry + rl)),  # Left edge
                ]

                # Check for shared walls (overlapping edges)
                for edge1 in room_edges:
                    for edge2 in room_edges_existing:
                        shared_length = self._calculate_shared_edge_length(edge1, edge2)
                        shared_wall_length += shared_length

        return shared_wall_length

    def _calculate_shared_edge_length(self, edge1, edge2) -> float:
        """
        Calculate the length of overlap between two edges.

        Args:
            edge1: ((x1, y1), (x2, y2)) first edge
            edge2: ((x1, y1), (x2, y2)) second edge

        Returns:
            float: Length of overlap or 0 if no overlap
        """
        # Unpack edges
        (x1, y1), (x2, y2) = edge1
        (x3, y3), (x4, y4) = edge2

        # Check if edges are parallel and can overlap
        if abs(x1 - x2) < 0.01 and abs(x3 - x4) < 0.01:
            # Vertical edges
            if abs(x1 - x3) < 0.01:
                # Edges are collinear, check for overlap
                y_min = max(min(y1, y2), min(y3, y4))
                y_max = min(max(y1, y2), max(y3, y4))
                return max(0, y_max - y_min)

        elif abs(y1 - y2) < 0.01 and abs(y3 - y4) < 0.01:
            # Horizontal edges
            if abs(y1 - y3) < 0.01:
                # Edges are collinear, check for overlap
                x_min = max(min(x1, x2), min(x3, x4))
                x_max = min(max(x1, x2), max(x3, x4))
                return max(0, x_max - x_min)

        # No overlap
        return 0.0

    def _find_position_adjacent_to_any_room(
        self, room: Room, z: float, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a position adjacent to any existing room to maximize compactness.

        Args:
            room: Room to place
            z: Z-coordinate (floor level)
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        best_position = None
        max_adjacency_score = -1

        # Get all rooms on this floor
        floor_rooms = []
        for room_type, room_ids in placed_rooms_by_type.items():
            for room_id in room_ids:
                if room_id not in self.spatial_grid.rooms:
                    continue

                room_data = self.spatial_grid.rooms[room_id]
                room_z = room_data["position"][2]

                # Check if on the same floor
                if abs(room_z - z) < 0.1:
                    floor_rooms.append(room_data)

        # Try positions adjacent to each existing room
        for existing_room in floor_rooms:
            ex, ey, ez = existing_room["position"]
            ew, el, eh = existing_room["dimensions"]

            # Try all four sides
            positions = [
                (ex + ew, ey, z),  # Right
                (ex - room.width, ey, z),  # Left
                (ex, ey + el, z),  # Bottom
                (ex, ey - room.length, z),  # Top
            ]

            for x, y, pos_z in positions:
                # Skip if out of bounds
                if (
                    x < 0
                    or y < 0
                    or x + room.width > self.width
                    or y + room.length > self.length
                ):
                    continue

                # Calculate adjacency score
                adjacency_score = self._calculate_adjacency_score(
                    x, y, room.width, room.length, placed_rooms_by_type
                )

                # Check if position is valid and has better adjacency
                if adjacency_score > max_adjacency_score and self._is_valid_position(
                    x, y, pos_z, room.width, room.length, room.height
                ):
                    max_adjacency_score = adjacency_score
                    best_position = (x, y, pos_z)

        return best_position

    def _find_compact_perimeter_position(
        self, room: Room, floor: int, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a position on the perimeter that maintains compactness with existing rooms.

        Args:
            room: Room to place
            floor: Floor number
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        z = floor * self.floor_height

        # Get perimeter positions
        perimeter_positions = []

        # Add perimeter positions (grid-aligned)
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            perimeter_positions.append((x, 0, z))  # Front edge
            perimeter_positions.append((x, self.length - room.length, z))  # Back edge

        for y in range(
            0, int(self.length - room.length) + 1, int(self.structural_grid[1])
        ):
            perimeter_positions.append((0, y, z))  # Left edge
            perimeter_positions.append((self.width - room.width, y, z))  # Right edge

        # Sort by adjacency to existing rooms
        scored_positions = []
        for position in perimeter_positions:
            x, y, pos_z = position

            # Check if position is valid
            if not self._is_valid_position(
                x, y, pos_z, room.width, room.length, room.height
            ):
                continue

            # Calculate adjacency score
            adjacency_score = self._calculate_adjacency_score(
                x, y, room.width, room.length, placed_rooms_by_type
            )

            # Add to candidates
            scored_positions.append((position, adjacency_score))

        # Sort by adjacency score (highest first)
        scored_positions.sort(key=lambda p: p[1], reverse=True)

        # Return best position
        return scored_positions[0][0] if scored_positions else None

    def _find_compact_position_on_floor(
        self, room: Room, floor: int, placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find any position on the floor that maximizes compactness.

        Args:
            room: Room to place
            floor: Floor number
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        z = floor * self.floor_height

        # Try all grid-aligned positions on this floor
        best_position = None
        max_adjacency_score = -1

        # Step size based on grid
        step_x = self.structural_grid[0]
        step_y = self.structural_grid[1]

        # Try all positions
        for x in range(0, int(self.width - room.width) + 1, int(step_x)):
            for y in range(0, int(self.length - room.length) + 1, int(step_y)):
                # Check if position is valid
                if not self._is_valid_position(
                    x, y, z, room.width, room.length, room.height
                ):
                    continue

                # Calculate adjacency score
                adjacency_score = self._calculate_adjacency_score(
                    x, y, room.width, room.length, placed_rooms_by_type
                )

                # If no other rooms exist yet, prioritize corner placement
                if adjacency_score == 0:
                    # Calculate distance to nearest corner
                    corners = [
                        (0, 0),
                        (0, self.length),
                        (self.width, 0),
                        (self.width, self.length),
                    ]

                    room_center = (x + room.width / 2, y + room.length / 2)
                    min_corner_dist = min(
                        ((room_center[0] - cx) ** 2 + (room_center[1] - cy) ** 2) ** 0.5
                        for cx, cy in corners
                    )

                    # Inverse distance as score (closer to corner = better)
                    corner_score = 1.0 / (min_corner_dist + 1.0)

                    if corner_score > max_adjacency_score:
                        max_adjacency_score = corner_score
                        best_position = (x, y, z)

                # Otherwise use adjacency score
                elif adjacency_score > max_adjacency_score:
                    max_adjacency_score = adjacency_score
                    best_position = (x, y, z)

        return best_position

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
        Positions vertical circulation at junctions of multiple rooms.
        """
        if hasattr(room, "position") and room.position is not None:
            # Already handled in _handle_special_room_type
            return False

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

        # Get all placed rooms to find optimal junction position
        placed_rooms = {}
        for room_id, room_data in self.spatial_grid.rooms.items():
            placed_rooms[room_id] = room_data

        # Divide building into cells and score each cell for vertical circulation placement
        cell_size = max(self.structural_grid)
        grid_cells_x = int(self.width / cell_size)
        grid_cells_y = int(self.length / cell_size)

        # Score map for cells - higher score = better placement
        cell_scores = np.zeros((grid_cells_x, grid_cells_y))

        # Method 1: Score based on room adjacency potential
        for x_cell in range(grid_cells_x):
            for y_cell in range(grid_cells_y):
                # Calculate cell center
                cell_center_x = (x_cell + 0.5) * cell_size
                cell_center_y = (y_cell + 0.5) * cell_size

                # Count rooms within access distance by floor
                adjacent_rooms_by_floor = defaultdict(int)
                room_types_nearby = set()

                for room_id, room_data in placed_rooms.items():
                    # Skip if it's a vertical circulation
                    if room_data["type"] == "vertical_circulation":
                        continue

                    rx, ry, rz = room_data["position"]
                    rw, rl, rh = room_data["dimensions"]

                    # Calculate room center
                    room_center_x = rx + rw / 2
                    room_center_y = ry + rl / 2

                    # Calculate distance to cell center
                    distance = (
                        (cell_center_x - room_center_x) ** 2
                        + (cell_center_y - room_center_y) ** 2
                    ) ** 0.5

                    # Only consider rooms within reasonable distance
                    if distance <= 1.5 * cell_size:
                        # Determine floor
                        floor = int(rz / self.floor_height)
                        adjacent_rooms_by_floor[floor] += 1
                        room_types_nearby.add(room_data["type"])

                # Score is based on:
                # 1. Number of floors with adjacent rooms
                # 2. Total number of adjacent rooms
                # 3. Diversity of room types (weighted heavily - varied access is important)
                floors_with_rooms = len(adjacent_rooms_by_floor)
                total_adjacent_rooms = sum(adjacent_rooms_by_floor.values())
                room_type_diversity = len(room_types_nearby)

                # Calculate score - prioritize type diversity and floor coverage
                score = (
                    floors_with_rooms * 1.0
                    + total_adjacent_rooms * 0.5
                    + room_type_diversity * 2.0
                )

                # Bonus for being near the center of the building
                center_dist = (
                    (cell_center_x - self.width / 2) ** 2
                    + (cell_center_y - self.length / 2) ** 2
                ) ** 0.5
                center_factor = 1.0 - (center_dist / (self.width / 2 + self.length / 2))
                score += center_factor * 1.0

                cell_scores[x_cell, y_cell] = score

        # Get top scoring positions
        max_score = np.max(cell_scores)
        if max_score == 0:
            # If no scores (empty building), default to central position
            best_cell_x = grid_cells_x // 2
            best_cell_y = grid_cells_y // 2
        else:
            # Find cell with highest score
            best_idx = np.argmax(cell_scores)
            best_cell_y = best_idx % grid_cells_y
            best_cell_x = best_idx // grid_cells_y

        # Calculate position from best cell
        x = best_cell_x * cell_size
        y = best_cell_y * cell_size

        # Adjust to ensure it fits within bounds
        x = min(x, self.width - width)
        y = min(y, self.length - length)

        # Snap to grid
        x = round(x / self.structural_grid[0]) * self.structural_grid[0]
        y = round(y / self.structural_grid[1]) * self.structural_grid[1]

        # Check if position is valid
        if self._is_valid_position(x, y, start_z, width, length, total_height):
            # Place the vertical circulation element
            success = self.spatial_grid.place_room(
                room_id=room.id,
                x=x,
                y=y,
                z=start_z,
                width=width,
                length=length,
                height=total_height,
                room_type=room.room_type,
                metadata=room.metadata,
            )
            if success:
                print(
                    f"Successfully placed vertical circulation at ({x}, {y}, {start_z})"
                )
                return True

        # If optimal position fails, try alternative positions
        # Try positions with decreasing score
        flat_scores = cell_scores.flatten()
        indices = np.argsort(flat_scores)[::-1]  # Sort indices by descending score

        for idx in indices:
            cell_y = idx % grid_cells_y
            cell_x = idx // grid_cells_y

            # Calculate position from cell
            x = cell_x * cell_size
            y = cell_y * cell_size

            # Adjust to ensure it fits within bounds
            x = min(x, self.width - width)
            y = min(y, self.length - length)

            # Snap to grid
            x = round(x / self.structural_grid[0]) * self.structural_grid[0]
            y = round(y / self.structural_grid[1]) * self.structural_grid[1]

            # Skip already tried position
            if cell_x == best_cell_x and cell_y == best_cell_y:
                continue

            # Check if position is valid
            if self._is_valid_position(x, y, start_z, width, length, total_height):
                # Place the vertical circulation element
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=start_z,
                    width=width,
                    length=length,
                    height=total_height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )
                if success:
                    print(
                        f"Successfully placed vertical circulation at ({x}, {y}, {start_z})"
                    )
                    return True

        # If all else fails, force placement (mostly for empty buildings)
        x = self.structural_grid[0]
        y = self.structural_grid[1]

        success = self.spatial_grid.place_room(
            room_id=room.id,
            x=x,
            y=y,
            z=start_z,
            width=width,
            length=length,
            height=total_height,
            room_type=room.room_type,
            metadata=room.metadata,
            force_placement=True,
        )

        if success:
            print(f"Forced placement of vertical circulation at ({x}, {y}, {start_z})")
            return True

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
