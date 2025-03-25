"""
Reinforcement Learning engine for hotel layout generation.
Inherits from BaseEngine to leverage common functionality.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union, Set

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room
from hotel_design_ai.core.base_engine import BaseEngine


class RLPolicyNetwork(nn.Module):
    """Neural network that represents the policy for the RL agent."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, action_dim: int = 256):
        super(RLPolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)


class RLEngine(BaseEngine):
    """
    Reinforcement Learning engine for hotel layout generation.
    Uses the BaseEngine abstract class for common functionality.
    """

    def __init__(
        self,
        bounding_box: Tuple[float, float, float],
        grid_size: float = 1.0,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.7,
        exploration_min: float = 0.01,
        exploration_decay: float = 0.995,
        memory_size: int = 10000,
        building_config: Dict[str, Any] = None,
    ):
        """
        Initialize the RL engine.

        Args:
            bounding_box: (width, length, height) of buildable area in meters
            grid_size: Size of spatial grid cells in meters
            structural_grid: (x_spacing, y_spacing) of structural grid in meters
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            exploration_min: Minimum exploration rate
            exploration_decay: Rate at which exploration decays
            memory_size: Size of experience replay buffer
            building_config: Building configuration parameters
        """
        # Initialize the BaseEngine
        super().__init__(bounding_box, grid_size, structural_grid, building_config)

        # Room type info
        self.room_types = [
            "entrance",
            "lobby",
            "vertical_circulation",
            "restaurant",
            "meeting_room",
            "guest_room",
            "service_area",
            "back_of_house",
            "kitchen",
            "ballroom",
            "lounge",
            "pre_function",
            "maintenance",
            "mechanical",
            "pool",
            "fitness",
            "retail",
            "recreation",
            "entertainment",
            "office",
            "staff_area",
            "food_service",
            "parking",
            "circulation",
            "storage",
        ]

        # Initialize room placement priorities
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
            "mechanical": 2,
            "parking": 3,
        }

        # Initialize adjacency preferences
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

        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        # Memory buffer for experience replay
        self.memory = deque(maxlen=memory_size)

        # State and action dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()

        # Neural networks
        self.policy_network = RLPolicyNetwork(
            self.state_dim, hidden_dim=256, action_dim=self.action_dim
        )
        self.target_network = RLPolicyNetwork(
            self.state_dim, hidden_dim=256, action_dim=self.action_dim
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Training metrics
        self.training_iterations = 0
        self.average_reward = 0

        # Create placement strategies
        self.placement_strategies = self._initialize_placement_strategies()

    def _calculate_state_dim(self) -> int:
        """Calculate the dimension of the state representation"""
        simplified_grid_cells = (
            int(self.width / self.structural_grid[0])
            * int(self.length / self.structural_grid[1])
            * int(self.height / self.floor_height)
        )
        room_props = len(self.room_types) + 3
        floor_prefs = self.max_floor - self.min_floor + 1
        fixed_mask = simplified_grid_cells
        return simplified_grid_cells + room_props + floor_prefs + fixed_mask

    def _calculate_action_dim(self) -> int:
        """Calculate the dimension of the action space"""
        grid_cells_x = int(self.width / self.structural_grid[0])
        grid_cells_y = int(self.length / self.structural_grid[1])
        floors = int(self.height / self.floor_height)
        return grid_cells_x * grid_cells_y * floors

    def _initialize_placement_strategies(self) -> Dict[str, callable]:
        """Initialize room placement strategies"""
        strategies = {
            "vertical_circulation": self._place_vertical_circulation,
            "entrance": self._place_entrance,
            "parking": self._place_parking,
            "guest_room": self._place_guest_room,
            "exterior": self._place_exterior_room,
            "default": self._place_default,
        }
        return strategies

    def place_room_by_constraints(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Place a room according to architectural constraints.
        Implementation of the abstract method from BaseEngine.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        # Check for standard floor zones for guest rooms on upper floors
        if hasattr(self, "standard_floor_mask") and room.room_type == "guest_room":
            # For guest rooms on higher floors, require them to be in standard floor zones
            if hasattr(room, "floor") and room.floor is not None and room.floor > 0:
                success = self._place_in_standard_floor_zone(room, placed_rooms_by_type)
                if success:
                    return True

        # Get preferred floors for this room
        preferred_floors = self._get_preferred_floors(room)

        # Select placement strategy based on room type or characteristics
        if room.room_type in self.placement_strategies:
            strategy = self.placement_strategies[room.room_type]
        elif self._needs_exterior_placement(room):
            strategy = self.placement_strategies["exterior"]
        else:
            strategy = self.placement_strategies["default"]

        # Try to place using the selected strategy
        return strategy(room, placed_rooms_by_type, preferred_floors)

    def _place_vertical_circulation(
        self,
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]],
        preferred_floors: List[int],
    ) -> bool:
        """Special placement for vertical circulation elements"""
        # Calculate full height from min to max floor
        total_floors = self.max_floor - self.min_floor + 1
        total_height = total_floors * self.floor_height

        # Starting z-coordinate from the lowest floor
        start_z = self.min_floor * self.floor_height

        # Update the metadata
        if not room.metadata:
            room.metadata = {}
        room.metadata["is_core"] = True
        room.metadata["spans_floors"] = list(range(self.min_floor, self.max_floor + 1))

        # Try positions near the center first
        center_x = self.width / 2 - room.width / 2
        center_y = self.length / 2 - room.length / 2

        positions = [(center_x, center_y)]

        # Add grid-aligned positions in a spiral
        grid_x, grid_y = self.structural_grid
        max_radius = min(self.width, self.length) / 4

        for radius in range(int(grid_x), int(max_radius), int(grid_x)):
            for angle in range(0, 360, 45):
                angle_rad = angle * np.pi / 180
                x = center_x + radius * np.cos(angle_rad)
                y = center_y + radius * np.sin(angle_rad)
                x = round(x / grid_x) * grid_x
                y = round(y / grid_y) * grid_y
                x = max(0, min(x, self.width - room.width))
                y = max(0, min(y, self.length - room.length))
                positions.append((x, y))

        # Remove duplicates
        positions = list(set(positions))

        # Try each position
        for x, y in positions:
            if self._is_valid_position(
                x, y, start_z, room.width, room.length, total_height
            ):
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
                    allow_overlap=["parking"],
                )
                if success:
                    return True

        # Force placement as last resort
        x, y = self.structural_grid
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
            allow_overlap=["parking"],
            force_placement=True,
        )
        return success

    def _place_entrance(
        self,
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]],
        preferred_floors: List[int],
    ) -> bool:
        """Place entrance on front perimeter"""
        # Use the first preferred floor
        floor = preferred_floors[0] if preferred_floors else 0
        z = floor * self.floor_height

        # Try front edge (y=0) with preference for center
        center_x = (self.width - room.width) / 2
        center_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]

        # Try center position
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

        # Try other positions along front edge
        for offset in range(1, 10):
            for direction in [-1, 1]:  # Try both left and right of center
                x = center_x + direction * offset * self.structural_grid[0]
                if 0 <= x <= self.width - room.width:
                    if self._is_valid_position(
                        x, 0, z, room.width, room.length, room.height
                    ):
                        return self.spatial_grid.place_room(
                            room_id=room.id,
                            x=x,
                            y=0,
                            z=z,
                            width=room.width,
                            length=room.length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=room.metadata,
                        )

        # Try other perimeter positions if front fails
        perimeter_positions = []

        # Add side edges
        for y in range(
            0, int(self.length - room.length) + 1, int(self.structural_grid[1])
        ):
            perimeter_positions.append((0, y))  # Left edge
            perimeter_positions.append((self.width - room.width, y))  # Right edge

        # Add back edge
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            perimeter_positions.append((x, self.length - room.length))  # Back edge

        random.shuffle(perimeter_positions)

        for x, y in perimeter_positions:
            if self._is_valid_position(x, y, z, room.width, room.length, room.height):
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

        return False

    def _place_parking(
        self,
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]],
        preferred_floors: List[int],
    ) -> bool:
        """Special placement for parking areas"""
        # Use the first preferred floor (usually basement)
        floor = preferred_floors[0] if preferred_floors else self.min_floor
        z = floor * self.floor_height

        # For very large parking areas, consider splitting into sections
        if room.width * room.length > 1000:
            return self._place_parking_in_sections(room, z)

        # Try standard placement on structural grid
        for x in range(
            0, int(self.width - room.width) + 1, int(self.structural_grid[0])
        ):
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
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
                        allow_overlap=["vertical_circulation"],
                    )
                    if success:
                        return True

        # Force placement as last resort
        return self.spatial_grid.place_room(
            room_id=room.id,
            x=self.structural_grid[0],
            y=self.structural_grid[1],
            z=z,
            width=room.width,
            length=room.length,
            height=room.height,
            room_type=room.room_type,
            metadata=room.metadata,
            allow_overlap=["vertical_circulation"],
            force_placement=True,
        )

    def _place_parking_in_sections(self, room: Room, z: float) -> bool:
        """Place large parking areas as multiple sections"""
        total_area = room.width * room.length
        num_sections = min(6, max(2, int(total_area / 800)))
        area_per_section = total_area / num_sections

        # Calculate section dimensions aligned to structural grid
        grid_x, grid_y = self.structural_grid
        section_width = round(np.sqrt(area_per_section) / grid_x) * grid_x
        section_length = round(area_per_section / section_width / grid_y) * grid_y

        # Ensure minimum dimensions
        section_width = max(section_width, 2 * grid_x)
        section_length = max(section_length, 2 * grid_y)

        sections_placed = 0

        # Try to place each section
        for i in range(num_sections):
            section_id = room.id * 100 + i + 1

            # Try various positions
            for x in range(0, int(self.width - section_width) + 1, int(grid_x)):
                for y in range(0, int(self.length - section_length) + 1, int(grid_y)):
                    if self._is_valid_position(
                        x, y, z, section_width, section_length, room.height
                    ):
                        # Create section metadata
                        section_metadata = room.metadata.copy() if room.metadata else {}
                        section_metadata["parent_room_id"] = room.id
                        section_metadata["is_parking_section"] = True
                        section_metadata["section_number"] = i + 1
                        section_metadata["of_sections"] = num_sections

                        success = self.spatial_grid.place_room(
                            room_id=section_id,
                            x=x,
                            y=y,
                            z=z,
                            width=section_width,
                            length=section_length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=section_metadata,
                            allow_overlap=["vertical_circulation"],
                        )

                        if success:
                            sections_placed += 1
                            break

                if sections_placed > i:
                    break

        return sections_placed > 0

    def _place_guest_room(
        self,
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]],
        preferred_floors: List[int],
    ) -> bool:
        """Place guest rooms with preference for exterior walls and standard zones"""
        if hasattr(self, "standard_floor_mask"):
            for floor in preferred_floors:
                z = floor * self.floor_height

                # Try perimeter positions within standard zones
                for x in range(
                    0, int(self.width - room.width) + 1, int(self.grid_size)
                ):
                    for y in range(
                        0, int(self.length - room.length) + 1, int(self.grid_size)
                    ):
                        # Check if on perimeter
                        is_perimeter = (
                            x < 0.1
                            or y < 0.1
                            or x + room.width > self.width - 0.1
                            or y + room.length > self.length - 0.1
                        )

                        if not is_perimeter:
                            continue

                        # Check if in standard zone
                        grid_x = int(x / self.grid_size)
                        grid_y = int(y / self.grid_size)
                        room_width_cells = int(room.width / self.grid_size)
                        room_length_cells = int(room.length / self.grid_size)

                        if not np.all(
                            self.standard_floor_mask[
                                grid_x : grid_x + room_width_cells,
                                grid_y : grid_y + room_length_cells,
                            ]
                        ):
                            continue

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

        # Fall back to regular exterior placement
        return self._place_exterior_room(room, placed_rooms_by_type, preferred_floors)

    def _place_exterior_room(
        self,
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]],
        preferred_floors: List[int],
    ) -> bool:
        """Place a room on the building perimeter"""
        for floor in preferred_floors:
            z = floor * self.floor_height

            # Generate perimeter positions
            perimeter_positions = []

            # Top edge (y=0)
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, 0))

            # Bottom edge
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, self.length - room.length))

            # Left edge
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((0, y))

            # Right edge
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((self.width - room.width, y))

            # Shuffle to avoid patterns
            random.shuffle(perimeter_positions)

            # Try each position
            for x, y in perimeter_positions:
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

        # Fall back to default placement
        return self._place_default(room, placed_rooms_by_type, preferred_floors)

    def _place_default(
        self,
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]],
        preferred_floors: List[int],
    ) -> bool:
        """Default placement strategy for rooms with no special requirements"""
        for floor in preferred_floors:
            z = floor * self.floor_height

            # Try adjacency-based placement if applicable
            if room.room_type in self.adjacency_preferences:
                success = self._try_adjacency_placement(
                    room, placed_rooms_by_type, floor
                )
                if success:
                    return True

            # Otherwise try grid-aligned positions
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                for y in range(
                    0, int(self.length - room.length) + 1, int(self.structural_grid[1])
                ):
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

        # If no placement found on preferred floors, try others
        other_floors = [
            f
            for f in range(self.min_floor, self.max_floor + 1)
            if f not in preferred_floors
        ]

        for floor in other_floors:
            z = floor * self.floor_height

            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                for y in range(
                    0, int(self.length - room.length) + 1, int(self.structural_grid[1])
                ):
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

        return False

    def _try_adjacency_placement(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]], floor: int
    ) -> bool:
        """Try to place a room adjacent to preferred room types"""
        z = floor * self.floor_height

        # Get adjacency preferences for this room type
        preferred_types = self.adjacency_preferences.get(room.room_type, [])

        for adj_type in preferred_types:
            if adj_type not in placed_rooms_by_type:
                continue

            # Try adjacent to each placed room of this type
            for adj_room_id in placed_rooms_by_type[adj_type]:
                if adj_room_id not in self.spatial_grid.rooms:
                    continue

                adj_room = self.spatial_grid.rooms[adj_room_id]
                adj_x, adj_y, adj_z = adj_room["position"]
                adj_w, adj_l, adj_h = adj_room["dimensions"]

                # Skip if not on same floor
                adj_floor = int(adj_z / self.floor_height)
                if adj_floor != floor:
                    continue

                # Try positions adjacent to this room
                positions = [
                    (adj_x + adj_w, adj_y, z),  # Right
                    (adj_x - room.width, adj_y, z),  # Left
                    (adj_x, adj_y + adj_l, z),  # Bottom
                    (adj_x, adj_y - room.length, z),  # Top
                ]

                for x, y, z_pos in positions:
                    if self._is_valid_position(
                        x, y, z_pos, room.width, room.length, room.height
                    ):
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

    def _needs_exterior_placement(self, room: Room) -> bool:
        """Determine if a room needs exterior placement based on type"""
        exterior_room_types = [
            "entrance",
            "lobby",
            "restaurant",
            "guest_room",
            "retail",
        ]
        return room.room_type in exterior_room_types

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout using reinforcement learning.
        Implementation of the abstract method from BaseEngine.

        Args:
            rooms: List of Room objects to place

        Returns:
            SpatialGrid: The generated layout
        """
        # Clear non-fixed elements
        self.clear_non_fixed_elements()

        # Restore fixed elements
        for room_id, position in self.fixed_elements.items():
            # Find the room in the input list
            room = next((r for r in rooms if r.id == room_id), None)
            if room:
                x, y, z = position
                self.spatial_grid.place_room(
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

        # Sort rooms by priority
        sorted_rooms = sorted(
            rooms,
            key=lambda r: (
                # Fixed rooms first
                r.id not in self.fixed_elements,
                # Then by priority
                -self.placement_priorities.get(r.room_type, 0),
                # Then by size (larger rooms first)
                -(r.width * r.length),
            ),
        )

        # Filter out fixed rooms
        rooms_to_place = [r for r in sorted_rooms if r.id not in self.fixed_elements]

        # Statistics tracking
        placement_stats = {
            "total": len(rooms_to_place),
            "placed": 0,
            "failed": 0,
            "by_type": defaultdict(lambda: {"total": 0, "placed": 0, "failed": 0}),
        }

        # Tracking placed rooms by type
        placed_rooms_by_type = defaultdict(list)
        for room_id, room_data in self.spatial_grid.rooms.items():
            room_type = room_data["type"]
            placed_rooms_by_type[room_type].append(room_id)

        # Place each room
        for room in rooms_to_place:
            # Track statistics
            placement_stats["by_type"][room.room_type]["total"] += 1

            # Place room
            success = self.place_room_by_constraints(room, placed_rooms_by_type)

            # Update statistics
            if success:
                placement_stats["placed"] += 1
                placement_stats["by_type"][room.room_type]["placed"] += 1
                placed_rooms_by_type[room.room_type].append(room.id)
            else:
                placement_stats["failed"] += 1
                placement_stats["by_type"][room.room_type]["failed"] += 1
                print(f"Failed to place {room.room_type} (id={room.id})")

        # Print statistics
        success_rate = 0
        if placement_stats["total"] > 0:
            success_rate = placement_stats["placed"] / placement_stats["total"] * 100
            print(
                f"Successfully placed: {placement_stats['placed']} ({success_rate:.1f}%)"
            )

        return self.spatial_grid

    def calculate_reward(self, layout: SpatialGrid) -> float:
        """
        Calculate reward for a layout based on architectural principles.

        Args:
            layout: Layout to evaluate

        Returns:
            float: Reward value (0 to 1)
        """
        # Space utilization
        space_util = layout.calculate_space_utilization()
        utilization_reward = space_util * 5.0

        # Adjacency satisfaction
        adjacency_reward = self._calculate_adjacency_reward(layout) * 3.0

        # Floor assignment
        floor_reward = self._calculate_floor_reward(layout) * 2.0

        # Structural alignment
        alignment_reward = self._calculate_alignment_reward(layout) * 1.0

        # Total reward normalized
        total_reward = (
            utilization_reward + adjacency_reward + floor_reward + alignment_reward
        ) / 11.0

        return total_reward

    def _calculate_adjacency_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward component for adjacency relationships."""
        adjacency_score = 0.0
        total_relationships = 0

        # Check each room's adjacencies
        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]
            preferred = self.adjacency_preferences.get(room_type, [])
            if not preferred:
                continue

            neighbors = layout.get_room_neighbors(room_id)
            satisfied = 0
            for neighbor_id in neighbors:
                if neighbor_id in layout.rooms:
                    neighbor_type = layout.rooms[neighbor_id]["type"]
                    if neighbor_type in preferred:
                        satisfied += 1

            if preferred:
                adjacency_score += min(1.0, satisfied / len(preferred))
                total_relationships += 1

        # Normalize score
        if total_relationships > 0:
            return adjacency_score / total_relationships
        return 0.0

    def _calculate_floor_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward component for floor placement."""
        floor_score = 0.0
        total_rooms = 0

        # Floor preferences mapping
        floor_preferences = {
            "entrance": [0],
            "lobby": [0],
            "restaurant": [0],
            "kitchen": [0],
            "meeting_room": [0],
            "ballroom": [0],
            "guest_room": list(range(1, self.max_floor + 1)),
            "service_area": list(range(self.min_floor, 1)),
            "back_of_house": list(range(self.min_floor, 1)),
            "mechanical": list(range(self.min_floor, 0)),
            "parking": list(range(self.min_floor, 0)),
        }

        # Check each room's floor placement
        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]
            preferred = floor_preferences.get(room_type)
            if not preferred:
                continue

            z = room_data["position"][2]
            floor = int(z / self.floor_height)

            if floor in preferred:
                floor_score += 1.0
            else:
                # Partial credit based on how far from preferred
                min_dist = min(abs(floor - p) for p in preferred)
                floor_score += max(0.0, 1.0 - min_dist * 0.2)

            total_rooms += 1

        # Normalize score
        if total_rooms > 0:
            return floor_score / total_rooms
        return 0.0

    def _calculate_alignment_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward for structural grid alignment."""
        aligned_rooms = 0
        total_rooms = len(layout.rooms)

        grid_x, grid_y = self.structural_grid

        for room_id, room_data in layout.rooms.items():
            x, y, _ = room_data["position"]

            # Check alignment with structural grid
            x_aligned = abs(x % grid_x) < 0.1 * grid_x
            y_aligned = abs(y % grid_y) < 0.1 * grid_y

            if x_aligned and y_aligned:
                aligned_rooms += 1

        if total_rooms > 0:
            return aligned_rooms / total_rooms
        return 0.0

    def update_model(self, user_rating: float):
        """
        Update the RL model based on user feedback.

        Args:
            user_rating: User rating (0-10) of the current layout
        """
        # Convert user rating to reward
        reward = user_rating / 10.0

        # Update exploration rate
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate * self.exploration_decay
        )

        # Update training metrics
        self.training_iterations += 1
        self.average_reward = self.average_reward * 0.95 + reward * 0.05

    def save_model(self, path: str):
        """Save the RL model to disk"""
        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "exploration_rate": self.exploration_rate,
                "training_iterations": self.training_iterations,
                "average_reward": self.average_reward,
            },
            path,
        )

    def load_model(self, path: str):
        """Load the RL model from disk"""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.exploration_rate = checkpoint["exploration_rate"]
        self.training_iterations = checkpoint["training_iterations"]
        self.average_reward = checkpoint["average_reward"]
