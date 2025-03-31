"""
Reinforcement Learning engine for hotel layout generation.
Inherits from BaseEngine to leverage common functionality.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Add these imports at the top
import math
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union, Set

from hotel_design_ai.config.config_loader_grid import calculate_grid_aligned_dimensions
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

    def _place_vertical_circulation(self, room, placed_rooms_by_type, preferred_floors):
        """Special placement for vertical circulation elements."""
        # Get grid-aligned dimensions
        width, length = self.adjust_room_dimensions_to_grid(room)

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
        # Align to the structural grid
        center_x = ((self.width / 2) // self.structural_grid[0]) * self.structural_grid[
            0
        ]
        center_y = (
            (self.length / 2) // self.structural_grid[1]
        ) * self.structural_grid[1]

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
            if self._is_valid_position(x, y, start_z, width, length, total_height):
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
            width=width,
            length=length,
            height=total_height,
            room_type=room.room_type,
            metadata=room.metadata,
            allow_overlap=["parking"],
            force_placement=True,
        )
        return success

    def _place_entrance(self, room, placed_rooms_by_type, preferred_floors):
        """Place entrance on front perimeter"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        # Use the first preferred floor
        floor = preferred_floors[0] if preferred_floors else 0
        z = floor * self.floor_height

        # Try front edge (y=0) with preference for center
        center_x = (
            ((self.width - width) / 2)
            // self.structural_grid[0]
            * self.structural_grid[0]
        )

        # Try center position
        if self._is_valid_position(center_x, 0, z, width, length, room.height):
            return self.spatial_grid.place_room(
                room_id=room.id,
                x=center_x,
                y=0,
                z=z,
                width=width,
                length=length,
                height=room.height,
                room_type=room.room_type,
                metadata=room.metadata,
            )

        # Try other positions along front edge
        for offset in range(1, 10):
            for direction in [-1, 1]:  # Try both left and right of center
                x = center_x + direction * offset * self.structural_grid[0]
                if 0 <= x <= self.width - width:
                    if self._is_valid_position(x, 0, z, width, length, room.height):
                        return self.spatial_grid.place_room(
                            room_id=room.id,
                            x=x,
                            y=0,
                            z=z,
                            width=width,
                            length=length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=room.metadata,
                        )

        # Try other grid-aligned perimeter positions if front fails
        perimeter_positions = []

        # Add side edges
        for y in range(0, int(self.length - length) + 1, int(self.structural_grid[1])):
            perimeter_positions.append((0, y))  # Left edge
            perimeter_positions.append((self.width - width, y))  # Right edge

        # Add back edge
        for x in range(0, int(self.width - width) + 1, int(self.structural_grid[0])):
            perimeter_positions.append((x, self.length - length))  # Back edge

        random.shuffle(perimeter_positions)

        for x, y in perimeter_positions:
            if self._is_valid_position(x, y, z, width, length, room.height):
                return self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=z,
                    width=width,
                    length=length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )

        return False

    def _place_parking(self, room, placed_rooms_by_type, preferred_floors):
        """Special placement for parking areas"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        # Use the first preferred floor (usually basement)
        floor = preferred_floors[0] if preferred_floors else self.min_floor
        z = floor * self.floor_height

        # For very large parking areas, consider splitting into sections
        if width * length > 1000:
            return self._place_parking_in_sections(room, z)

        # Try standard placement on structural grid
        for x in range(0, int(self.width - width) + 1, int(self.structural_grid[0])):
            for y in range(
                0, int(self.length - length) + 1, int(self.structural_grid[1])
            ):
                if self._is_valid_position(x, y, z, width, length, room.height):
                    success = self.spatial_grid.place_room(
                        room_id=room.id,
                        x=x,
                        y=y,
                        z=z,
                        width=width,
                        length=length,
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
            width=width,
            length=length,
            height=room.height,
            room_type=room.room_type,
            metadata=room.metadata,
            allow_overlap=["vertical_circulation"],
            force_placement=True,
        )

    def _place_parking_in_sections(self, room, z):
        """Place large parking areas as multiple sections"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        total_area = width * length
        num_sections = min(6, max(2, int(total_area / 800)))
        area_per_section = total_area / num_sections

        # Calculate section dimensions aligned to structural grid
        grid_x, grid_y = self.structural_grid

        # Try to make sections align with full grid cells
        section_width_cells = max(
            1, round(math.sqrt(area_per_section / (grid_x * grid_y)))
        )
        section_width = section_width_cells * grid_x

        section_length_cells = max(
            1, round(area_per_section / (section_width * grid_y) / grid_y)
        )
        section_length = section_length_cells * grid_y

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

    def _place_guest_room(self, room, placed_rooms_by_type, preferred_floors):
        """Place guest rooms with preference for exterior walls and standard zones"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        if hasattr(self, "standard_floor_mask"):
            for floor in preferred_floors:
                z = floor * self.floor_height

                # Try perimeter positions within standard zones
                for x in range(
                    0, int(self.width - width) + 1, int(self.structural_grid[0])
                ):
                    for y in range(
                        0, int(self.length - length) + 1, int(self.structural_grid[1])
                    ):
                        # Check if on perimeter
                        is_perimeter = (
                            x < 0.1
                            or y < 0.1
                            or x + width > self.width - 0.1
                            or y + length > self.length - 0.1
                        )

                        if not is_perimeter:
                            continue

                        # Check if in standard zone
                        grid_x = int(x / self.grid_size)
                        grid_y = int(y / self.grid_size)
                        room_width_cells = int(width / self.grid_size)
                        room_length_cells = int(length / self.grid_size)

                        if not np.all(
                            self.standard_floor_mask[
                                grid_x : grid_x + room_width_cells,
                                grid_y : grid_y + room_length_cells,
                            ]
                        ):
                            continue

                        if self._is_valid_position(x, y, z, width, length, room.height):
                            success = self.spatial_grid.place_room(
                                room_id=room.id,
                                x=x,
                                y=y,
                                z=z,
                                width=width,
                                length=length,
                                height=room.height,
                                room_type=room.room_type,
                                metadata=room.metadata,
                            )
                            if success:
                                return True

        # Fall back to regular exterior placement
        return self._place_exterior_room(room, placed_rooms_by_type, preferred_floors)

    def _place_exterior_room(self, room, placed_rooms_by_type, preferred_floors):
        """Place a room on the building perimeter"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        for floor in preferred_floors:
            z = floor * self.floor_height

            # Generate perimeter positions
            perimeter_positions = []

            # Top edge (y=0)
            for x in range(
                0, int(self.width - width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, 0))

            # Bottom edge
            for x in range(
                0, int(self.width - width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, self.length - length))

            # Left edge
            for y in range(
                0, int(self.length - length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((0, y))

            # Right edge
            for y in range(
                0, int(self.length - length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((self.width - width, y))

            # Shuffle to avoid patterns
            random.shuffle(perimeter_positions)

            # Try each position
            for x, y in perimeter_positions:
                if self._is_valid_position(x, y, z, width, length, room.height):
                    success = self.spatial_grid.place_room(
                        room_id=room.id,
                        x=x,
                        y=y,
                        z=z,
                        width=width,
                        length=length,
                        height=room.height,
                        room_type=room.room_type,
                        metadata=room.metadata,
                    )
                    if success:
                        return True

        # Fall back to default placement
        return self._place_default(room, placed_rooms_by_type, preferred_floors)

    def _place_default(self, room, placed_rooms_by_type, preferred_floors):
        """Default placement strategy for rooms with no special requirements"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

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
                0, int(self.width - width) + 1, int(self.structural_grid[0])
            ):
                for y in range(
                    0, int(self.length - length) + 1, int(self.structural_grid[1])
                ):
                    if self._is_valid_position(x, y, z, width, length, room.height):
                        success = self.spatial_grid.place_room(
                            room_id=room.id,
                            x=x,
                            y=y,
                            z=z,
                            width=width,
                            length=length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=room.metadata,
                        )
                        if success:
                            return True

        # If all preferred floors failed, try alternative floors
        other_floors = [
            f
            for f in range(self.min_floor, self.max_floor + 1)
            if f not in preferred_floors
        ]

        for floor in other_floors:
            z = floor * self.floor_height

            for x in range(
                0, int(self.width - width) + 1, int(self.structural_grid[0])
            ):
                for y in range(
                    0, int(self.length - length) + 1, int(self.structural_grid[1])
                ):
                    if self._is_valid_position(x, y, z, width, length, room.height):
                        success = self.spatial_grid.place_room(
                            room_id=room.id,
                            x=x,
                            y=y,
                            z=z,
                            width=width,
                            length=length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=room.metadata,
                        )
                        if success:
                            return True

        return False

    def _try_adjacency_placement(self, room, placed_rooms_by_type, floor):
        """Try to place a room adjacent to preferred room types"""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

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
                    (adj_x - width, adj_y, z),  # Left
                    (adj_x, adj_y + adj_l, z),  # Bottom
                    (adj_x, adj_y - length, z),  # Top
                ]

                for x, y, z_pos in positions:
                    # Snap to grid
                    x, y, z_pos = self.snap_position_to_grid(x, y, z_pos)

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
                            return True

        return False

    def _needs_exterior_placement(self, room_type: str) -> bool:
        """
        Determine if a room needs exterior placement based on type.

        Args:
            room_type: The type of room as a string

        Returns:
            bool: True if the room type needs exterior placement
        """
        exterior_room_types = [
            "entrance",
            "lobby",
            "restaurant",
            "guest_room",
            "retail",
        ]
        return room_type in exterior_room_types

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
        Add extra reward for grid alignment.
        """
        # Original reward calculation
        original_reward = super().calculate_reward(layout)

        # Calculate grid alignment score
        grid_alignment_score = self._calculate_grid_alignment_score(layout)

        # Combine rewards (giving grid alignment a high weight of 2.0)
        total_reward = (original_reward * 0.8) + (grid_alignment_score * 0.2)

        return total_reward

    def _calculate_grid_alignment_score(self, layout: SpatialGrid) -> float:
        """Calculate how well rooms align with the structural grid."""
        grid_x, grid_y = self.structural_grid
        aligned_rooms = 0
        total_rooms = len(layout.rooms)

        if total_rooms == 0:
            return 1.0  # No rooms to check

        for room_id, room_data in layout.rooms.items():
            x, y, _ = room_data["position"]
            w, l, _ = room_data["dimensions"]

            # Check position alignment with full grid
            x_aligned = abs(x % grid_x) < 0.1 * grid_x
            y_aligned = abs(y % grid_y) < 0.1 * grid_y

            # Check dimension alignment with full grid
            w_aligned = abs(w % grid_x) < 0.1 * grid_x
            l_aligned = abs(l % grid_y) < 0.1 * grid_y

            # Room is fully aligned if both position and dimensions are aligned
            if (x_aligned and y_aligned) and (w_aligned or l_aligned):
                aligned_rooms += 1

        return aligned_rooms / total_rooms

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

    def place_room_by_constraints(self, room, placed_rooms_by_type):
        """
        Place a room according to architectural constraints with priority on direct adjacency.
        Modified RL version that creates tight packing with no corridors.
        """
        # First adjust dimensions to align with grid
        original_width = room.width
        original_length = room.length

        # Apply grid alignment to dimensions
        width, length = self.adjust_room_dimensions_to_grid(room)

        # Temporarily update room dimensions for placement
        room.width = width
        room.length = length

        # Check for standard floor zones for guest rooms on upper floors
        if hasattr(self, "standard_floor_mask") and room.room_type == "guest_room":
            # For guest rooms on higher floors, require them to be in standard floor zones
            if hasattr(room, "floor") and room.floor is not None and room.floor > 0:
                success = self._place_in_standard_floor_zone(room, placed_rooms_by_type)
                if success:
                    # Restore original dimensions
                    room.width = original_width
                    room.length = original_length
                    return True

        # Get preferred floors for this room
        preferred_floors = self._get_preferred_floors(room)

        # Select placement strategy based on room type or characteristics
        if room.room_type == "vertical_circulation":
            strategy = self._place_vertical_circulation_space_focused
        elif room.room_type == "entrance":
            strategy = self._place_entrance_space_focused
        elif room.room_type == "parking":
            strategy = self._place_parking_space_focused
        elif self._needs_exterior_placement(room.room_type):
            strategy = self._place_exterior_room_space_focused
        else:
            strategy = self._place_default_space_focused

        # Try to place using the selected strategy
        success = strategy(room, placed_rooms_by_type, preferred_floors)

        # Restore original dimensions regardless of outcome
        room.width = original_width
        room.length = original_length

        return success

    def _place_vertical_circulation_space_focused(
        self, room, placed_rooms_by_type, preferred_floors
    ):
        """Special placement for vertical circulation elements in a space-focused layout."""
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

        # Analyze existing rooms to find optimal placement for maximum adjacency
        adjacency_scores = {}

        # Step size for evaluation
        step_size = min(self.structural_grid[0], self.structural_grid[1]) / 2

        # For each possible position, calculate an adjacency potential
        for x in range(0, int(self.width - room.width) + 1, int(step_size)):
            for y in range(0, int(self.length - room.length) + 1, int(step_size)):
                # Skip if position not valid
                if not self._is_valid_position(
                    x, y, start_z, room.width, room.length, total_height
                ):
                    continue

                # Calculate proximity to other rooms on different floors
                score = self._calculate_circulation_adjacency_score(
                    x, y, room.width, room.length, placed_rooms_by_type
                )

                # Store score
                adjacency_scores[(x, y)] = score

        # If no valid positions, try central position
        if not adjacency_scores:
            # Try center of building
            center_x = (self.width - room.width) / 2
            center_y = (self.length - room.length) / 2

            # Snap to grid
            center_x = (
                round(center_x / self.structural_grid[0]) * self.structural_grid[0]
            )
            center_y = (
                round(center_y / self.structural_grid[1]) * self.structural_grid[1]
            )

            # Try to place at center
            if self._is_valid_position(
                center_x, center_y, start_z, room.width, room.length, total_height
            ):
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=center_x,
                    y=center_y,
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

            # If center doesn't work, try a corner
            corners = [
                (0, 0),
                (0, self.length - room.length),
                (self.width - room.width, 0),
                (self.width - room.width, self.length - room.length),
            ]

            for corner_x, corner_y in corners:
                if self._is_valid_position(
                    corner_x, corner_y, start_z, room.width, room.length, total_height
                ):
                    success = self.spatial_grid.place_room(
                        room_id=room.id,
                        x=corner_x,
                        y=corner_y,
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

        # Try positions in order of decreasing score
        sorted_positions = sorted(
            adjacency_scores.items(), key=lambda x: x[1], reverse=True
        )

        for (x, y), score in sorted_positions:
            # Snap to grid
            x_snapped = round(x / self.structural_grid[0]) * self.structural_grid[0]
            y_snapped = round(y / self.structural_grid[1]) * self.structural_grid[1]

            # Try to place
            if self._is_valid_position(
                x_snapped, y_snapped, start_z, room.width, room.length, total_height
            ):
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x_snapped,
                    y=y_snapped,
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

    def _calculate_circulation_adjacency_score(
        self, x, y, width, length, placed_rooms_by_type
    ):
        """
        Calculate a score for vertical circulation based on maximizing adjacency across floors.
        Higher score means better placement for vertical circulation.
        """
        circulation_position = (x, y, width, length)

        # Track room adjacency by floor
        floor_adjacency = {}
        total_adjacency = 0
        served_room_types = set()

        # Examine each placed room
        for room_type, room_ids in placed_rooms_by_type.items():
            # Skip other vertical circulation
            if room_type == "vertical_circulation":
                continue

            for room_id in room_ids:
                if room_id not in self.spatial_grid.rooms:
                    continue

                room_data = self.spatial_grid.rooms[room_id]
                rx, ry, rz = room_data["position"]
                rw, rl, rh = room_data["dimensions"]

                # Determine floor
                room_floor = int(rz / self.floor_height)

                # Check if circulation would be adjacent to this room
                is_adjacent = self._check_adjacency(
                    (x, y, width, length), (rx, ry, rw, rl)
                )

                if is_adjacent:
                    # Count adjacency for this floor
                    if room_floor not in floor_adjacency:
                        floor_adjacency[room_floor] = 0

                    floor_adjacency[room_floor] += 1
                    total_adjacency += 1
                    served_room_types.add(room_type)

        # Calculate score components:
        # 1. Number of floors with adjacent rooms
        # 2. Total adjacency count
        # 3. Diversity of room types served
        floor_count = len(floor_adjacency)
        type_diversity = len(served_room_types)

        # Weight components to prioritize floor coverage and type diversity
        score = (floor_count * 2.0) + (total_adjacency * 0.5) + (type_diversity * 1.5)

        # Add central bias - give slight preference to central positions
        center_x = self.width / 2
        center_y = self.length / 2

        # Calculate distance to center as a percentage of maximum possible distance
        center_distance = (
            (x + width / 2 - center_x) ** 2 + (y + length / 2 - center_y) ** 2
        ) ** 0.5
        max_distance = ((self.width) ** 2 + (self.length) ** 2) ** 0.5 / 2

        # Convert to a 0-1 value (1 = at center, 0 = at furthest corner)
        center_factor = 1.0 - (center_distance / max_distance)

        # Add to score (with lower weight)
        score += center_factor * 0.5

        return score

    def _check_adjacency(self, box1, box2):
        """
        Check if two boxes are adjacent.

        Args:
            box1: (x, y, width, length) of first box
            box2: (x, y, width, length) of second box

        Returns:
            bool: True if boxes are adjacent
        """
        x1, y1, w1, l1 = box1
        x2, y2, w2, l2 = box2

        # Calculate box boundaries
        left1, right1 = x1, x1 + w1
        bottom1, top1 = y1, y1 + l1

        left2, right2 = x2, x2 + w2
        bottom2, top2 = y2, y2 + l2

        # Check if boxes intersect (which is not adjacency)
        if left1 < right2 and right1 > left2 and bottom1 < top2 and top1 > bottom2:
            return False

        # Check if boxes are adjacent
        adjacent_horizontally = (
            abs(right1 - left2) < 0.01 or abs(right2 - left1) < 0.01
        ) and (  # x-adjacency
            bottom1 < top2 and top1 > bottom2
        )  # y-overlap

        adjacent_vertically = (
            abs(top1 - bottom2) < 0.01 or abs(top2 - bottom1) < 0.01
        ) and (  # y-adjacency
            left1 < right2 and right1 > left2
        )  # x-overlap

        return adjacent_horizontally or adjacent_vertically

    def _place_entrance_space_focused(
        self, room, placed_rooms_by_type, preferred_floors
    ):
        """Place entrance with direct adjacency to preferred room types."""
        # Use the first preferred floor
        floor = preferred_floors[0] if preferred_floors else 0
        z = floor * self.floor_height

        # Try front edge (y=0) with preference for center
        center_x = (
            ((self.width - room.width) / 2)
            // self.structural_grid[0]
            * self.structural_grid[0]
        )

        # Try center position
        if self._is_valid_position(
            center_x, 0, z, room.width, room.length, room.height
        ):
            # Check if we can place it adjacent to any lobby
            if "lobby" in placed_rooms_by_type:
                # Find a position adjacent to lobby
                lobby_adjacent_pos = self._find_position_adjacent_to_type(
                    room, "lobby", placed_rooms_by_type, z
                )
                if lobby_adjacent_pos:
                    x, y, z_pos = lobby_adjacent_pos
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

            # If no lobby or lobby adjacency fails, place at center front
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

        # If front fails, try position with maximum adjacency to important rooms
        position = self._find_position_with_max_adjacency_for_type_for_type(
            room,
            room.room_type,
            placed_rooms_by_type,
            z,
            ["lobby", "vertical_circulation"],
        )
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

        # Try other grid-aligned perimeter positions if front fails
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

    def _find_position_adjacent_to_type(
        self, room, target_type, placed_rooms_by_type, z
    ):
        """
        Find a position adjacent to a specific room type.

        Args:
            room: Room to place
            target_type: Target room type to be adjacent to
            placed_rooms_by_type: Dictionary of already placed rooms by type
            z: Z-coordinate for placement

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        if target_type not in placed_rooms_by_type:
            return None

        for room_id in placed_rooms_by_type[target_type]:
            if room_id not in self.spatial_grid.rooms:
                continue

            target_room = self.spatial_grid.rooms[room_id]
            tx, ty, tz = target_room["position"]
            tw, tl, th = target_room["dimensions"]

            # Skip if not on same floor
            if abs(tz - z) > 0.1:
                continue

            # Try positions directly adjacent to the target room
            positions = [
                (tx + tw, ty, z),  # Right of target
                (tx - room.width, ty, z),  # Left of target
                (tx, ty + tl, z),  # Below target
                (tx, ty - room.length, z),  # Above target
            ]

            # Try each position
            for pos in positions:
                x, y, pos_z = pos

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
                    x, y, pos_z, room.width, room.length, room.height
                ):
                    return pos

        return None

    def _find_position_with_max_adjacency_for_type(
        self, room, room_type, placed_rooms_by_type, z, priority_types=None
    ):
        """
        Find position with maximum adjacency, prioritizing certain room types.

        Args:
            room: Room to place
            room_type: Type of room being placed
            placed_rooms_by_type: Dictionary of already placed rooms by type
            z: Z-coordinate for placement
            priority_types: List of room types to prioritize for adjacency

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        best_position = None
        best_score = -1

        # Set default priority types if none provided
        if priority_types is None:
            priority_types = []

        # Get all positions with potential adjacency
        candidate_positions = []

        # Add positions adjacent to existing rooms
        for type_name, room_ids in placed_rooms_by_type.items():
            # Prioritize adjacency to priority types
            priority_factor = 2.0 if type_name in priority_types else 1.0

            for room_id in room_ids:
                if room_id not in self.spatial_grid.rooms:
                    continue

                room_data = self.spatial_grid.rooms[room_id]
                rx, ry, rz = room_data["position"]
                rw, rl, rh = room_data["dimensions"]

                # Skip if not on same floor
                if abs(rz - z) > 0.1:
                    continue

                # Add positions adjacent to this room
                positions = [
                    (rx + rw, ry, z),  # Right
                    (rx - room.width, ry, z),  # Left
                    (rx, ry + rl, z),  # Below
                    (rx, ry - room.length, z),  # Above
                ]

                for pos in positions:
                    candidate_positions.append((pos, priority_factor))

        # If no candidates, return None
        if not candidate_positions:
            return None

        # Try each candidate position
        for (x, y, pos_z), priority_factor in candidate_positions:
            # Check bounds
            if (
                x < 0
                or y < 0
                or x + room.width > self.width
                or y + room.length > self.length
            ):
                continue

            # Check position validity
            if not self._is_valid_position(
                x, y, pos_z, room.width, room.length, room.height
            ):
                continue

            # Calculate adjacency score
            adjacency_score = self._calculate_adjacency_score(
                x, y, room.width, room.length, placed_rooms_by_type
            )

            # Apply priority factor
            weighted_score = adjacency_score * priority_factor

            # Update best position if better
            if weighted_score > best_score:
                best_score = weighted_score
                best_position = (x, y, pos_z)

        return best_position

    def _place_parking_space_focused(
        self, room, placed_rooms_by_type, preferred_floors
    ):
        """Special placement for parking areas in a space-focused layout."""
        # Use the first preferred floor (usually basement)
        floor = preferred_floors[0] if preferred_floors else self.min_floor
        z = floor * self.floor_height

        # For very large parking areas, consider splitting into sections
        if room.width * room.length > 1000:
            return self._place_parking_in_sections_space_focused(
                room, z, placed_rooms_by_type
            )

        # Try to place near vertical circulation if possible
        if "vertical_circulation" in placed_rooms_by_type:
            position = self._find_position_adjacent_to_type(
                room, "vertical_circulation", placed_rooms_by_type, z
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
                    allow_overlap=["vertical_circulation"],
                )
                if success:
                    return True

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

    def _place_parking_in_sections_space_focused(self, room, z, placed_rooms_by_type):
        """Place large parking areas as multiple compact sections."""
        # First adjust dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        total_area = width * length
        num_sections = min(6, max(2, int(total_area / 800)))
        area_per_section = total_area / num_sections

        # Calculate section dimensions aligned to structural grid
        grid_x, grid_y = self.structural_grid

        # Try to make sections align with full grid cells
        section_width_cells = max(
            1, round(math.sqrt(area_per_section / (grid_x * grid_y)))
        )
        section_width = section_width_cells * grid_x

        section_length_cells = max(
            1, round(area_per_section / (section_width * grid_y) / grid_y)
        )
        section_length = section_length_cells * grid_y

        # Ensure minimum dimensions
        section_width = max(section_width, 2 * grid_x)
        section_length = max(section_length, 2 * grid_y)

        sections_placed = 0

        # First, try to place sections adjacent to vertical circulation
        if "vertical_circulation" in placed_rooms_by_type:
            for circ_id in placed_rooms_by_type["vertical_circulation"]:
                if circ_id in self.spatial_grid.rooms:
                    circ_room = self.spatial_grid.rooms[circ_id]
                    cx, cy, cz = circ_room["position"]
                    cw, cl, ch = circ_room["dimensions"]

                    # Only consider vertical circulation on the same floor
                    if abs(cz - z) > 0.1:
                        continue

                    # Try adjacent positions
                    positions = [
                        (cx + cw, cy, z),  # Right
                        (cx - section_width, cy, z),  # Left
                        (cx, cy + cl, z),  # Bottom
                        (cx, cy - section_length, z),  # Top
                    ]

                    # Try each position
                    for section_id in range(1, num_sections + 1):
                        if sections_placed >= section_id:
                            continue

                        for px, py, pz in positions:
                            # Skip if out of bounds
                            if (
                                px < 0
                                or py < 0
                                or px + section_width > self.width
                                or py + section_length > self.length
                            ):
                                continue

                            # Check if position is valid
                            if self._is_valid_position(
                                px, py, pz, section_width, section_length, room.height
                            ):
                                # Create section metadata
                                section_metadata = (
                                    room.metadata.copy() if room.metadata else {}
                                )
                                section_metadata["parent_room_id"] = room.id
                                section_metadata["is_parking_section"] = True
                                section_metadata["section_number"] = section_id
                                section_metadata["of_sections"] = num_sections

                                success = self.spatial_grid.place_room(
                                    room_id=room.id * 100 + section_id,
                                    x=px,
                                    y=py,
                                    z=pz,
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

                        if sections_placed >= section_id:
                            continue

        # For remaining sections, place them adjacent to existing parking sections when possible
        for section_id in range(sections_placed + 1, num_sections + 1):
            # Try to place adjacent to already placed sections
            position = None

            # Get existing parking sections
            for i in range(1, section_id):
                existing_id = room.id * 100 + i
                if existing_id in self.spatial_grid.rooms:
                    ex_room = self.spatial_grid.rooms[existing_id]
                    ex, ey, ez = ex_room["position"]
                    ew, el, eh = ex_room["dimensions"]

                    # Try adjacent positions
                    positions = [
                        (ex + ew, ey, z),  # Right
                        (ex - section_width, ey, z),  # Left
                        (ex, ey + el, z),  # Bottom
                        (ex, ey - section_length, z),  # Top
                    ]

                    # Try each position
                    for px, py, pz in positions:
                        # Skip if out of bounds
                        if (
                            px < 0
                            or py < 0
                            or px + section_width > self.width
                            or py + section_length > self.length
                        ):
                            continue

                        # Check if position is valid
                        if self._is_valid_position(
                            px, py, pz, section_width, section_length, room.height
                        ):
                            position = (px, py, pz)
                            break

                    if position:
                        break

            # If no position found, try any valid position
            if not position:
                for x in range(0, int(self.width - section_width) + 1, int(grid_x)):
                    for y in range(
                        0, int(self.length - section_length) + 1, int(grid_y)
                    ):
                        if self._is_valid_position(
                            x, y, z, section_width, section_length, room.height
                        ):
                            position = (x, y, z)
                            break
                    if position:
                        break

            # If position found, place section
            if position:
                px, py, pz = position

                # Create section metadata
                section_metadata = room.metadata.copy() if room.metadata else {}
                section_metadata["parent_room_id"] = room.id
                section_metadata["is_parking_section"] = True
                section_metadata["section_number"] = section_id
                section_metadata["of_sections"] = num_sections

                success = self.spatial_grid.place_room(
                    room_id=room.id * 100 + section_id,
                    x=px,
                    y=py,
                    z=pz,
                    width=section_width,
                    length=section_length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=section_metadata,
                    allow_overlap=["vertical_circulation"],
                )

                if success:
                    sections_placed += 1

        # Return success if at least one section was placed
        return sections_placed > 0

    def _place_exterior_room_space_focused(
        self, room, placed_rooms_by_type, preferred_floors
    ):
        """Place a room that needs external facing on perimeter with maximum adjacency."""
        for floor in preferred_floors:
            z = floor * self.floor_height

            # Try to place with maximum adjacency first
            position = self._find_position_with_max_adjacency_for_type(
                room, room.room_type, placed_rooms_by_type, z
            )
            if position:
                x, y, z_pos = position

                # Check if position is on perimeter
                is_perimeter = (
                    abs(x) < 0.1
                    or abs(y) < 0.1
                    or abs(x + room.width - self.width) < 0.1
                    or abs(y + room.length - self.length) < 0.1
                )

                # If on perimeter, use this position
                if is_perimeter and self._is_valid_position(
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

            # If adjacency approach fails, try perimeter positions
            # Generate perimeter positions
            perimeter_positions = []

            # Add all perimeter positions on this floor
            # Top edge (y=0)
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, 0, z))

            # Bottom edge
            for x in range(
                0, int(self.width - room.width) + 1, int(self.structural_grid[0])
            ):
                perimeter_positions.append((x, self.length - room.length, z))

            # Left edge
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((0, y, z))

            # Right edge
            for y in range(
                0, int(self.length - room.length) + 1, int(self.structural_grid[1])
            ):
                perimeter_positions.append((self.width - room.width, y, z))

            # Score each perimeter position by adjacency
            scored_positions = []
            for x, y, pos_z in perimeter_positions:
                # Skip if invalid
                if not self._is_valid_position(
                    x, y, pos_z, room.width, room.length, room.height
                ):
                    continue

                # Calculate adjacency score
                score = self._calculate_adjacency_score(
                    x, y, room.width, room.length, placed_rooms_by_type
                )

                scored_positions.append(((x, y, pos_z), score))

            # Sort by score (highest first)
            scored_positions.sort(key=lambda p: p[1], reverse=True)

            # Try each position in order
            for (x, y, pos_z), _ in scored_positions:
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=pos_z,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )
                if success:
                    return True

        # Fall back to default placement
        return self._place_default_space_focused(
            room, placed_rooms_by_type, preferred_floors
        )

    def _place_default_space_focused(
        self, room, placed_rooms_by_type, preferred_floors
    ):
        """Default placement strategy for rooms with maximized adjacency."""
        for floor in preferred_floors:
            z = floor * self.floor_height

            # Try adjacency-based placement if applicable
            if room.room_type in self.adjacency_preferences:
                position = self._find_position_with_max_adjacency_for_type(
                    room, room.room_type, placed_rooms_by_type, z
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

            # Try to place adjacent to any existing room
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

        # If all preferred floors failed, try alternative floors
        other_floors = [
            f
            for f in range(self.min_floor, self.max_floor + 1)
            if f not in preferred_floors
        ]

        for floor in other_floors:
            z = floor * self.floor_height

            # Try to place adjacent to any existing room
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
                    return True

            # Try any valid position
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

    def _find_position_adjacent_to_any_room(self, room, z, placed_rooms_by_type):
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

    def _calculate_adjacency_score(self, x, y, width, length, placed_rooms_by_type):
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

    # RL-specific reward calculation modifications

    def calculate_reward(self, layout: SpatialGrid) -> float:
        """
        Calculate reward for a space-focused layout without corridors.

        Args:
            layout: The layout to evaluate

        Returns:
            float: Reward value (0.0 to 1.0)
        """
        # Calculate traditional metrics with lower weight
        space_util = layout.calculate_space_utilization()

        # Calculate compactness (new metric)
        compactness = self._calculate_compactness(layout)

        # Calculate wall sharing efficiency (new metric)
        wall_sharing = self._calculate_wall_sharing(layout)

        # Calculate adjacency satisfaction
        adjacency_satisfaction = self._calculate_adjacency_satisfaction(layout)

        # Calculate weighted reward
        reward = (
            space_util * 0.1  # Minor factor
            + compactness * 0.3  # Major factor
            + wall_sharing * 0.3  # Major factor
            + adjacency_satisfaction * 0.3  # Major factor
        )

        return reward

    def _calculate_compactness(self, layout: SpatialGrid) -> float:
        """
        Calculate the compactness of the layout (area utilization within bounding box).

        Args:
            layout: The layout to evaluate

        Returns:
            float: Compactness score (0.0 to 1.0)
        """
        # Track bounding box by floor
        floor_areas = {}
        floor_boxes = {}

        for room_id, room_data in layout.rooms.items():
            x, y, z = room_data["position"]
            w, l, h = room_data["dimensions"]

            # Determine floor
            floor = int(z / self.floor_height)

            # Initialize floor data if needed
            if floor not in floor_areas:
                floor_areas[floor] = 0
                floor_boxes[floor] = [
                    float("inf"),
                    float("inf"),
                    float("-inf"),
                    float("-inf"),
                ]

            # Add room area
            floor_areas[floor] += w * l

            # Update floor bounding box
            floor_boxes[floor][0] = min(floor_boxes[floor][0], x)
            floor_boxes[floor][1] = min(floor_boxes[floor][1], y)
            floor_boxes[floor][2] = max(floor_boxes[floor][2], x + w)
            floor_boxes[floor][3] = max(floor_boxes[floor][3], y + l)

        # Calculate compactness by floor
        floor_compactness = {}
        for floor in floor_areas:
            box = floor_boxes[floor]
            box_area = (box[2] - box[0]) * (box[3] - box[1])

            if box_area > 0:
                floor_compactness[floor] = floor_areas[floor] / box_area
            else:
                floor_compactness[floor] = 1.0

        # Overall compactness is average of floor compactness values
        if floor_compactness:
            return sum(floor_compactness.values()) / len(floor_compactness)
        else:
            return 0.0

    def _calculate_wall_sharing(self, layout: SpatialGrid) -> float:
        """
        Calculate the efficiency of wall sharing between rooms.

        Args:
            layout: The layout to evaluate

        Returns:
            float: Wall sharing efficiency (0.0 to 1.0)
        """
        total_wall_length = 0.0
        shared_wall_length = 0.0

        # Process each floor
        floor_rooms = {}

        # Group rooms by floor
        for room_id, room_data in layout.rooms.items():
            z = room_data["position"][2]
            floor = int(z / self.floor_height)

            if floor not in floor_rooms:
                floor_rooms[floor] = []

            floor_rooms[floor].append(room_data)

        # Process each floor
        for floor, rooms in floor_rooms.items():
            # Calculate all walls
            walls = []  # (start, end, room_id) format

            for room_data in rooms:
                room_id = room_data["id"]
                x, y, _ = room_data["position"]
                w, l, _ = room_data["dimensions"]

                # Add four walls
                walls.append(((x, y), (x + w, y), room_id))  # Bottom
                walls.append(((x + w, y), (x + w, y + l), room_id))  # Right
                walls.append(((x, y + l), (x + w, y + l), room_id))  # Top
                walls.append(((x, y), (x, y + l), room_id))  # Left

                # Add to total wall length
                total_wall_length += 2 * (w + l)

            # Check for shared walls
            for i, (start1, end1, id1) in enumerate(walls):
                for j, (start2, end2, id2) in enumerate(walls[i + 1 :], i + 1):
                    if id1 == id2:
                        continue  # Skip same room

                    # Check if walls are collinear and opposite
                    overlap = self._calculate_wall_overlap(start1, end1, start2, end2)
                    shared_wall_length += overlap

        # Avoid double-counting shared walls
        shared_wall_length /= 2

        # Calculate percentage of shared walls
        if total_wall_length > 0:
            return shared_wall_length / total_wall_length
        else:
            return 0.0

    def _calculate_wall_overlap(self, start1, end1, start2, end2):
        """Calculate overlap between two wall segments."""
        (x1, y1), (x2, y2) = start1, end1
        (x3, y3), (x4, y4) = start2, end2

        # Vertical walls
        if abs(x1 - x2) < 0.01 and abs(x3 - x4) < 0.01:
            if abs(x1 - x3) < 0.01:  # Same x-coordinate
                # Check for y-overlap
                y_min = max(min(y1, y2), min(y3, y4))
                y_max = min(max(y1, y2), max(y3, y4))
                return max(0, y_max - y_min)

        # Horizontal walls
        elif abs(y1 - y2) < 0.01 and abs(y3 - y4) < 0.01:
            if abs(y1 - y3) < 0.01:  # Same y-coordinate
                # Check for x-overlap
                x_min = max(min(x1, x2), min(x3, x4))
                x_max = min(max(x1, x2), max(x3, x4))
                return max(0, x_max - x_min)

        return 0  # No overlap

    def _calculate_adjacency_satisfaction(self, layout: SpatialGrid) -> float:
        """
        Calculate how well room adjacency preferences are satisfied.

        Args:
            layout: The layout to evaluate

        Returns:
            float: Adjacency satisfaction score (0.0 to 1.0)
        """
        satisfied = 0
        total = 0

        # Check adjacency for each room type
        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]

            # Skip if no preferences
            if room_type not in self.adjacency_preferences:
                continue

            preferred_types = self.adjacency_preferences[room_type]

            # Get neighbors
            neighbors = layout.get_room_neighbors(room_id)
            neighbor_types = [
                layout.rooms[n]["type"] for n in neighbors if n in layout.rooms
            ]

            # Check each preferred adjacency
            for pref_type in preferred_types:
                total += 1
                if pref_type in neighbor_types:
                    satisfied += 1

        # Calculate satisfaction ratio
        if total > 0:
            return satisfied / total
        else:
            return 1.0  # No preferences = fully satisfied

    # Update the placment strategies map
    def _initialize_placement_strategies(self) -> Dict[str, callable]:
        """Initialize room placement strategies focused on space efficiency."""
        strategies = {
            "vertical_circulation": self._place_vertical_circulation_space_focused,
            "entrance": self._place_entrance_space_focused,
            "parking": self._place_parking_space_focused,
            "guest_room": self._place_default_space_focused,
            "exterior": self._place_exterior_room_space_focused,
            "default": self._place_default_space_focused,
        }
        return strategies
