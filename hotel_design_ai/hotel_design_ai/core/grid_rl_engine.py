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

    def place_room_by_constraints(self, room, placed_rooms_by_type):
        """
        Place a room according to architectural constraints.
        Enhanced to use grid-aligned dimensions.
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
        if room.room_type in self.placement_strategies:
            strategy = self.placement_strategies[room.room_type]
        elif self._needs_exterior_placement(room):
            strategy = self.placement_strategies["exterior"]
        else:
            strategy = self.placement_strategies["default"]

        # Try to place using the selected strategy
        success = strategy(room, placed_rooms_by_type, preferred_floors)

        # Restore original dimensions regardless of outcome
        room.width = original_width
        room.length = original_length

        return success

    def _place_vertical_circulation(self, room, placed_rooms_by_type, preferred_floors):
        """Special placement for vertical circulation elements."""

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

    def update_fixed_elements(self, fixed_positions: Dict[int, Any]):
        """
        Update the engine's fixed elements dictionary.

        Args:
            fixed_positions: Dict mapping room IDs to positions
        """
        self.fixed_elements = fixed_positions
        print(f"Updated fixed elements: {len(fixed_positions)} positions")

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
        Generate a hotel layout based on configurations.
        Enhanced to prioritize fixed position rooms and place them exactly.

        Args:
            rooms: List of Room objects to place

        Returns:
            SpatialGrid: The generated layout
        """
        # Clear non-fixed elements
        self.clear_non_fixed_elements()

        # Initialize placed_rooms_by_type as defaultdict instead of regular dict
        # This ensures a key will be automatically initialized when accessed
        placed_rooms_by_type = defaultdict(list)

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
                    force_placement=True,  # Force fixed elements
                )
                # Add to placed_rooms_by_type (no need to check if key exists with defaultdict)
                placed_rooms_by_type[room.room_type].append(room.id)

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
        failed_fixed_rooms = []

        for room in fixed_rooms:
            print(
                f"\nPlacing fixed room: id={room.id}, type={room.room_type}, name={room.name}, pos={room.position}"
            )
            # Use force placement to ensure fixed positions are respected
            success = self.place_fixed_room_exactly(room, placed_rooms_by_type)

            if success:
                # No need to check if key exists with defaultdict
                placed_rooms_by_type[room.room_type].append(room.id)
                print(
                    f"  ✓ Successfully placed fixed room id={room.id} at {room.position}"
                )
            else:
                failed_fixed_rooms.append(room)
                print(f"  ✗ Failed to place fixed room id={room.id} at {room.position}")

        # Sort rooms by priority
        sorted_rooms = sorted(
            unfixed_rooms,
            key=lambda r: (
                # Then by priority
                -self.placement_priorities.get(r.room_type, 0),
                # Then by size (larger rooms first)
                -(r.width * r.length),
            ),
        )

        # Statistics tracking
        placement_stats = {
            "total": len(
                unfixed_rooms
            ),  # FIXED: Changed from rooms_to_place to unfixed_rooms
            "placed": 0,
            "failed": 0,
            "by_type": defaultdict(lambda: {"total": 0, "placed": 0, "failed": 0}),
        }

        # Add existing rooms from spatial grid to placed_rooms_by_type
        for room_id, room_data in self.spatial_grid.rooms.items():
            room_type = room_data["type"]
            # With defaultdict, no need to check if key exists
            if room_id not in placed_rooms_by_type[room_type]:
                placed_rooms_by_type[room_type].append(room_id)

        # Place each room
        for room in sorted_rooms:
            # Track statistics
            placement_stats["by_type"][room.room_type]["total"] += 1

            # Place room
            success = self.place_room_by_constraints(room, placed_rooms_by_type)

            # Update statistics
            if success:
                placement_stats["placed"] += 1
                placement_stats["by_type"][room.room_type]["placed"] += 1
                # With defaultdict, no need to check if key exists
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
