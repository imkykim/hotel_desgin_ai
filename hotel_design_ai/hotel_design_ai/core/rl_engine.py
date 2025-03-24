"""
Simplified RL engine architecture for hotel room placement.
This refactored architecture focuses on modularity, simplicity, and effectiveness.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union, Set, Callable

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


# ---------------------------------------------------------
# Core RL Components (remain largely the same)
# ---------------------------------------------------------


class RLPolicyNetwork(nn.Module):
    """Neural network that represents the policy for the RL agent."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
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


# ---------------------------------------------------------
# Room Placement Strategy Classes
# ---------------------------------------------------------


class PlacementStrategy:
    """Base class for room placement strategies."""

    def __init__(self, spatial_grid, building_config):
        self.spatial_grid = spatial_grid
        self.building_config = building_config
        self.min_floor = building_config.get("min_floor", -1)
        self.max_floor = building_config.get("max_floor", 3)
        self.floor_height = building_config.get("floor_height", 4.0)
        self.structural_grid = (
            building_config.get("structural_grid_x", 8.0),
            building_config.get("structural_grid_y", 8.0),
        )

    def place(self, room: Room) -> bool:
        """Attempt to place a room using this strategy.

        Args:
            room: The room to place

        Returns:
            bool: True if placement was successful
        """
        raise NotImplementedError("Subclasses must implement place()")

    def is_applicable(self, room: Room) -> bool:
        """Check if this strategy applies to the given room.

        Args:
            room: The room to check

        Returns:
            bool: True if this strategy applies to this room
        """
        raise NotImplementedError("Subclasses must implement is_applicable()")

    def _is_valid_position(self, x, y, z, width, length, height) -> bool:
        """Check if a position is valid for room placement."""
        # Convert to grid coordinates
        grid_x = int(x / self.spatial_grid.grid_size)
        grid_y = int(y / self.spatial_grid.grid_size)
        grid_z = int(z / self.spatial_grid.grid_size)
        grid_width = int(width / self.spatial_grid.grid_size)
        grid_length = int(length / self.spatial_grid.grid_size)
        grid_height = int(height / self.spatial_grid.grid_size)

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


class DefaultPlacementStrategy(PlacementStrategy):
    """Default placement strategy for most room types."""

    def is_applicable(self, room: Room) -> bool:
        """This is the default strategy applicable to all room types."""
        return True

    def place(self, room: Room) -> bool:
        """Standard room placement on preferred floors."""
        preferred_floors = self._get_preferred_floors(room)

        # Try center positions on preferred floors
        for floor in preferred_floors:
            z = floor * self.floor_height

            # Try center position
            center_x = max(0, (self.spatial_grid.width - room.width) / 2)
            center_y = max(0, (self.spatial_grid.length - room.length) / 2)
            center_x = (
                round(center_x / self.structural_grid[0]) * self.structural_grid[0]
            )
            center_y = (
                round(center_y / self.structural_grid[1]) * self.structural_grid[1]
            )

            if self._is_valid_position(
                center_x, center_y, z, room.width, room.length, room.height
            ):
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=center_x,
                    y=center_y,
                    z=z,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                )
                if success:
                    return True

            # Try grid-aligned positions
            grid_x, grid_y = self.structural_grid
            for x in range(
                0, int(self.spatial_grid.width - room.width) + 1, int(grid_x)
            ):
                for y in range(
                    0, int(self.spatial_grid.length - room.length) + 1, int(grid_y)
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

        # Try other floors if preferred floors fail
        other_floors = [
            f
            for f in range(self.min_floor, self.max_floor + 1)
            if f not in preferred_floors
        ]

        for floor in other_floors:
            z = floor * self.floor_height
            grid_x, grid_y = self.structural_grid

            for x in range(
                0, int(self.spatial_grid.width - room.width) + 1, int(grid_x)
            ):
                for y in range(
                    0, int(self.spatial_grid.length - room.length) + 1, int(grid_y)
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

    def _get_preferred_floors(self, room: Room) -> List[int]:
        """Get preferred floors for a room, searching multiple sources."""
        preferred_floors = []

        # 1. Check for preferred_floors attribute
        if hasattr(room, "preferred_floors") and room.preferred_floors is not None:
            if isinstance(room.preferred_floors, list):
                preferred_floors = room.preferred_floors
            else:
                preferred_floors = [room.preferred_floors]

        # 2. Or check metadata for preferred_floors
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            floors = room.metadata["preferred_floors"]
            if isinstance(floors, list):
                preferred_floors = floors
            else:
                preferred_floors = [floors]

        # 3. Or check for a floor attribute
        elif hasattr(room, "floor") and room.floor is not None:
            if isinstance(room.floor, list):
                preferred_floors = room.floor
            else:
                preferred_floors = [room.floor]

        # 4. Or check floor in metadata
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            floor = room.metadata["floor"]
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor]

        # 5. Default to all floors if nothing is specified
        if not preferred_floors:
            preferred_floors = list(range(self.min_floor, self.max_floor + 1))

        # Ensure all floor values are integers
        return [int(floor) for floor in preferred_floors if floor is not None]


class PerimeterPlacementStrategy(PlacementStrategy):
    """Strategy for rooms that should be placed on the perimeter."""

    def is_applicable(self, room: Room) -> bool:
        """Check if this room needs exterior access."""
        exterior_room_types = [
            "entrance",
            "lobby",
            "restaurant",
            "retail",
            "guest_room",
        ]
        return room.room_type in exterior_room_types

    def place(self, room: Room) -> bool:
        """Place room on the perimeter."""
        preferred_floors = self._get_preferred_floors(room)

        for floor in preferred_floors:
            z = floor * self.floor_height

            # For entrance, prefer front perimeter
            if room.room_type == "entrance":
                # Try front edge first (y = 0)
                for x in range(
                    0,
                    int(self.spatial_grid.width - room.width) + 1,
                    int(self.structural_grid[0]),
                ):
                    if self._is_valid_position(
                        x, 0, z, room.width, room.length, room.height
                    ):
                        success = self.spatial_grid.place_room(
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
                        if success:
                            return True

            # Try all perimeter positions
            perimeter_positions = []

            # Back edge
            back_y = self.spatial_grid.length - room.length
            for x in range(
                0,
                int(self.spatial_grid.width - room.width) + 1,
                int(self.structural_grid[0]),
            ):
                perimeter_positions.append((x, back_y))

            # Front edge (for non-entrance types)
            if room.room_type != "entrance":
                for x in range(
                    0,
                    int(self.spatial_grid.width - room.width) + 1,
                    int(self.structural_grid[0]),
                ):
                    perimeter_positions.append((x, 0))

            # Left edge
            for y in range(
                0,
                int(self.spatial_grid.length - room.length) + 1,
                int(self.structural_grid[1]),
            ):
                perimeter_positions.append((0, y))

            # Right edge
            right_x = self.spatial_grid.width - room.width
            for y in range(
                0,
                int(self.spatial_grid.length - room.length) + 1,
                int(self.structural_grid[1]),
            ):
                perimeter_positions.append((right_x, y))

            # Try all perimeter positions
            random.shuffle(perimeter_positions)
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

        return False

    def _get_preferred_floors(self, room: Room) -> List[int]:
        """Get preferred floors for a room, searching multiple sources."""
        preferred_floors = []

        # 1. Check for preferred_floors attribute
        if hasattr(room, "preferred_floors") and room.preferred_floors is not None:
            if isinstance(room.preferred_floors, list):
                preferred_floors = room.preferred_floors
            else:
                preferred_floors = [room.preferred_floors]

        # 2. Or check metadata for preferred_floors
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            floors = room.metadata["preferred_floors"]
            if isinstance(floors, list):
                preferred_floors = floors
            else:
                preferred_floors = [floors]

        # 3. Or check for a floor attribute
        elif hasattr(room, "floor") and room.floor is not None:
            if isinstance(room.floor, list):
                preferred_floors = room.floor
            else:
                preferred_floors = [room.floor]

        # 4. Or check floor in metadata
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            floor = room.metadata["floor"]
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor]

        # 5. Default to all floors if nothing is specified
        if not preferred_floors:
            if room.room_type == "entrance":
                preferred_floors = [
                    max(0, self.min_floor)
                ]  # Entrance usually on ground floor
            else:
                preferred_floors = list(range(self.min_floor, self.max_floor + 1))

        # Ensure all floor values are integers
        return [int(floor) for floor in preferred_floors if floor is not None]


class LargeRoomStrategy(PlacementStrategy):
    """Strategy for large rooms (like ballrooms, meeting spaces, parking)."""

    def is_applicable(self, room: Room) -> bool:
        """Check if this is a large room (> 300 m²)."""
        large_area = room.width * room.length > 300
        large_types = ["ballroom", "meeting_room", "parking", "pool"]
        return large_area or room.room_type in large_types

    def place(self, room: Room) -> bool:
        """Place large room with adaptive sizing if needed."""
        if room.room_type == "parking" and room.width * room.length > 1000:
            return self._place_parking_sections(room)

        preferred_floors = self._get_preferred_floors(room)
        original_width, original_length = room.width, room.length

        # Try with original dimensions on preferred floors
        for floor in preferred_floors:
            z = floor * self.floor_height

            # Try center first (architecturally preferable)
            center_x = max(0, (self.spatial_grid.width - room.width) / 2)
            center_y = max(0, (self.spatial_grid.length - room.length) / 2)
            center_x = (
                round(center_x / self.structural_grid[0]) * self.structural_grid[0]
            )
            center_y = (
                round(center_y / self.structural_grid[1]) * self.structural_grid[1]
            )

            if self._is_valid_position(
                center_x, center_y, z, room.width, room.length, room.height
            ):
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=center_x,
                    y=center_y,
                    z=z,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata,
                    allow_overlap=(
                        ["vertical_circulation"]
                        if room.room_type == "parking"
                        else None
                    ),
                )
                if success:
                    return True

            # Try grid search
            grid_x, grid_y = self.structural_grid
            for x in range(
                0, int(self.spatial_grid.width - room.width) + 1, int(grid_x)
            ):
                for y in range(
                    0, int(self.spatial_grid.length - room.length) + 1, int(grid_y)
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
                            allow_overlap=(
                                ["vertical_circulation"]
                                if room.room_type == "parking"
                                else None
                            ),
                        )
                        if success:
                            return True

        # If original dimensions don't work, try alternative shapes with same area
        area = original_width * original_length
        grid_x, grid_y = self.structural_grid

        alternative_shapes = []

        # More square shape
        side = np.sqrt(area)
        width1 = round(side / grid_x) * grid_x
        length1 = round(area / width1 / grid_y) * grid_y
        alternative_shapes.append((width1, length1))

        # Wide shape
        width2 = round(side * 1.5 / grid_x) * grid_x
        length2 = round(area / width2 / grid_y) * grid_y
        alternative_shapes.append((width2, length2))

        # Tall shape
        width3 = round(side / 1.5 / grid_x) * grid_x
        length3 = round(area / width3 / grid_y) * grid_y
        alternative_shapes.append((width3, length3))

        # Filter shapes that fit within building
        valid_shapes = [
            (w, l)
            for w, l in alternative_shapes
            if w > 0
            and l > 0
            and w <= self.spatial_grid.width
            and l <= self.spatial_grid.length
        ]

        # Try each alternative shape
        for width, length in valid_shapes:
            # Temporarily update room dimensions
            room.width, room.length = width, length

            for floor in preferred_floors:
                z = floor * self.floor_height

                for x in range(
                    0, int(self.spatial_grid.width - width) + 1, int(grid_x)
                ):
                    for y in range(
                        0, int(self.spatial_grid.length - length) + 1, int(grid_y)
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
                                allow_overlap=(
                                    ["vertical_circulation"]
                                    if room.room_type == "parking"
                                    else None
                                ),
                            )
                            if success:
                                return True

        # Reset dimensions
        room.width, room.length = original_width, original_length

        # Last resort: force placement
        for floor in preferred_floors:
            z = floor * self.floor_height
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
                force_placement=True,
            )
            if success:
                return True

        return False

    def _place_parking_sections(self, room: Room) -> bool:
        """Place large parking as multiple sections."""
        preferred_floors = self._get_preferred_floors(room)

        # Calculate section sizes
        total_area = room.width * room.length
        grid_x, grid_y = self.structural_grid

        # More sections for large areas
        num_sections = min(6, max(2, int(total_area / 800)))
        area_per_section = total_area / num_sections

        # Calculate section dimensions
        section_width = round(np.sqrt(area_per_section) / grid_x) * grid_x
        section_length = round(area_per_section / section_width / grid_y) * grid_y

        # Distribute sections across floors
        sections_per_floor = {}
        for i, floor in enumerate(preferred_floors):
            sections_this_floor = max(1, num_sections // len(preferred_floors))
            if i == len(preferred_floors) - 1:
                remaining = num_sections - sum(sections_per_floor.values())
                sections_this_floor = remaining
            sections_per_floor[floor] = sections_this_floor

        # Place sections
        sections_placed = 0

        for floor, num_to_place in sections_per_floor.items():
            z = floor * self.floor_height

            # Build positions for this floor
            positions = []
            for x in range(
                0, int(self.spatial_grid.width - section_width) + 1, int(grid_x)
            ):
                for y in range(
                    0, int(self.spatial_grid.length - section_length) + 1, int(grid_y)
                ):
                    positions.append((x, y))

            # Shuffle positions
            random.shuffle(positions)

            # Place sections on this floor
            for i in range(num_to_place):
                section_id = room.id * 100 + sections_placed + 1

                # Create section metadata
                section_metadata = room.metadata.copy() if room.metadata else {}
                section_metadata["parent_room_id"] = room.id
                section_metadata["is_parking_section"] = True
                section_metadata["section_number"] = sections_placed + 1
                section_metadata["of_sections"] = num_sections

                # Try positions
                placed = False
                for x, y in positions:
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
                        placed = True
                        break

                # If normal placement failed, try force placement
                if not placed:
                    x = grid_x
                    y = grid_y * (sections_placed + 1)
                    y = min(y, self.spatial_grid.length - section_length)

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
                        force_placement=True,
                    )

                    if success:
                        sections_placed += 1

        return sections_placed > 0

    def _get_preferred_floors(self, room: Room) -> List[int]:
        """Get preferred floors for large rooms."""
        preferred_floors = []

        # 1. Check for preferred_floors attribute
        if hasattr(room, "preferred_floors") and room.preferred_floors is not None:
            if isinstance(room.preferred_floors, list):
                preferred_floors = room.preferred_floors
            else:
                preferred_floors = [room.preferred_floors]

        # 2. Or check metadata for preferred_floors
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            floors = room.metadata["preferred_floors"]
            if isinstance(floors, list):
                preferred_floors = floors
            else:
                preferred_floors = [floors]

        # 3. Or check for a floor attribute
        elif hasattr(room, "floor") and room.floor is not None:
            if isinstance(room.floor, list):
                preferred_floors = room.floor
            else:
                preferred_floors = [room.floor]

        # 4. Or check floor in metadata
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            floor = room.metadata["floor"]
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor]

        # 5. Default to room-type specific floors if nothing is specified
        if not preferred_floors:
            if room.room_type == "parking":
                preferred_floors = list(
                    range(self.min_floor, min(0, self.max_floor) + 1)
                )  # Basement floors
            elif room.room_type == "ballroom" or room.room_type == "meeting_room":
                preferred_floors = [max(0, self.min_floor)]  # Ground floor
            else:
                preferred_floors = list(range(self.min_floor, self.max_floor + 1))

        # Ensure all floor values are integers
        return [int(floor) for floor in preferred_floors if floor is not None]


class CirculationStrategy(PlacementStrategy):
    """Special strategy for vertical circulation that spans all floors."""

    def is_applicable(self, room: Room) -> bool:
        """Check if this is a vertical circulation room."""
        return room.room_type == "vertical_circulation"

    def place(self, room: Room) -> bool:
        """Place vertical circulation spanning all floors."""
        # Calculate height to span all floors
        total_floors = self.max_floor - self.min_floor + 1
        total_height = total_floors * self.floor_height

        # Start z at lowest floor
        z = self.min_floor * self.floor_height

        # Update metadata
        if not room.metadata:
            room.metadata = {}
        room.metadata["is_core"] = True
        room.metadata["spans_floors"] = list(range(self.min_floor, self.max_floor + 1))

        # Try center position first
        center_x = (self.spatial_grid.width - room.width) / 2
        center_y = (self.spatial_grid.length - room.length) / 2
        center_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        center_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]

        if self._is_valid_position(
            center_x, center_y, z, room.width, room.length, total_height
        ):
            success = self.spatial_grid.place_room(
                room_id=room.id,
                x=center_x,
                y=center_y,
                z=z,
                width=room.width,
                length=room.length,
                height=total_height,
                room_type=room.room_type,
                metadata=room.metadata,
            )
            if success:
                return True

        # Try grid search
        grid_x, grid_y = self.structural_grid
        for x in range(0, int(self.spatial_grid.width - room.width) + 1, int(grid_x)):
            for y in range(
                0, int(self.spatial_grid.length - room.length) + 1, int(grid_y)
            ):
                if self._is_valid_position(
                    x, y, z, room.width, room.length, total_height
                ):
                    success = self.spatial_grid.place_room(
                        room_id=room.id,
                        x=x,
                        y=y,
                        z=z,
                        width=room.width,
                        length=room.length,
                        height=total_height,
                        room_type=room.room_type,
                        metadata=room.metadata,
                    )
                    if success:
                        return True

        # Force placement as last resort
        success = self.spatial_grid.place_room(
            room_id=room.id,
            x=grid_x,
            y=grid_y,
            z=z,
            width=room.width,
            length=room.length,
            height=total_height,
            room_type=room.room_type,
            metadata=room.metadata,
            force_placement=True,
        )
        return success


class AdjacencyPlacementStrategy(PlacementStrategy):
    """Strategy that places rooms adjacent to existing rooms of specific types."""

    def __init__(self, spatial_grid, building_config):
        super().__init__(spatial_grid, building_config)
        # Define adjacency requirements
        self.adjacency_requirements = {
            "kitchen": ["restaurant"],
            "restaurant": ["lobby", "kitchen"],
            "meeting_room": ["pre_function", "lobby"],
            "pre_function": ["meeting_room", "ballroom"],
            "ballroom": ["pre_function"],
            "lounge": ["lobby", "ballroom"],
            "food_service": ["restaurant"],
            "retail": ["lobby"],
            "service_area": ["lobby", "back_of_house"],
            "back_of_house": ["service_area"],
        }

    def is_applicable(self, room: Room) -> bool:
        """Check if this room has adjacency requirements."""
        return room.room_type in self.adjacency_requirements

    def place(self, room: Room) -> bool:
        """Place room adjacent to its required neighbors."""
        required_adjacencies = self.adjacency_requirements.get(room.room_type, [])
        if not required_adjacencies:
            return False

        preferred_floors = self._get_preferred_floors(room)

        # For each adjacency requirement, find rooms of that type
        for adj_type in required_adjacencies:
            adj_rooms = []
            for room_id, room_data in self.spatial_grid.rooms.items():
                if room_data["type"] == adj_type:
                    adj_rooms.append(room_data)

            # For each adjacent room, try to place next to it
            for adj_room in adj_rooms:
                x, y, z = adj_room["position"]
                w, l, h = adj_room["dimensions"]

                # Calculate what floor this room is on
                floor = int(z / self.floor_height)

                # Skip if not on a preferred floor
                if floor not in preferred_floors:
                    continue

                # Try positions around the adjacent room
                positions = [
                    (x + w, y, z),  # Right
                    (x - room.width, y, z),  # Left
                    (x, y + l, z),  # Behind
                    (x, y - room.length, z),  # In front
                ]

                for pos_x, pos_y, pos_z in positions:
                    if self._is_valid_position(
                        pos_x, pos_y, pos_z, room.width, room.length, room.height
                    ):
                        success = self.spatial_grid.place_room(
                            room_id=room.id,
                            x=pos_x,
                            y=pos_y,
                            z=pos_z,
                            width=room.width,
                            length=room.length,
                            height=room.height,
                            room_type=room.room_type,
                            metadata=room.metadata,
                        )
                        if success:
                            return True

        # If adjacency placement fails, fall back to default placement
        return DefaultPlacementStrategy(self.spatial_grid, self.building_config).place(
            room
        )

    def _get_preferred_floors(self, room: Room) -> List[int]:
        """Get preferred floors for adjacent placement."""
        preferred_floors = []

        # 1. Check for preferred_floors attribute
        if hasattr(room, "preferred_floors") and room.preferred_floors is not None:
            if isinstance(room.preferred_floors, list):
                preferred_floors = room.preferred_floors
            else:
                preferred_floors = [room.preferred_floors]

        # 2. Or check metadata for preferred_floors
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            floors = room.metadata["preferred_floors"]
            if isinstance(floors, list):
                preferred_floors = floors
            else:
                preferred_floors = [floors]

        # 3. Or check for a floor attribute
        elif hasattr(room, "floor") and room.floor is not None:
            if isinstance(room.floor, list):
                preferred_floors = room.floor
            else:
                preferred_floors = [room.floor]

        # 4. Or check floor in metadata
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            floor = room.metadata["floor"]
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor]

        # 5. If still no preferred floors, check existing rooms of required types
        if not preferred_floors:
            required_adjacencies = self.adjacency_requirements.get(room.room_type, [])

            found_floors = set()
            for adj_type in required_adjacencies:
                for room_id, room_data in self.spatial_grid.rooms.items():
                    if room_data["type"] == adj_type:
                        z = room_data["position"][2]
                        found_floors.add(int(z / self.floor_height))

            if found_floors:
                preferred_floors = list(found_floors)
            else:
                preferred_floors = list(range(self.min_floor, self.max_floor + 1))

        # Ensure all floor values are integers
        return [int(floor) for floor in preferred_floors if floor is not None]


class ForcePlacementStrategy(PlacementStrategy):
    """Last resort strategy that forces placement even if it overlaps."""

    def is_applicable(self, room: Room) -> bool:
        """This strategy applies to all rooms as a last resort."""
        return True

    def place(self, room: Room) -> bool:
        """Force placement of a room."""
        preferred_floors = self._get_preferred_floors(room)
        grid_x, grid_y = self.structural_grid

        for floor in preferred_floors:
            z = floor * self.floor_height

            # Try several positions
            positions = [
                (grid_x, grid_y),  # Corner
                (self.spatial_grid.width / 2 - room.width / 2, grid_y),  # Top center
                (grid_x, self.spatial_grid.length / 2 - room.length / 2),  # Left center
            ]

            for x, y in positions:
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
                    force_placement=True,
                )
                if success:
                    return True

        # If all else fails, try a reduced size version
        original_width, original_length = room.width, room.length
        reduced_width = original_width * 0.8
        reduced_length = original_length * 0.8

        # Update room dimensions
        room.width, room.length = reduced_width, reduced_length

        for floor in preferred_floors:
            z = floor * self.floor_height

            success = self.spatial_grid.place_room(
                room_id=room.id,
                x=grid_x,
                y=grid_y,
                z=z,
                width=reduced_width,
                length=reduced_length,
                height=room.height,
                room_type=room.room_type,
                metadata=room.metadata,
                force_placement=True,
            )
            if success:
                return True

        # Reset dimensions
        room.width, room.length = original_width, original_length

        return False

    def _get_preferred_floors(self, room: Room) -> List[int]:
        """Get preferred floors, with fallbacks."""
        preferred_floors = []

        # Check all possible sources
        if hasattr(room, "preferred_floors") and room.preferred_floors:
            if isinstance(room.preferred_floors, list):
                preferred_floors = room.preferred_floors
            else:
                preferred_floors = [room.preferred_floors]
        elif (
            hasattr(room, "metadata")
            and room.metadata
            and "preferred_floors" in room.metadata
        ):
            floors = room.metadata["preferred_floors"]
            if isinstance(floors, list):
                preferred_floors = floors
            else:
                preferred_floors = [floors]
        elif hasattr(room, "floor") and room.floor is not None:
            if isinstance(room.floor, list):
                preferred_floors = room.floor
            else:
                preferred_floors = [room.floor]
        elif hasattr(room, "metadata") and room.metadata and "floor" in room.metadata:
            floor = room.metadata["floor"]
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor]

        # If still no preferred floors, use defaults
        if not preferred_floors:
            # Prioritize ground floor for most public rooms
            public_types = ["lobby", "entrance", "restaurant", "retail"]
            service_types = [
                "back_of_house",
                "mechanical",
                "maintenance",
                "service_area",
                "parking",
            ]

            if room.room_type in public_types:
                preferred_floors = [max(0, self.min_floor)]
            elif room.room_type in service_types:
                if self.min_floor < 0:
                    preferred_floors = [self.min_floor]  # Basement
                else:
                    preferred_floors = [self.min_floor]  # Ground floor
            else:
                # Try all floors with preference for ground floor
                preferred_floors = [max(0, self.min_floor)] + [
                    f
                    for f in range(self.min_floor, self.max_floor + 1)
                    if f != max(0, self.min_floor)
                ]

        # Ensure all floor values are integers
        return [int(floor) for floor in preferred_floors if floor is not None]


# ---------------------------------------------------------
# Room Placement Manager
# ---------------------------------------------------------


class RoomPlacementManager:
    """Manages room placement using a chain of strategies."""

    def __init__(self, spatial_grid, building_config):
        self.spatial_grid = spatial_grid
        self.building_config = building_config

        # Initialize placement strategies
        self.strategies = [
            CirculationStrategy(spatial_grid, building_config),
            PerimeterPlacementStrategy(spatial_grid, building_config),
            AdjacencyPlacementStrategy(spatial_grid, building_config),
            LargeRoomStrategy(spatial_grid, building_config),
            DefaultPlacementStrategy(spatial_grid, building_config),
            ForcePlacementStrategy(spatial_grid, building_config),
        ]

    def place_room(self, room: Room) -> bool:
        """
        Place a room using the appropriate strategy.

        Args:
            room: Room to place

        Returns:
            bool: True if placement was successful
        """
        # Try each strategy in order
        for strategy in self.strategies:
            if strategy.is_applicable(room):
                success = strategy.place(room)
                if success:
                    return True

        return False


# ---------------------------------------------------------
# Main RL Engine Class
# ---------------------------------------------------------


class RLEngine:
    """
    Reinforcement Learning engine for hotel layout generation.
    Simplified architecture for better maintainability.
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
        self.width, self.length, self.height = bounding_box
        self.grid_size = grid_size
        self.structural_grid = structural_grid
        self.building_config = building_config or {
            "floor_height": 4.0,
            "min_floor": -1,
            "max_floor": int(self.height / 4.0),
            "structural_grid_x": structural_grid[0],
            "structural_grid_y": structural_grid[1],
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

        # Floor range properties
        self.min_floor = self.building_config.get("min_floor", -1)
        self.max_floor = self.building_config.get("max_floor", 3)
        self.floor_height = self.building_config.get("floor_height", 4.0)

        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        # Memory buffer for experience replay
        self.memory = deque(maxlen=memory_size)

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

        # State and action dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()

        # Neural networks
        self.policy_network = RLPolicyNetwork(self.state_dim, self.action_dim)
        self.target_network = RLPolicyNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Fixed elements (set by user)
        self.fixed_elements = {}  # room_id -> position

        # Training metrics
        self.training_iterations = 0
        self.average_reward = 0

        # Create room placement manager
        self.placement_manager = RoomPlacementManager(
            self.spatial_grid, self.building_config
        )

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

    def update_fixed_elements(
        self, fixed_elements: Dict[int, Tuple[float, float, float]]
    ):
        """
        Update the fixed elements (user placed/modified rooms).

        Args:
            fixed_elements: Dictionary mapping room IDs to positions
        """
        self.fixed_elements = fixed_elements.copy()

    def clear_non_fixed_elements(self):
        """Remove all non-fixed elements from the spatial grid"""
        all_room_ids = list(self.spatial_grid.rooms.keys())
        for room_id in all_room_ids:
            if room_id not in self.fixed_elements:
                self.spatial_grid.remove_room(room_id)

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout.

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

        # Set up room priorities
        room_type_priority = {
            "entrance": 10,
            "lobby": 9,
            "vertical_circulation": 8,
            "restaurant": 7,
            "kitchen": 7,
            "meeting_room": 6,
            "ballroom": 6,
            "guest_room": 5,
            "service_area": 4,
            "back_of_house": 3,
            "mechanical": 2,
            "parking": 3,
        }

        # Sort rooms by priority and size
        sorted_rooms = sorted(
            rooms,
            key=lambda r: (
                # Fixed rooms first
                r.id not in self.fixed_elements,
                # Then by architectural priority
                -room_type_priority.get(r.room_type, 0),
                # Then by size (large rooms first)
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

        # Place each room
        for room in rooms_to_place:
            # Track stats by room type
            placement_stats["by_type"][room.room_type]["total"] += 1

            # Place room using the placement manager
            success = self.placement_manager.place_room(room)

            # Update statistics
            if success:
                placement_stats["placed"] += 1
                placement_stats["by_type"][room.room_type]["placed"] += 1
            else:
                placement_stats["failed"] += 1
                placement_stats["by_type"][room.room_type]["failed"] += 1
                print(
                    f"Failed to place {room.room_type} (id={room.id}, area={room.width * room.length:.1f}m²)"
                )

        # Print statistics
        print("\nRoom placement statistics:")
        print(f"Total rooms: {placement_stats['total']}")

        if placement_stats["total"] > 0:
            success_rate = placement_stats["placed"] / placement_stats["total"] * 100
            print(
                f"Successfully placed: {placement_stats['placed']} ({success_rate:.1f}%)"
            )
            print(
                f"Failed to place: {placement_stats['failed']} ({placement_stats['failed']/placement_stats['total']*100:.1f}%)"
            )

        print("\nPlacement by room type:")
        for room_type, stats in placement_stats["by_type"].items():
            if stats["total"] > 0:
                success_rate = stats["placed"] / stats["total"] * 100
                print(
                    f"  {room_type}: {stats['placed']}/{stats['total']} ({success_rate:.1f}%)"
                )

        return self.spatial_grid

    def calculate_reward(self, layout: SpatialGrid) -> float:
        """
        Calculate a reward for a layout based on architectural principles.

        Args:
            layout: The layout to evaluate

        Returns:
            float: Reward value
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

        # Total reward
        total_reward = (
            utilization_reward + adjacency_reward + floor_reward + alignment_reward
        ) / 11.0  # Normalize

        return total_reward

    def _calculate_adjacency_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward component for adjacency relationships."""
        adjacency_score = 0.0
        total_relationships = 0

        # Desired adjacency relationships
        adjacency_preferences = {
            "entrance": ["lobby", "vertical_circulation"],
            "lobby": ["entrance", "restaurant", "vertical_circulation", "meeting_room"],
            "vertical_circulation": ["lobby", "guest_room", "service_area"],
            "restaurant": ["lobby", "kitchen"],
            "kitchen": ["restaurant", "service_area"],
            "meeting_room": ["lobby", "pre_function", "vertical_circulation"],
            "pre_function": ["meeting_room", "ballroom"],
            "ballroom": ["pre_function", "lobby"],
            "lounge": ["lobby", "ballroom"],
            "guest_room": ["vertical_circulation"],
            "service_area": ["vertical_circulation", "back_of_house"],
            "back_of_house": ["service_area", "kitchen"],
            "food_service": ["restaurant", "lobby"],
            "retail": ["lobby"],
        }

        # Check each room's adjacencies
        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]
            preferred = adjacency_preferences.get(room_type, [])
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

        # Floor preferences
        floor_preferences = {
            "entrance": [0],
            "lobby": [0],
            "restaurant": [0],
            "kitchen": [0],
            "meeting_room": [0],
            "ballroom": [0],
            "pre_function": [0],
            "lounge": [0],
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
