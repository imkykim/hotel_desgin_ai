"""
Grid-based placement engine for hotel layout generation.
This engine places rooms by aligning them to the structural grid,
making better use of space and creating more efficient layouts.
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
import numpy as np
import random
import math

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room
from hotel_design_ai.core.base_engine import BaseEngine


class GridPlacementEngine(BaseEngine):
    """
    Grid-based placement engine for hotel layouts.
    Places rooms to align with the structural grid for better space utilization.
    """

    def __init__(
        self,
        bounding_box: Tuple[float, float, float],
        grid_size: float = 1.0,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        grid_fraction: float = 0.5,  # Fraction of grid to align to (0.5 = half grid)
        building_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the grid placement engine.

        Args:
            bounding_box: (width, length, height) of buildable area in meters
            grid_size: Size of spatial grid cells in meters
            structural_grid: (x_spacing, y_spacing) of structural grid in meters
            grid_fraction: Smallest fraction of grid to align to (default: 0.5)
            building_config: Building configuration parameters
        """
        # Initialize the base engine
        super().__init__(bounding_box, grid_size, structural_grid, building_config)

        # Store the grid fraction
        self.grid_fraction = grid_fraction
        self.snap_size_x = structural_grid[0] * grid_fraction
        self.snap_size_y = structural_grid[1] * grid_fraction

        # Initialize room placement priorities (similar to rule engine)
        self._init_placement_priorities()

        # Initialize floor preferences dynamically
        self._init_floor_preferences()

        # Track grid occupancy
        self._init_grid_occupancy()

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
            "mechanical": 2,
            "parking": 3,
        }

        # Add any additional room types from configuration
        if (
            hasattr(self, "building_config")
            and "room_priorities" in self.building_config
        ):
            self.placement_priorities.update(self.building_config["room_priorities"])

    def _init_floor_preferences(self):
        """Initialize floor preferences for room types"""
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
            min_floor = self.min_floor
            max_floor = self.max_floor

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

    def _init_grid_occupancy(self):
        """
        Initialize grid occupancy tracking.
        This tracks which grid cells are occupied on each floor.
        """
        # Calculate number of grid cells in each dimension
        grid_cells_x = math.ceil(
            self.width / self.structural_grid[0] / self.grid_fraction
        )
        grid_cells_y = math.ceil(
            self.length / self.structural_grid[1] / self.grid_fraction
        )

        # Create occupancy grid for each floor
        self.grid_occupancy = {}
        for floor in range(self.min_floor, self.max_floor + 1):
            self.grid_occupancy[floor] = np.zeros(
                (grid_cells_x, grid_cells_y), dtype=bool
            )

        # Track adjacency preferences for room placement
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

    def adjust_room_dimensions_to_grid(self, room: Room) -> Tuple[float, float]:
        """
        Adjust room dimensions to align with the grid.
        Prioritizes full grid alignment over fractional grid.

        Args:
            room: Room object with required area

        Returns:
            Tuple[float, float]: Adjusted (width, length) in meters
        """
        # Get structural grid dimensions
        grid_x, grid_y = self.structural_grid

        # Calculate grid cell area
        grid_cell_area = grid_x * grid_y

        # Determine required area (use existing or calculate from width * length)
        required_area = getattr(room, "min_area", room.width * room.length)

        # Calculate number of grid cells needed
        grid_cells_needed = required_area / grid_cell_area

        # First try to use full grid cells
        full_grid_cells = math.ceil(grid_cells_needed)

        # Try different full-grid combinations
        best_width_cells = 1
        best_length_cells = full_grid_cells
        best_area_ratio = (
            best_width_cells * best_length_cells * grid_cell_area
        ) / required_area

        # Find the most square-like dimensions that use only full grid cells
        for w in range(1, int(math.sqrt(full_grid_cells)) + 1):
            l = math.ceil(full_grid_cells / w)
            area = w * l * grid_cell_area
            ratio = area / required_area

            # Prefer dimensions that are more square-like and don't waste too much space
            if ratio < best_area_ratio or (
                abs(ratio - best_area_ratio) < 0.1
                and abs(w - l) < abs(best_width_cells - best_length_cells)
            ):
                best_width_cells = w
                best_length_cells = l
                best_area_ratio = ratio

        # If using full grid cells would waste too much space (>25%), try fractional grid
        if best_area_ratio > 1.25 and self.grid_fraction < 1.0:
            # Round up to the nearest fraction of grid
            grid_cells_rounded = (
                math.ceil(grid_cells_needed / self.grid_fraction) * self.grid_fraction
            )

            # Calculate optimal grid-aligned dimensions with fractional grid
            width_cells = math.sqrt(grid_cells_rounded)
            length_cells = grid_cells_rounded / width_cells

            # Round to nearest grid fraction, prioritizing full grid if close
            # For width
            full_grid_width = round(width_cells)
            if abs(full_grid_width - width_cells) <= 0.2:  # Within 20% of full grid
                width_cells = full_grid_width
            else:
                width_cells = (
                    round(width_cells / self.grid_fraction) * self.grid_fraction
                )

            # Ensure at least half a grid for width
            if width_cells < 0.5:
                width_cells = 0.5

            # Recalculate length based on width to maintain area
            length_cells = grid_cells_rounded / width_cells

            # For length - prioritize full grid if close
            full_grid_length = round(length_cells)
            if abs(full_grid_length - length_cells) <= 0.2:  # Within 20% of full grid
                length_cells = full_grid_length
            else:
                length_cells = (
                    round(length_cells / self.grid_fraction) * self.grid_fraction
                )

            # Ensure at least half a grid for length
            if length_cells < 0.5:
                length_cells = 0.5
                width_cells = grid_cells_rounded / length_cells

                # Check again if width can be full grid
                full_grid_width = round(width_cells)
                if abs(full_grid_width - width_cells) <= 0.2:
                    width_cells = full_grid_width
                else:
                    width_cells = (
                        round(width_cells / self.grid_fraction) * self.grid_fraction
                    )
        else:
            # Use the best full-grid dimensions we found
            width_cells = best_width_cells
            length_cells = best_length_cells

        # Convert from grid cells to meters
        width = width_cells * grid_x
        length = length_cells * grid_y

        # Make sure we meet minimum dimensions if specified
        if hasattr(room, "width") and width < room.width:
            # First try to use full grid if possible
            full_grid_width = math.ceil(room.width / grid_x) * grid_x
            fractional_width = (
                math.ceil(room.width / self.snap_size_x) * self.snap_size_x
            )

            # Use full grid width if it's not too much bigger
            if full_grid_width <= 1.25 * room.width:
                width = full_grid_width
            else:
                width = fractional_width

            # Recalculate length to maintain area
            length = required_area / width

            # Try full grid length first
            full_grid_length = round(length / grid_y) * grid_y
            fractional_length = round(length / self.snap_size_y) * self.snap_size_y

            # Use full grid length if close enough
            if abs(full_grid_length - length) <= 0.2 * length:
                length = full_grid_length
            else:
                length = fractional_length

        # Add a bit of buffer to ensure area requirements
        actual_area = width * length
        if actual_area < required_area * 0.95:  # Allow 5% tolerance
            # Try to increase dimensions using full grid first
            if width <= length:
                full_grid_width = math.ceil(width / grid_x) * grid_x
                if full_grid_width / width <= 1.25:  # Don't increase by more than 25%
                    width = full_grid_width
                else:
                    width = math.ceil(width / self.snap_size_x) * self.snap_size_x
            else:
                full_grid_length = math.ceil(length / grid_y) * grid_y
                if full_grid_length / length <= 1.25:  # Don't increase by more than 25%
                    length = full_grid_length
                else:
                    length = math.ceil(length / self.snap_size_y) * self.snap_size_y

        return width, length

    def find_grid_position(
        self,
        width: float,
        length: float,
        floor: int,
        room_type: str,
        placed_rooms_by_type: Dict[str, List[int]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a valid position for a room aligned to the grid.
        Prioritizes positions adjacent to preferred room types.

        Args:
            width: Room width in meters
            length: Room length in meters
            floor: Floor to place the room on
            room_type: Type of room for adjacency preferences
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None if no valid position
        """
        # Calculate z coordinate for the floor
        z = floor * self.floor_height

        # Convert dimensions to grid cells
        width_cells = round(width / self.snap_size_x)
        length_cells = round(length / self.snap_size_y)

        # Get grid occupancy for this floor
        if floor not in self.grid_occupancy:
            return None

        occupancy = self.grid_occupancy[floor]
        max_x, max_y = occupancy.shape

        # First try adjacency-based placement if applicable
        if room_type in self.adjacency_preferences:
            position = self._find_position_by_adjacency(
                width,
                length,
                width_cells,
                length_cells,
                floor,
                room_type,
                placed_rooms_by_type,
            )
            if position:
                return position

        # Try to place along perimeter if it's a public-facing room type
        perimeter_room_types = ["entrance", "lobby", "retail", "restaurant"]
        if room_type in perimeter_room_types:
            position = self._find_perimeter_position(
                width, length, width_cells, length_cells, floor
            )
            if position:
                return position

        # Standard grid placement - try all positions
        # Start from edges and work inward for better space utilization
        positions = []

        # Add positions along edges first
        for i in range(0, max_x - width_cells + 1):
            positions.append((i, 0))  # Top edge
            positions.append((i, max_y - length_cells))  # Bottom edge

        for j in range(1, max_y - length_cells):
            positions.append((0, j))  # Left edge
            positions.append((max_x - width_cells, j))  # Right edge

        # Add interior positions in a spiral pattern
        for layer in range(1, min(max_x, max_y) // 2):
            for i in range(layer, max_x - width_cells - layer + 1):
                for j in range(layer, max_y - length_cells - layer + 1):
                    if (
                        i == layer
                        or i == max_x - width_cells - layer
                        or j == layer
                        or j == max_y - length_cells - layer
                    ):
                        positions.append((i, j))

        # Add any remaining positions
        for i in range(1, max_x - width_cells):
            for j in range(1, max_y - length_cells):
                if (i, j) not in positions:
                    positions.append((i, j))

        # Shuffle positions to avoid predictable patterns
        random.shuffle(positions)

        # Check each position
        for i, j in positions:
            # Check if position is free
            if np.any(occupancy[i : i + width_cells, j : j + length_cells]):
                continue

            # Convert grid position to real coordinates
            x = i * self.snap_size_x
            y = j * self.snap_size_y

            # Check if position is valid
            if self._is_valid_position(x, y, z, width, length, self.floor_height):
                return (x, y, z)

        return None

    def _find_position_by_adjacency(
        self,
        width: float,
        length: float,
        width_cells: int,
        length_cells: int,
        floor: int,
        room_type: str,
        placed_rooms_by_type: Dict[str, List[int]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a position adjacent to preferred room types.

        Args:
            width, length: Room dimensions in meters
            width_cells, length_cells: Room dimensions in grid cells
            floor: Floor to place on
            room_type: Type of room
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        z = floor * self.floor_height

        # Get adjacency preferences
        preferred_types = self.adjacency_preferences.get(room_type, [])
        if not preferred_types:
            return None

        # Try to place adjacent to each preferred type
        for adj_type in preferred_types:
            if adj_type not in placed_rooms_by_type:
                continue

            # Try each room of this type
            for adj_room_id in placed_rooms_by_type[adj_type]:
                if adj_room_id not in self.spatial_grid.rooms:
                    continue

                adj_room = self.spatial_grid.rooms[adj_room_id]
                adj_x, adj_y, adj_z = adj_room["position"]
                adj_w, adj_l, adj_h = adj_room["dimensions"]

                # Only consider rooms on the same floor
                adj_floor = int(adj_z / self.floor_height)
                if adj_floor != floor:
                    continue

                # Try positions adjacent to this room
                # Convert to grid cells
                adj_i = round(adj_x / self.snap_size_x)
                adj_j = round(adj_y / self.snap_size_y)
                adj_width_cells = round(adj_w / self.snap_size_x)
                adj_length_cells = round(adj_l / self.snap_size_y)

                # Try right side
                i = adj_i + adj_width_cells
                for j in range(adj_j, adj_j + adj_length_cells - length_cells + 1):
                    x = i * self.snap_size_x
                    y = j * self.snap_size_y
                    if self._check_grid_position(
                        i, j, width_cells, length_cells, floor
                    ) and self._is_valid_position(
                        x, y, z, width, length, self.floor_height
                    ):
                        return (x, y, z)

                # Try left side
                i = adj_i - width_cells
                for j in range(adj_j, adj_j + adj_length_cells - length_cells + 1):
                    x = i * self.snap_size_x
                    y = j * self.snap_size_y
                    if self._check_grid_position(
                        i, j, width_cells, length_cells, floor
                    ) and self._is_valid_position(
                        x, y, z, width, length, self.floor_height
                    ):
                        return (x, y, z)

                # Try top side
                j = adj_j - length_cells
                for i in range(adj_i, adj_i + adj_width_cells - width_cells + 1):
                    x = i * self.snap_size_x
                    y = j * self.snap_size_y
                    if self._check_grid_position(
                        i, j, width_cells, length_cells, floor
                    ) and self._is_valid_position(
                        x, y, z, width, length, self.floor_height
                    ):
                        return (x, y, z)

                # Try bottom side
                j = adj_j + adj_length_cells
                for i in range(adj_i, adj_i + adj_width_cells - width_cells + 1):
                    x = i * self.snap_size_x
                    y = j * self.snap_size_y
                    if self._check_grid_position(
                        i, j, width_cells, length_cells, floor
                    ) and self._is_valid_position(
                        x, y, z, width, length, self.floor_height
                    ):
                        return (x, y, z)

        return None

    def _find_perimeter_position(
        self,
        width: float,
        length: float,
        width_cells: int,
        length_cells: int,
        floor: int,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a position along the building perimeter.

        Args:
            width, length: Room dimensions in meters
            width_cells, length_cells: Room dimensions in grid cells
            floor: Floor to place on

        Returns:
            Optional[Tuple[float, float, float]]: Position (x, y, z) or None
        """
        z = floor * self.floor_height

        # Get grid occupancy for this floor
        if floor not in self.grid_occupancy:
            return None

        occupancy = self.grid_occupancy[floor]
        max_x, max_y = occupancy.shape

        # Try placing along perimeter
        perimeter_positions = []

        # Top edge
        for i in range(0, max_x - width_cells + 1):
            perimeter_positions.append((i, 0))

        # Bottom edge
        for i in range(0, max_x - width_cells + 1):
            perimeter_positions.append((i, max_y - length_cells))

        # Left edge
        for j in range(1, max_y - length_cells):
            perimeter_positions.append((0, j))

        # Right edge
        for j in range(1, max_y - length_cells):
            perimeter_positions.append((max_x - width_cells, j))

        # Shuffle positions to avoid predictable patterns
        random.shuffle(perimeter_positions)

        # Check each position
        for i, j in perimeter_positions:
            # Check if position is free
            if np.any(occupancy[i : i + width_cells, j : j + length_cells]):
                continue

            # Convert grid position to real coordinates
            x = i * self.snap_size_x
            y = j * self.snap_size_y

            # Check if position is valid
            if self._is_valid_position(x, y, z, width, length, self.floor_height):
                return (x, y, z)

        return None

    def _check_grid_position(
        self, i: int, j: int, width_cells: int, length_cells: int, floor: int
    ) -> bool:
        """
        Check if a grid position is valid and unoccupied.

        Args:
            i, j: Grid coordinates
            width_cells, length_cells: Room dimensions in grid cells
            floor: Floor to check

        Returns:
            bool: True if position is valid
        """
        # Get grid occupancy for this floor
        if floor not in self.grid_occupancy:
            return False

        occupancy = self.grid_occupancy[floor]
        max_x, max_y = occupancy.shape

        # Check if position is within bounds
        if i < 0 or j < 0 or i + width_cells > max_x or j + length_cells > max_y:
            return False

        # Check if position is free
        return not np.any(occupancy[i : i + width_cells, j : j + length_cells])

    def _update_grid_occupancy(
        self, i: int, j: int, width_cells: int, length_cells: int, floor: int
    ):
        """
        Mark grid cells as occupied.

        Args:
            i, j: Grid coordinates
            width_cells, length_cells: Room dimensions in grid cells
            floor: Floor to update
        """
        if floor not in self.grid_occupancy:
            return

        occupancy = self.grid_occupancy[floor]

        # Check bounds
        max_x, max_y = occupancy.shape
        if i < 0 or j < 0 or i + width_cells > max_x or j + length_cells > max_y:
            return

        # Mark cells as occupied
        occupancy[i : i + width_cells, j : j + length_cells] = True

    def place_room_by_constraints(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Place a room according to grid-based constraints.
        Implementation of the abstract method from BaseEngine.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        # Get preferred floors
        preferred_floors = self._get_preferred_floors(room)

        # Adjust room dimensions to align with grid
        width, length = self.adjust_room_dimensions_to_grid(room)

        # Get dimensions in grid cells
        width_cells = round(width / self.snap_size_x)
        length_cells = round(length / self.snap_size_y)

        # Try special handling for certain room types
        if room.room_type == "vertical_circulation":
            return self._place_vertical_circulation(room, placed_rooms_by_type)
        elif room.room_type == "entrance":
            return self._place_entrance(room, placed_rooms_by_type)

        # Try each preferred floor
        for floor in preferred_floors:
            position = self.find_grid_position(
                width, length, floor, room.room_type, placed_rooms_by_type
            )

            if position:
                x, y, z = position

                # Place room
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
                    # Update grid occupancy
                    i = round(x / self.snap_size_x)
                    j = round(y / self.snap_size_y)
                    self._update_grid_occupancy(i, j, width_cells, length_cells, floor)
                    return True

        # If all preferred floors failed, try alternative floors
        alternate_floors = self._get_alternate_floors(room.room_type, preferred_floors)

        for floor in alternate_floors:
            position = self.find_grid_position(
                width, length, floor, room.room_type, placed_rooms_by_type
            )

            if position:
                x, y, z = position

                # Place room
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
                    # Update grid occupancy
                    i = round(x / self.snap_size_x)
                    j = round(y / self.snap_size_y)
                    self._update_grid_occupancy(i, j, width_cells, length_cells, floor)
                    return True

        return False

    def _place_vertical_circulation(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Special placement for vertical circulation elements.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        # Calculate full height from min to max floor
        total_floors = self.max_floor - self.min_floor + 1
        total_height = total_floors * self.floor_height

        # Starting z-coordinate from the lowest floor
        start_z = self.min_floor * self.floor_height

        # Update room metadata
        if not room.metadata:
            room.metadata = {}
        room.metadata["is_core"] = True
        room.metadata["spans_floors"] = list(range(self.min_floor, self.max_floor + 1))

        # Adjust dimensions to grid
        width, length = self.adjust_room_dimensions_to_grid(room)
        width_cells = round(width / self.snap_size_x)
        length_cells = round(length / self.snap_size_y)

        # Try positions near the center first
        grid_cells_x = math.ceil(self.width / self.snap_size_x)
        grid_cells_y = math.ceil(self.length / self.snap_size_y)

        center_i = grid_cells_x // 2 - width_cells // 2
        center_j = grid_cells_y // 2 - length_cells // 2

        # Generate positions in a spiral pattern from center
        positions = [(center_i, center_j)]
        max_radius = min(grid_cells_x, grid_cells_y) // 2

        for radius in range(1, max_radius):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if abs(i) == radius or abs(j) == radius:
                        pos_i = center_i + i
                        pos_j = center_j + j

                        # Ensure within bounds
                        if (
                            0 <= pos_i < grid_cells_x - width_cells
                            and 0 <= pos_j < grid_cells_y - length_cells
                        ):
                            positions.append((pos_i, pos_j))

        # Remove duplicates
        positions = list(set(positions))

        # Try each position
        for i, j in positions:
            x = i * self.snap_size_x
            y = j * self.snap_size_y

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
                    allow_overlap=["parking"],  # Allow overlap with parking
                )

                if success:
                    # Update grid occupancy for all floors
                    for floor in range(self.min_floor, self.max_floor + 1):
                        self._update_grid_occupancy(
                            i, j, width_cells, length_cells, floor
                        )
                    return True

        # Force placement as last resort
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
            allow_overlap=["parking"],
            force_placement=True,
        )

        if success:
            # Update grid occupancy for all floors
            i = round(x / self.snap_size_x)
            j = round(y / self.snap_size_y)
            for floor in range(self.min_floor, self.max_floor + 1):
                self._update_grid_occupancy(i, j, width_cells, length_cells, floor)
            return True

        return False

    def _place_entrance(
        self, room: Room, placed_rooms_by_type: Dict[str, List[int]]
    ) -> bool:
        """
        Special placement for entrance on perimeter with preference for front facade.

        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type

        Returns:
            bool: True if placed successfully
        """
        # Get preferred floor
        preferred_floors = self._get_preferred_floors(room)
        floor = preferred_floors[0] if preferred_floors else 0
        z = floor * self.floor_height

        # Adjust dimensions to grid
        width, length = self.adjust_room_dimensions_to_grid(room)
        width_cells = round(width / self.snap_size_x)
        length_cells = round(length / self.snap_size_y)

        # Get total grid cells
        grid_cells_x = math.ceil(self.width / self.snap_size_x)

        # Try front edge first (y = 0)
        # Center position on front edge
        center_i = grid_cells_x // 2 - width_cells // 2

        # Try center front position
        j = 0
        x = center_i * self.snap_size_x
        y = j * self.snap_size_y

        if self._check_grid_position(
            center_i, j, width_cells, length_cells, floor
        ) and self._is_valid_position(x, y, z, width, length, room.height):
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
                self._update_grid_occupancy(
                    center_i, j, width_cells, length_cells, floor
                )
                return True

        # Try other front positions
        for offset in range(1, grid_cells_x):
            for direction in [-1, 1]:
                i = center_i + direction * offset

                if i < 0 or i + width_cells > grid_cells_x:
                    continue

                x = i * self.snap_size_x
                y = j * self.snap_size_y

                if self._check_grid_position(
                    i, j, width_cells, length_cells, floor
                ) and self._is_valid_position(x, y, z, width, length, room.height):
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
                        self._update_grid_occupancy(
                            i, j, width_cells, length_cells, floor
                        )
                        return True

        # If front fails, try other perimeter positions
        position = self._find_perimeter_position(
            width, length, width_cells, length_cells, floor
        )

        if position:
            x, y, z = position
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
                i = round(x / self.snap_size_x)
                j = round(y / self.snap_size_y)
                self._update_grid_occupancy(i, j, width_cells, length_cells, floor)
                return True

        return False

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
        # Define floor ranges
        habitable_floors = [
            f for f in range(max(0, self.min_floor), self.max_floor + 1)
        ]
        upper_floors = [f for f in range(1, self.max_floor + 1)]
        basement_floors = [f for f in range(self.min_floor, min(0, self.max_floor + 1))]

        # Get all floors to try
        all_floors = list(range(self.min_floor, self.max_floor + 1))

        # Remove floors we've already tried
        candidate_floors = [f for f in all_floors if f not in used_floors]

        if not candidate_floors:
            return []

        # Prioritize floors based on room type
        if room_type == "guest_room":
            return sorted(
                candidate_floors,
                key=lambda f: (-1 if f in upper_floors else 0 if f == 0 else 1),
            )
        elif room_type in ["office", "staff_area"]:
            return sorted(
                candidate_floors,
                key=lambda f: (-1 if f in upper_floors else 0 if f == 0 else 1),
            )
        elif room_type in ["mechanical", "maintenance", "back_of_house", "parking"]:
            return sorted(
                candidate_floors, key=lambda f: (-1 if f in basement_floors else 0)
            )
        elif room_type in ["meeting_room", "food_service", "restaurant", "retail"]:
            priority_floors = [0, 1] if 1 in all_floors else [0]
            return sorted(
                candidate_floors, key=lambda f: (-1 if f in priority_floors else 0)
            )
        else:
            return candidate_floors

    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout with grid-aligned room placement.
        Implementation of the abstract method from BaseEngine.

        Args:
            rooms: List of Room objects to place

        Returns:
            SpatialGrid: The generated layout
        """
        # Reset grid occupancy
        self._init_grid_occupancy()

        # Sort rooms by architectural priority and size
        sorted_rooms = sorted(
            rooms,
            key=lambda r: (
                -self.placement_priorities.get(r.room_type, 0),  # Higher priority first
                -(r.width * r.length),  # Larger rooms first
            ),
        )

        # Track placed rooms by type
        placed_rooms_by_type = {}

        # Track placement statistics
        placed_count = 0
        failed_count = 0

        # Place each room
        for room in sorted_rooms:
            print(
                f"Placing {room.room_type} (id={room.id}, size={room.width}x{room.length})"
            )

            success = self.place_room_by_constraints(room, placed_rooms_by_type)

            if success:
                placed_count += 1
                if room.room_type not in placed_rooms_by_type:
                    placed_rooms_by_type[room.room_type] = []
                placed_rooms_by_type[room.room_type].append(room.id)
                print(
                    f"  Success - placed at {self.spatial_grid.rooms[room.id]['position']}"
                )
            else:
                failed_count += 1
                print(f"  Failed to place room")

        # Report statistics
        total_rooms = len(rooms)
        success_rate = (placed_count / total_rooms) * 100 if total_rooms > 0 else 0

        print(f"\nPlacement statistics:")
        print(f"  Total rooms: {total_rooms}")
        print(f"  Placed successfully: {placed_count} ({success_rate:.1f}%)")
        print(f"  Failed to place: {failed_count}")

        # Return the spatial grid with placed rooms
        return self.spatial_grid
