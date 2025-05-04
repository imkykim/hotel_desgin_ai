"""
Optimized standard floor generator for hotel tower designs.
This module generates standard floor layouts based on the podium's vertical circulation core.
Supports both horizontal and vertical room arrangements based on the dimensions.
"""

from typing import Dict, List, Tuple, Any, Optional
import math

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


def find_vertical_circulation_core(
    spatial_grid: SpatialGrid, building_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Find or create the main vertical circulation core.
    """
    # Get floor height for potential default core
    floor_height = building_config.get("floor_height", 4.0)

    # Look for vertical circulation elements
    core_candidates = []

    for room_id, room_data in spatial_grid.rooms.items():
        # Skip non-circulation elements
        if room_data["type"] != "vertical_circulation":
            continue

        # Extract metadata
        metadata = room_data.get("metadata", {})
        is_core = metadata.get("is_core", False)
        name = metadata.get("name", "")

        # If explicitly marked as core, prioritize it
        if is_core:
            print(f"Found marked core circulation: Room {room_id}")
            return room_data

        # If it has "main" or "core" in its name, also prioritize it
        if name and ("main" in name.lower() or "core" in name.lower()):
            print(f"Found main core circulation by name: {name} (Room {room_id})")
            return room_data

        # Otherwise, add to candidates
        core_candidates.append((room_id, room_data))

    # If we didn't find a marked core, look for the largest vertical circulation
    if core_candidates:
        # Sort by size (descending)
        core_candidates.sort(
            key=lambda x: x[1]["dimensions"][0] * x[1]["dimensions"][1], reverse=True
        )
        print(
            f"Using largest vertical circulation as core: Room {core_candidates[0][0]}"
        )
        return core_candidates[0][1]

    # If no circulation core found, create a default position
    print("Warning: No vertical circulation core found in podium")
    print("Will create a default core position in standard floors")

    # Get building dimensions
    building_width = building_config.get("width", 60.0)
    building_length = building_config.get("length", 80.0)

    # Create a default core in the north-center of the layout
    return {
        "id": 9999,  # Placeholder ID
        "position": (building_width / 2 - 4.0, building_length / 3, 0),
        "dimensions": (8.0, 8.0, floor_height),
        "type": "vertical_circulation",
        "metadata": {  # Ensure metadata is always included
            "name": "Default Core Circulation",
            "is_core": True,
        },
    }


def _generate_suite_rooms(
    floor_number: int,
    building_config: Dict[str, Any],
    spatial_grid: SpatialGrid,
    room_id_offset: int,
    is_horizontal_layout: bool,
) -> List[Room]:
    """
    Generate suite rooms at the corners of the standard floor boundary.
    Supports both horizontal and vertical layouts.

    Args:
        floor_number: The floor number to generate.
        building_config: Building configuration data.
        spatial_grid: The spatial grid to place rooms.
        room_id_offset: Offset for room IDs to avoid conflicts.
        is_horizontal_layout: Whether the layout is horizontal (True) or vertical (False)

    Returns:
        List[Room]: List of suite room objects.
    """
    suite_rooms = []
    suite_width = building_config.get(
        "structural_grid_x", 8.0
    )  # Width of one column grid
    suite_length = building_config.get(
        "structural_grid_y", 8.0
    )  # Length of one row grid
    std_floor_config = building_config.get("standard_floor", {})
    boundary_width = std_floor_config.get("width", 56.0)
    boundary_length = std_floor_config.get("length", 24.0)
    boundary_x = std_floor_config.get("position_x", 0.0)
    boundary_y = std_floor_config.get("position_y", 24.0)

    # Define suite room positions (one at each corner)
    suite_positions = [
        (boundary_x, boundary_y),  # Bottom-left corner
        (boundary_x + boundary_width - suite_width, boundary_y),  # Bottom-right corner
        (boundary_x, boundary_y + boundary_length - suite_length),  # Top-left corner
        (
            boundary_x + boundary_width - suite_width,
            boundary_y + boundary_length - suite_length,
        ),  # Top-right corner
    ]

    for i, (suite_x, suite_y) in enumerate(suite_positions):
        suite_id = room_id_offset + 100 + i

        # For vertical layout, swap width and length of suites for better corridor alignment
        if not is_horizontal_layout:
            room_width, room_length = suite_length, suite_width
        else:
            room_width, room_length = suite_width, suite_length

        spatial_grid.place_room(
            room_id=suite_id,
            x=suite_x,
            y=suite_y,
            z=floor_number * building_config.get("floor_height", 4.0),
            width=room_width,
            length=room_length,
            height=building_config.get("floor_height", 4.0),
            room_type="suite_room",
            metadata={"name": f"Suite Room {i + 1}", "is_suite": True},
        )
        suite_room = Room(
            width=room_width,
            length=room_length,
            height=building_config.get("floor_height", 4.0),
            room_type="suite_room",
            name=f"Suite Room {i + 1}",
            floor=floor_number,
            id=suite_id,
            metadata={"is_suite": True},
        )
        suite_room.position = (
            suite_x,
            suite_y,
            floor_number * building_config.get("floor_height", 4.0),
        )  # Set position separately
        suite_rooms.append(suite_room)

    return suite_rooms


def _generate_single_rooms(
    floor_number: int,
    building_config: Dict[str, Any],
    spatial_grid: SpatialGrid,
    circulation_core: Dict[str, Any],
    room_id_offset: int,
    is_horizontal_layout: bool,
) -> List[Room]:
    """
    Generate single rooms along the sides of the standard floor.
    Supports both horizontal (rooms on north/south) and vertical (rooms on east/west) layouts.

    Args:
        floor_number: The floor number to generate.
        building_config: Building configuration data.
        spatial_grid: The spatial grid to place rooms.
        circulation_core: Data for the vertical circulation core.
        room_id_offset: Offset for room IDs to avoid conflicts.
        is_horizontal_layout: Whether the layout is horizontal (True) or vertical (False)

    Returns:
        List[Room]: List of single room objects.
    """
    single_rooms = []
    structural_grid_x = building_config.get("structural_grid_x", 8.0)
    structural_grid_y = building_config.get("structural_grid_y", 8.0)
    std_floor_config = building_config.get("standard_floor", {})
    boundary_width = std_floor_config.get("width", 56.0)
    boundary_length = std_floor_config.get("length", 20.0)
    boundary_x = std_floor_config.get("position_x", 0.0)
    boundary_y = std_floor_config.get("position_y", 32.0)
    room_depth = std_floor_config.get("room_depth", 8.0)

    # Get floor height for z coordinate calculation
    floor_height = building_config.get("floor_height", 4.0)
    z_position = floor_number * floor_height

    # Get circulation core dimensions and position
    core_x, core_y, _ = circulation_core["position"]
    core_width, core_length, _ = circulation_core["dimensions"]

    # Different room generation based on layout orientation
    if is_horizontal_layout:
        # HORIZONTAL LAYOUT (corridor east-west, rooms north-south)
        single_width = structural_grid_x / 2  # Half grid width
        single_length = room_depth  # Use room depth for north-south facing rooms

        # South side rooms
        current_x = boundary_x
        while current_x + single_width <= boundary_x + boundary_width - single_width:
            room_id = room_id_offset + len(single_rooms) + 1
            room_x = current_x
            room_y = boundary_y  # Bottom edge

            spatial_grid.place_room(
                room_id=room_id,
                x=room_x,
                y=room_y,
                z=z_position,
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                metadata={"name": f"Single Room South {len(single_rooms) + 1}"},
            )

            single_room = Room(
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                name=f"Single Room South {len(single_rooms) + 1}",
                floor=floor_number,
                id=room_id,
                metadata={},
            )
            single_room.position = (room_x, room_y, z_position)
            single_rooms.append(single_room)
            current_x += single_width

        # North side rooms
        current_x = boundary_x
        while current_x + single_width <= boundary_x + boundary_width - single_width:
            room_id = room_id_offset + len(single_rooms) + 1
            room_x = current_x
            room_y = boundary_y + boundary_length - single_length  # Top edge

            spatial_grid.place_room(
                room_id=room_id,
                x=room_x,
                y=room_y,
                z=z_position,
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                metadata={"name": f"Single Room North {len(single_rooms) + 1}"},
            )

            single_room = Room(
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                name=f"Single Room North {len(single_rooms) + 1}",
                floor=floor_number,
                id=room_id,
                metadata={},
            )
            single_room.position = (room_x, room_y, z_position)
            single_rooms.append(single_room)
            current_x += single_width

    else:
        # VERTICAL LAYOUT (corridor north-south, rooms east-west)
        single_length = structural_grid_y / 2  # Half grid length
        single_width = room_depth  # Use room depth for east-west facing rooms

        # East side rooms
        current_y = boundary_y
        while current_y + single_length <= boundary_y + boundary_length - single_length:
            room_id = room_id_offset + len(single_rooms) + 1
            room_y = current_y
            room_x = boundary_x  # Left edge

            spatial_grid.place_room(
                room_id=room_id,
                x=room_x,
                y=room_y,
                z=z_position,
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                metadata={"name": f"Single Room East {len(single_rooms) + 1}"},
            )

            single_room = Room(
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                name=f"Single Room East {len(single_rooms) + 1}",
                floor=floor_number,
                id=room_id,
                metadata={},
            )
            single_room.position = (room_x, room_y, z_position)
            single_rooms.append(single_room)
            current_y += single_length

        # West side rooms
        current_y = boundary_y
        while current_y + single_length <= boundary_y + boundary_length - single_length:
            room_id = room_id_offset + len(single_rooms) + 1
            room_y = current_y
            room_x = boundary_x + boundary_width - single_width  # Right edge

            spatial_grid.place_room(
                room_id=room_id,
                x=room_x,
                y=room_y,
                z=z_position,
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                metadata={"name": f"Single Room West {len(single_rooms) + 1}"},
            )

            single_room = Room(
                width=single_width,
                length=single_length,
                height=floor_height,
                room_type="single_room",
                name=f"Single Room West {len(single_rooms) + 1}",
                floor=floor_number,
                id=room_id,
                metadata={},
            )
            single_room.position = (room_x, room_y, z_position)
            single_rooms.append(single_room)
            current_y += single_length

    return single_rooms


def _generate_corridor(
    floor_number: int,
    building_config: Dict[str, Any],
    spatial_grid: SpatialGrid,
    room_id_offset: int,
    is_horizontal_layout: bool,
) -> Room:
    """
    Generate the corridor between rows of double-loaded rooms.
    Supports both horizontal and vertical layouts.

    Args:
        floor_number: The floor number to generate.
        building_config: Building configuration data.
        spatial_grid: The spatial grid to place the corridor.
        room_id_offset: Offset for room IDs to avoid conflicts.
        is_horizontal_layout: Whether the layout is horizontal (True) or vertical (False)

    Returns:
        Room: The corridor room object.
    """
    std_floor_config = building_config.get("standard_floor", {})
    boundary_width = std_floor_config.get("width", 56.0)
    boundary_length = std_floor_config.get("length", 20.0)
    boundary_x = std_floor_config.get("position_x", 0.0)
    boundary_y = std_floor_config.get("position_y", 32.0)
    corridor_width = std_floor_config.get("corridor_width", 4.0)
    floor_height = building_config.get("floor_height", 4.0)
    z_position = floor_number * floor_height

    if is_horizontal_layout:
        # HORIZONTAL LAYOUT - corridor runs east-west through the center
        corridor_x = boundary_x
        corridor_y = boundary_y + (boundary_length - corridor_width) / 2
        corridor_length = corridor_width
        corridor_width = boundary_width  # Full width of standard floor
    else:
        # VERTICAL LAYOUT - corridor runs north-south through the center
        corridor_x = boundary_x + (boundary_width - corridor_width) / 2
        corridor_y = boundary_y
        corridor_length = boundary_length  # Full length of standard floor
        corridor_width = corridor_width  # Width as specified in config

    # Place the corridor
    corridor_id = room_id_offset
    spatial_grid.place_room(
        room_id=corridor_id,
        x=corridor_x,
        y=corridor_y,
        z=z_position,
        width=corridor_width,
        length=corridor_length,
        height=floor_height,
        room_type="corridor",
        metadata={"name": f"Corridor Floor {floor_number}"},
    )

    corridor_room = Room(
        width=corridor_width,
        length=corridor_length,
        height=floor_height,
        room_type="corridor",
        name=f"Corridor Floor {floor_number}",
        floor=floor_number,
        id=corridor_id,
        metadata={},
    )
    corridor_room.position = (corridor_x, corridor_y, z_position)

    return corridor_room


def generate_standard_floor(
    floor_number: int,
    building_config: Dict[str, Any],
    circulation_core: Optional[Dict[str, Any]] = None,
    spatial_grid: Optional[SpatialGrid] = None,
    room_id_offset: int = 0,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate a standard floor layout anchored to the vertical circulation core.
    Automatically determines optimal orientation based on standard floor dimensions.
    """
    # Initialize spatial grid if not provided
    if spatial_grid is None:
        spatial_grid = SpatialGrid(
            width=building_config.get("width", 60.0),
            length=building_config.get("length", 80.0),
            height=building_config.get("height", 100.0),
            grid_size=building_config.get("grid_size", 1.0),
            min_floor=building_config.get("min_floor", -2),
            floor_height=building_config.get("floor_height", 4.0),
        )

    # Get standard floor dimensions to determine orientation
    std_floor_config = building_config.get("standard_floor", {})
    boundary_width = std_floor_config.get("width", 56.0)
    boundary_length = std_floor_config.get("length", 20.0)

    # Determine layout orientation - horizontal if width >= length, vertical otherwise
    is_horizontal_layout = boundary_width >= boundary_length

    print(f"Standard floor dimensions: {boundary_width}m x {boundary_length}m")
    print(f"Using {'horizontal' if is_horizontal_layout else 'vertical'} layout")

    # Place vertical circulation core
    core = circulation_core or find_vertical_circulation_core(
        spatial_grid, building_config
    )
    core_metadata = core.get(
        "metadata", {"name": "Unnamed Core", "is_core": False}
    )  # Fallback metadata
    spatial_grid.place_room(
        room_id=room_id_offset + 1,
        x=core["position"][0],
        y=core["position"][1],
        z=floor_number * building_config.get("floor_height", 4.0),
        width=core["dimensions"][0],
        length=core["dimensions"][1],
        height=core["dimensions"][2],
        room_type="vertical_circulation",
        metadata=core_metadata,
    )

    # Place suite rooms
    suite_rooms = _generate_suite_rooms(
        floor_number=floor_number,
        building_config=building_config,
        spatial_grid=spatial_grid,
        room_id_offset=room_id_offset,
        is_horizontal_layout=is_horizontal_layout,
    )

    # Place single rooms
    single_rooms = _generate_single_rooms(
        floor_number=floor_number,
        building_config=building_config,
        spatial_grid=spatial_grid,
        circulation_core=core,
        room_id_offset=room_id_offset + 200,  # Offset to avoid ID conflicts
        is_horizontal_layout=is_horizontal_layout,
    )

    # Place corridor
    corridor = _generate_corridor(
        floor_number=floor_number,
        building_config=building_config,
        spatial_grid=spatial_grid,
        room_id_offset=room_id_offset + 500,  # Offset to avoid ID conflicts
        is_horizontal_layout=is_horizontal_layout,
    )

    # Combine all rooms
    all_rooms = suite_rooms + single_rooms + [corridor]

    # Return spatial grid and all rooms
    return spatial_grid, all_rooms


def generate_all_standard_floors(
    building_config: Dict[str, Any],
    spatial_grid: Optional[SpatialGrid] = None,
    target_room_count: int = 380,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate all standard floors for a hotel tower.
    """
    # Initialize spatial grid if not provided
    if spatial_grid is None:
        spatial_grid = SpatialGrid(
            width=building_config.get("width", 60.0),
            length=building_config.get("length", 80.0),
            height=building_config.get("height", 100.0),
            grid_size=building_config.get("grid_size", 1.0),
            min_floor=building_config.get("min_floor", -2),
            floor_height=building_config.get("floor_height", 4.0),
        )

    # Find circulation core
    circulation_core = find_vertical_circulation_core(spatial_grid, building_config)

    # Generate floors
    start_floor = building_config.get("standard_floor", {}).get("start_floor", 5)
    end_floor = building_config.get("standard_floor", {}).get("end_floor", 20)
    all_rooms = []

    for floor_number in range(start_floor, end_floor + 1):
        _, floor_rooms = generate_standard_floor(
            floor_number=floor_number,
            building_config=building_config,
            circulation_core=circulation_core,
            spatial_grid=spatial_grid,
            room_id_offset=floor_number * 1000,
        )
        all_rooms.extend(floor_rooms)

    return spatial_grid, all_rooms
