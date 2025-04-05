"""
Optimized standard floor generator for hotel tower designs.
This module generates standard floor layouts based on the podium's vertical circulation core.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import numpy as np
import math

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


def find_vertical_circulation_core(
    spatial_grid: SpatialGrid, building_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Find the main vertical circulation core in the podium that will extend up to standard floors.

    Args:
        spatial_grid: The spatial grid containing the podium layout
        building_config: Building configuration parameters

    Returns:
        Optional[Dict[str, Any]]: Core data if found, None otherwise
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
        "metadata": {"name": "Default Core Circulation", "is_core": True},
    }


def generate_standard_floor(
    floor_number: int,
    building_config: Dict[str, Any],
    circulation_core: Optional[Dict[str, Any]] = None,
    spatial_grid: Optional[SpatialGrid] = None,
    room_id_offset: int = 0,
    include_service_rooms: bool = False,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate a standard floor layout anchored to the vertical circulation core.

    Args:
        floor_number: The floor number to generate
        building_config: Building configuration data
        circulation_core: Data for the vertical circulation core (if known)
        spatial_grid: Optional existing spatial grid to add to
        room_id_offset: Offset for room IDs to avoid conflicts
        include_service_rooms: Whether to include housekeeping/linen rooms

    Returns:
        Tuple containing the spatial grid and list of room objects
    """
    # Get configuration parameters
    std_floor_config = building_config.get("standard_floor", {})
    floor_height = building_config.get("floor_height", 4.0)
    building_width = building_config.get("width", 60.0)
    building_length = building_config.get("length", 80.0)
    grid_size = building_config.get("grid_size", 1.0)

    # Get grid sizes and calculate half-grid for alignment
    structural_grid_x = building_config.get("structural_grid_x", 8.0)
    structural_grid_y = building_config.get("structural_grid_y", 8.0)
    half_grid_x = structural_grid_x / 2
    half_grid_y = structural_grid_y / 2

    # Calculate z-coordinate for this floor
    z_position = floor_number * floor_height

    # Create spatial grid if needed
    if spatial_grid is None:
        spatial_grid = SpatialGrid(
            width=building_width,
            length=building_length,
            height=building_config.get("height", 100.0),
            grid_size=grid_size,
            min_floor=building_config.get("min_floor", -2),
            floor_height=floor_height,
        )

    # Initialize room list and create base ID for this floor
    rooms = []
    floor_base_id = room_id_offset + (floor_number * 10000)

    # STEP 1: Place or identify vertical circulation core
    core_info = _place_vertical_circulation_core(
        spatial_grid,
        floor_number,
        floor_base_id,
        z_position,
        floor_height,
        circulation_core,
        rooms,
        building_width,
        building_length,
    )

    if not core_info:
        print(f"Failed to place vertical circulation core on floor {floor_number}")
        return spatial_grid, rooms

    core_x, core_y, core_width, core_length = core_info

    # STEP 2: Place service rooms if requested
    service_rooms_info = None
    if include_service_rooms:
        service_rooms_info = _place_service_rooms(
            spatial_grid,
            floor_number,
            floor_base_id,
            z_position,
            floor_height,
            core_x,
            core_y,
            core_width,
            core_length,
            half_grid_x,
            half_grid_y,
            rooms,
        )

    # STEP 3: Place corridor
    corridor_info = _place_corridor(
        spatial_grid,
        floor_number,
        floor_base_id,
        z_position,
        floor_height,
        core_x,
        core_y,
        core_width,
        core_length,
        half_grid_x,
        half_grid_y,
        service_rooms_info,
        std_floor_config,
    )

    if not corridor_info:
        print(f"Failed to place corridor on floor {floor_number}")
        return spatial_grid, rooms

    corridor_x, corridor_y, corridor_width, corridor_length = corridor_info

    # STEP 4: Place guest rooms on both sides of the corridor
    # Get room dimensions from config or use defaults
    room_depth = std_floor_config.get("room_depth", 6.0)
    room_width = std_floor_config.get("room_width", 4.0)

    # Align to grid
    room_depth = round(room_depth / half_grid_y) * half_grid_y
    room_width = round(room_width / half_grid_x) * half_grid_x

    # Calculate how many rooms can fit along the corridor
    max_rooms_per_side = math.floor(corridor_length / room_width)
    print(f"Can fit approximately {max_rooms_per_side} rooms per side")

    # Place rooms on both sides of corridor
    room_placement_info = {
        "core": (core_x, core_y, core_width, core_length),
        "service_rooms": service_rooms_info,
        "corridor": (corridor_x, corridor_y, corridor_width, corridor_length),
        "room_dimensions": (room_width, room_depth, floor_height),
        "building_bounds": (building_width, building_length),
        "max_rooms": max_rooms_per_side,
    }

    north_rooms = _place_north_rooms(
        spatial_grid,
        floor_number,
        floor_base_id,
        z_position,
        rooms,
        room_placement_info,
    )

    south_rooms = _place_south_rooms(
        spatial_grid,
        floor_number,
        floor_base_id,
        z_position,
        rooms,
        room_placement_info,
    )

    # Report results
    total_rooms = north_rooms + south_rooms
    print(
        f"Floor {floor_number}: Placed {total_rooms} guest rooms ({north_rooms} north + {south_rooms} south)"
    )

    return spatial_grid, rooms


def _place_vertical_circulation_core(
    spatial_grid,
    floor_number,
    floor_base_id,
    z_position,
    floor_height,
    circulation_core,
    rooms,
    building_width,
    building_length,
):
    """Helper function to place vertical circulation core"""
    if circulation_core:
        core_x, core_y, _ = circulation_core["position"]
        core_width, core_length, _ = circulation_core["dimensions"]
    else:
        # Default core position and dimensions if none provided
        core_x = building_width / 2 - 4.0
        core_y = building_length / 2 - 4.0
        core_width = 8.0
        core_length = 8.0

    # Place core on this floor
    core_id = floor_base_id + 1
    core_success = spatial_grid.place_room(
        room_id=core_id,
        x=core_x,
        y=core_y,
        z=z_position,
        width=core_width,
        length=core_length,
        height=floor_height,
        room_type="vertical_circulation",
        metadata={
            "name": f"Core Circulation Floor {floor_number}",
            "is_core": True,
        },
    )

    if not core_success:
        return None

    # Create core room object and add to rooms list
    print(
        f"Placed vertical circulation core on floor {floor_number} at ({core_x}, {core_y})"
    )
    core_room = Room(
        width=core_width,
        length=core_length,
        height=floor_height,
        room_type="vertical_circulation",
        name=f"Core Circulation Floor {floor_number}",
        floor=floor_number,
        id=core_id,
        metadata={"is_core": True},
    )
    core_room.position = (core_x, core_y, z_position)
    rooms.append(core_room)

    return core_x, core_y, core_width, core_length


def _place_service_rooms(
    spatial_grid,
    floor_number,
    floor_base_id,
    z_position,
    floor_height,
    core_x,
    core_y,
    core_width,
    core_length,
    half_grid_x,
    half_grid_y,
    rooms,
):
    """Helper function to place service rooms (housekeeping, linen room)"""
    service_room_width = round(3.0 / half_grid_x) * half_grid_x
    service_room_length = round(5.0 / half_grid_y) * half_grid_y

    # Housekeeping room
    service_room1_id = floor_base_id + 10
    service_room1_x = core_x
    service_room1_y = core_y + core_length + half_grid_y

    # Align to grid
    service_room1_x = round(service_room1_x / half_grid_x) * half_grid_x
    service_room1_y = round(service_room1_y / half_grid_y) * half_grid_y

    housekeeping_success = spatial_grid.place_room(
        room_id=service_room1_id,
        x=service_room1_x,
        y=service_room1_y,
        z=z_position,
        width=service_room_width,
        length=service_room_length,
        height=floor_height,
        room_type="service_area",
        metadata={
            "name": f"Housekeeping Floor {floor_number}",
            "subspace_name": "housekeeping",
        },
    )

    if housekeeping_success:
        print(f"Placed housekeeping room on floor {floor_number}")
        housekeeping_room = Room(
            width=service_room_width,
            length=service_room_length,
            height=floor_height,
            room_type="service_area",
            name=f"Housekeeping Floor {floor_number}",
            floor=floor_number,
            id=service_room1_id,
            metadata={"subspace_name": "housekeeping"},
        )
        housekeeping_room.position = (service_room1_x, service_room1_y, z_position)
        rooms.append(housekeeping_room)

    # Linen room
    service_room2_id = floor_base_id + 11
    service_room2_x = (
        service_room1_x + service_room_width + half_grid_x
        if housekeeping_success
        else core_x + core_width + half_grid_x
    )
    service_room2_y = service_room1_y if housekeeping_success else core_y

    # Align to grid
    service_room2_x = round(service_room2_x / half_grid_x) * half_grid_x

    linen_success = spatial_grid.place_room(
        room_id=service_room2_id,
        x=service_room2_x,
        y=service_room2_y,
        z=z_position,
        width=service_room_width,
        length=service_room_length,
        height=floor_height,
        room_type="service_area",
        metadata={
            "name": f"Linen Room Floor {floor_number}",
            "subspace_name": "linen_room",
        },
    )

    if linen_success:
        print(f"Placed linen room on floor {floor_number}")
        linen_room = Room(
            width=service_room_width,
            length=service_room_length,
            height=floor_height,
            room_type="service_area",
            name=f"Linen Room Floor {floor_number}",
            floor=floor_number,
            id=service_room2_id,
            metadata={"subspace_name": "linen_room"},
        )
        linen_room.position = (service_room2_x, service_room2_y, z_position)
        rooms.append(linen_room)

    if housekeeping_success or linen_success:
        return {
            "housekeeping": (
                (
                    service_room1_x,
                    service_room1_y,
                    service_room_width,
                    service_room_length,
                )
                if housekeeping_success
                else None
            ),
            "linen": (
                (
                    service_room2_x,
                    service_room2_y,
                    service_room_width,
                    service_room_length,
                )
                if linen_success
                else None
            ),
        }

    return None


def _place_corridor(
    spatial_grid,
    floor_number,
    floor_base_id,
    z_position,
    floor_height,
    core_x,
    core_y,
    core_width,
    core_length,
    half_grid_x,
    half_grid_y,
    service_rooms_info,
    std_floor_config,
):
    """Helper function to place the corridor"""
    corridor_width = std_floor_config.get("corridor_width", 2.4)
    corridor_width = round(corridor_width / half_grid_y) * half_grid_y  # Align to grid

    # Use a reasonable default corridor length or get from config
    corridor_length = std_floor_config.get("corridor_length", 40.0)

    # Calculate corridor position (centered on the core)
    core_center_x = core_x + core_width / 2
    corridor_x = core_center_x - corridor_length / 2

    # Position the corridor depending on service rooms
    if service_rooms_info and service_rooms_info["housekeeping"]:
        hk_x, hk_y, hk_width, hk_length = service_rooms_info["housekeeping"]
        corridor_y = hk_y + hk_length
    else:
        corridor_y = core_y + core_length

    # Align to grid
    corridor_x = round(corridor_x / half_grid_x) * half_grid_x
    corridor_y = round(corridor_y / half_grid_y) * half_grid_y
    corridor_length = round(corridor_length / half_grid_x) * half_grid_x

    # Place corridor
    corridor_id = floor_base_id + 2
    corridor_success = spatial_grid.place_room(
        room_id=corridor_id,
        x=corridor_x,
        y=corridor_y,
        z=z_position,
        width=corridor_length,
        length=corridor_width,
        height=floor_height,
        room_type="circulation",
        metadata={"name": f"Corridor Floor {floor_number}", "is_corridor": True},
    )

    if corridor_success:
        return corridor_x, corridor_y, corridor_width, corridor_length

    return None


def _place_north_rooms(
    spatial_grid, floor_number, floor_base_id, z_position, rooms, room_placement_info
):
    """Helper function to place rooms on the north side of the corridor"""
    corridor_x, corridor_y, corridor_width, corridor_length = room_placement_info[
        "corridor"
    ]
    room_width, room_depth, floor_height = room_placement_info["room_dimensions"]
    core_x, core_y, core_width, core_length = room_placement_info["core"]
    max_rooms = room_placement_info["max_rooms"]
    service_rooms_info = room_placement_info["service_rooms"]

    north_rooms_placed = 0

    for i in range(max_rooms):
        room_id = floor_base_id + 100 + i

        # Position rooms along corridor
        room_x = corridor_x + (i * room_width)
        room_y = corridor_y + corridor_width  # North of corridor

        # Skip if room would overlap with core
        if _check_room_overlap(
            room_x,
            room_y,
            room_width,
            room_depth,
            core_x,
            core_y,
            core_width,
            core_length,
        ):
            continue

        # Skip if room would overlap with service rooms
        if service_rooms_info:
            if service_rooms_info["housekeeping"] and _check_room_overlap(
                room_x,
                room_y,
                room_width,
                room_depth,
                *service_rooms_info["housekeeping"],
            ):
                continue

            if service_rooms_info["linen"] and _check_room_overlap(
                room_x, room_y, room_width, room_depth, *service_rooms_info["linen"]
            ):
                continue

        # Create unique room number
        room_number = f"{floor_number}{i+1:02d}"

        # Place room
        north_success = spatial_grid.place_room(
            room_id=room_id,
            x=room_x,
            y=room_y,
            z=z_position,
            width=room_width,
            length=room_depth,
            height=floor_height,
            room_type="guest_room",
            metadata={"name": f"Room {room_number}", "side": "north"},
        )

        if north_success:
            # Create room object
            guest_room = Room(
                width=room_width,
                length=room_depth,
                height=floor_height,
                room_type="guest_room",
                name=f"Room {room_number}",
                floor=floor_number,
                id=room_id,
                metadata={"side": "north"},
            )
            guest_room.position = (room_x, room_y, z_position)
            rooms.append(guest_room)
            north_rooms_placed += 1

    return north_rooms_placed


def _place_south_rooms(
    spatial_grid, floor_number, floor_base_id, z_position, rooms, room_placement_info
):
    """Helper function to place rooms on the south side of the corridor"""
    corridor_x, corridor_y, corridor_width, corridor_length = room_placement_info[
        "corridor"
    ]
    room_width, room_depth, floor_height = room_placement_info["room_dimensions"]
    building_width, building_length = room_placement_info["building_bounds"]
    max_rooms = room_placement_info["max_rooms"]

    south_rooms_placed = 0

    for i in range(max_rooms):
        room_id = floor_base_id + 200 + i

        # Position rooms along corridor
        room_x = corridor_x + (i * room_width)
        room_y = corridor_y - room_depth  # South of corridor

        # Check if room would be outside building bounds
        if (
            room_x < 0
            or room_x + room_width > building_width
            or room_y < 0
            or room_y + room_depth > building_length
        ):
            continue

        # Create unique room number
        room_number = f"{floor_number}{max_rooms+i+1:02d}"

        # Place room
        south_success = spatial_grid.place_room(
            room_id=room_id,
            x=room_x,
            y=room_y,
            z=z_position,
            width=room_width,
            length=room_depth,
            height=floor_height,
            room_type="guest_room",
            metadata={"name": f"Room {room_number}", "side": "south"},
        )

        if south_success:
            # Create room object
            guest_room = Room(
                width=room_width,
                length=room_depth,
                height=floor_height,
                room_type="guest_room",
                name=f"Room {room_number}",
                floor=floor_number,
                id=room_id,
                metadata={"side": "south"},
            )
            guest_room.position = (room_x, room_y, z_position)
            rooms.append(guest_room)
            south_rooms_placed += 1

    return south_rooms_placed


def _check_room_overlap(
    room1_x,
    room1_y,
    room1_width,
    room1_length,
    room2_x,
    room2_y,
    room2_width,
    room2_length,
):
    """Helper function to check if two rectangular rooms overlap"""
    return (
        room1_x < room2_x + room2_width
        and room1_x + room1_width > room2_x
        and room1_y < room2_y + room2_length
        and room1_y + room1_length > room2_y
    )


def generate_all_standard_floors(
    building_config: Dict[str, Any],
    spatial_grid: Optional[SpatialGrid] = None,
    target_room_count: int = 380,
    include_service_rooms: bool = False,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate all standard floors for a hotel tower.

    Args:
        building_config: Building configuration with standard floor parameters
        spatial_grid: Optional existing spatial grid to add to
        target_room_count: Target number of guest rooms to generate
        include_service_rooms: Whether to include service rooms on each floor

    Returns:
        Tuple containing the spatial grid and list of room objects
    """
    print("\nGenerating all standard floors...")

    # Get standard floor range from config
    std_floor_config = building_config.get("standard_floor", {})
    start_floor = std_floor_config.get("start_floor", 5)
    end_floor = std_floor_config.get("end_floor", 20)

    # Create spatial grid if needed
    if spatial_grid is None:
        spatial_grid = SpatialGrid(
            width=building_config.get("width", 60.0),
            length=building_config.get("length", 80.0),
            height=building_config.get("height", 100.0),
            grid_size=building_config.get("grid_size", 1.0),
            min_floor=building_config.get("min_floor", -2),
            floor_height=building_config.get("floor_height", 4.0),
        )

    # Find the circulation core from the podium
    circulation_core = find_vertical_circulation_core(spatial_grid, building_config)

    if circulation_core:
        print(f"Found vertical circulation core to extend to standard floors")
        print(f"Core position: {circulation_core['position']}")
        print(f"Core dimensions: {circulation_core['dimensions']}")
    else:
        print(
            "No existing vertical circulation core found, will create one for standard floors"
        )

    # Calculate room ID offset to avoid conflicts
    room_id_offset = 0
    if hasattr(spatial_grid, "rooms") and spatial_grid.rooms:
        existing_ids = [int(room_id) for room_id in spatial_grid.rooms.keys()]
        if existing_ids:
            room_id_offset = max(existing_ids) + 1000

    # Initialize list for all rooms
    all_rooms = []

    # First generate a test floor to estimate rooms per floor
    test_spatial_grid = SpatialGrid(
        width=building_config.get("width", 60.0),
        length=building_config.get("length", 80.0),
        height=building_config.get("height", 100.0),
        grid_size=building_config.get("grid_size", 1.0),
        min_floor=building_config.get("min_floor", -2),
        floor_height=building_config.get("floor_height", 4.0),
    )

    print("\nGenerating test floor to estimate rooms per floor...")
    _, test_rooms = generate_standard_floor(
        floor_number=start_floor,
        building_config=building_config,
        circulation_core=circulation_core,
        spatial_grid=test_spatial_grid,
        room_id_offset=0,
        include_service_rooms=include_service_rooms,
    )

    # Count guest rooms in test floor
    test_guest_rooms = sum(1 for room in test_rooms if room.room_type == "guest_room")
    print(f"Test floor has {test_guest_rooms} guest rooms")

    # Calculate required floors to reach target room count
    if test_guest_rooms > 0:
        floors_needed = math.ceil(target_room_count / test_guest_rooms)
        actual_end_floor = min(end_floor, start_floor + floors_needed - 1)
        print(
            f"Need approximately {floors_needed} floors to reach {target_room_count} rooms"
        )
        print(f"Will generate floors {start_floor} to {actual_end_floor}")
    else:
        actual_end_floor = end_floor
        print("Warning: Test floor has no guest rooms, using full floor range")

    # Generate each standard floor until we reach the target room count
    current_room_count = 0
    current_floor = start_floor

    while current_floor <= actual_end_floor and current_room_count < target_room_count:
        print(f"\nGenerating floor {current_floor}...")

        # Generate this floor's layout
        _, floor_rooms = generate_standard_floor(
            floor_number=current_floor,
            building_config=building_config,
            circulation_core=circulation_core,
            spatial_grid=spatial_grid,
            room_id_offset=room_id_offset + (current_floor - start_floor) * 1000,
            include_service_rooms=include_service_rooms,
        )

        all_rooms.extend(floor_rooms)

        # Count guest rooms on this floor
        floor_guest_rooms = sum(
            1 for room in floor_rooms if room.room_type == "guest_room"
        )
        current_room_count += floor_guest_rooms

        print(f"Floor {current_floor} complete: Added {floor_guest_rooms} guest rooms")
        print(
            f"Progress: {current_room_count}/{target_room_count} guest rooms ({current_room_count/target_room_count*100:.1f}%)"
        )

        # Move to next floor
        current_floor += 1

    # Generate summary statistics
    total_guest_rooms = sum(1 for room in all_rooms if room.room_type == "guest_room")
    total_floors = current_floor - start_floor

    print(f"\nStandard floor generation complete:")
    print(f"- Generated {total_floors} floors from {start_floor} to {current_floor-1}")
    print(f"- Placed {total_guest_rooms} guest rooms (target: {target_room_count})")

    if total_floors > 0:
        print(f"- Average {total_guest_rooms/total_floors:.1f} guest rooms per floor")

    # Report on target achievement
    if total_guest_rooms >= target_room_count:
        print(f"✓ Successfully reached target of {target_room_count} rooms")
    else:
        print(
            f"⚠ Only generated {total_guest_rooms}/{target_room_count} rooms ({total_guest_rooms/target_room_count*100:.1f}%)"
        )
        print("  Consider adjusting room sizes or floor range to meet the target")

    return spatial_grid, all_rooms
