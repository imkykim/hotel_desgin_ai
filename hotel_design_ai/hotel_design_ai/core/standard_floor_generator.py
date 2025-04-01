"""
Standard floor generator for hotel tower designs.
This module provides functions to generate standard floor layouts with guest rooms.
"""

from typing import Dict, List, Tuple, Any, Optional
import json
import os
import numpy as np
import math
import random

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


def generate_standard_floor(
    floor_number: int,
    building_config: Dict[str, Any],
    template_path: Optional[str] = None,
    spatial_grid: Optional[SpatialGrid] = None,
    room_id_offset: int = 0,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate a standard floor layout for a hotel tower with specified room types.

    Args:
        floor_number: The floor number to generate
        building_config: Building configuration with standard floor parameters
        template_path: Optional path to a floor template JSON file
        spatial_grid: Optional existing spatial grid to add to
        room_id_offset: Offset for room IDs to avoid conflicts

    Returns:
        Tuple containing the spatial grid and list of room objects
    """
    # Get standard floor parameters from building config
    std_floor_config = building_config.get("standard_floor", {})

    # Get floor dimensions and position
    floor_width = std_floor_config.get("width", 40.0)
    floor_length = std_floor_config.get("length", 20.0)
    floor_position_x = std_floor_config.get("position_x", 10.0)
    floor_position_y = std_floor_config.get("position_y", 30.0)

    # Get corridor and room parameters
    corridor_width = std_floor_config.get("corridor_width", 2.4)
    room_depth = std_floor_config.get("room_depth", 8.0)
    floor_height = building_config.get("floor_height", 4.0)

    # Room type distribution and sizes (from Chinese requirements)
    room_types_distribution = {
        "double_bed": 0.70,  # 70% double-bed rooms
        "single_bed": 0.15,  # 15% single-bed rooms
        "suite": 0.15,  # 15% suite rooms
    }

    room_sizes = {
        "double_bed": {"width": 4.5, "length": 5.5},  # ~25m² bedroom
        "single_bed": {"width": 4.5, "length": 5.5},  # ~25m² bedroom
        "suite": {"width": 6.5, "length": 8.0},  # ~52m² bedroom
    }

    # Service room sizes (from Chinese requirements)
    service_rooms = {
        "duty_room": {"width": 3.0, "length": 5.0},  # 15m²
        "linen_room": {"width": 3.0, "length": 5.0},  # 15m²
        "staff_bathroom": {"width": 1.5, "length": 2.0},  # 3m²
    }

    # Calculate z-coordinate for this floor
    z_position = floor_number * floor_height

    # If no spatial grid provided, create a new one
    if spatial_grid is None:
        # Use the overall building envelope dimensions
        building_width = building_config.get("width", 60.0)
        building_length = building_config.get("length", 80.0)
        building_height = building_config.get("height", 100.0)
        grid_size = building_config.get("grid_size", 1.0)
        min_floor = building_config.get("min_floor", -2)

        spatial_grid = SpatialGrid(
            width=building_width,
            length=building_length,
            height=building_height,
            grid_size=grid_size,
            min_floor=min_floor,
            floor_height=floor_height,
        )

    # Load template if specified
    template = None
    if template_path and os.path.exists(template_path):
        with open(template_path, "r") as f:
            template = json.load(f)

        # Override default values with template values if available
        if "parameters" in template:
            params = template["parameters"]
            corridor_width = params.get("corridor_width", corridor_width)
            room_depth = params.get("room_depth", room_depth)
            if "floor_width" in params:
                floor_width = params["floor_width"]
            if "floor_length" in params:
                floor_length = params["floor_length"]

    # Initialize list for room objects
    rooms = []

    # Place corridor
    corridor_y = floor_position_y + floor_length / 2 - corridor_width / 2
    corridor_id = room_id_offset + 10000 + floor_number * 100

    spatial_grid.place_room(
        room_id=corridor_id,
        x=floor_position_x,
        y=corridor_y,
        z=z_position,
        width=floor_width,
        length=corridor_width,
        height=floor_height,
        room_type="circulation",
        metadata={"name": f"Corridor Floor {floor_number}", "is_corridor": True},
    )

    # Create corridor room object
    corridor_room = Room(
        width=floor_width,
        length=corridor_width,
        height=floor_height,
        room_type="circulation",
        name=f"Corridor Floor {floor_number}",
        floor=floor_number,
        id=corridor_id,
        metadata={"is_corridor": True},
    )
    corridor_room.position = (floor_position_x, corridor_y, z_position)
    rooms.append(corridor_room)

    # Place vertical circulation (elevator and stairs)
    # Place in the middle of the floor
    elevator_width = 4.0
    elevator_length = 4.0
    stairs_width = 4.0
    stairs_length = 4.0

    # Position elevator in the center of the floor width
    elevator_x = floor_position_x + floor_width / 2 - elevator_width / 2
    elevator_y = corridor_y - elevator_length / 2  # Center relative to corridor

    # Ensure elevator ID is unique per floor using floor number
    elevator_id = room_id_offset + floor_number * 1000 + 1
    spatial_grid.place_room(
        room_id=elevator_id,
        x=elevator_x,
        y=elevator_y,
        z=z_position,
        width=elevator_width,
        length=elevator_length,
        height=floor_height,
        room_type="vertical_circulation",
        metadata={"name": f"Elevator Core Floor {floor_number}", "is_elevator": True},
    )

    # Create elevator room object
    elevator_room = Room(
        width=elevator_width,
        length=elevator_length,
        height=floor_height,
        room_type="vertical_circulation",
        name=f"Elevator Core Floor {floor_number}",
        floor=floor_number,
        id=elevator_id,
        metadata={"is_elevator": True},
    )
    elevator_room.position = (elevator_x, elevator_y, z_position)
    rooms.append(elevator_room)

    # Place stairs next to elevator
    stairs_x = elevator_x - stairs_width - 1.0  # 1m gap between elevator and stairs
    stairs_y = elevator_y

    # Use a floor-specific ID for stairs
    stairs_id = room_id_offset + floor_number * 1000 + 2
    spatial_grid.place_room(
        room_id=stairs_id,
        x=stairs_x,
        y=stairs_y,
        z=z_position,
        width=stairs_width,
        length=stairs_length,
        height=floor_height,
        room_type="vertical_circulation",
        metadata={"name": f"Emergency Stairs Floor {floor_number}", "is_stairs": True},
    )

    # Create stairs room object
    stairs_room = Room(
        width=stairs_width,
        length=stairs_length,
        height=floor_height,
        room_type="vertical_circulation",
        name=f"Emergency Stairs Floor {floor_number}",
        floor=floor_number,
        id=stairs_id,
        metadata={"is_stairs": True},
    )
    stairs_room.position = (stairs_x, stairs_y, z_position)
    rooms.append(stairs_room)

    # Place service rooms near the vertical circulation
    service_x = stairs_x - service_rooms["duty_room"]["width"] - 1.0
    service_y = stairs_y

    # Duty room - use floor-specific ID
    duty_room_id = room_id_offset + floor_number * 1000 + 3
    duty_width = service_rooms["duty_room"]["width"]
    duty_length = service_rooms["duty_room"]["length"]

    spatial_grid.place_room(
        room_id=duty_room_id,
        x=service_x,
        y=service_y,
        z=z_position,
        width=duty_width,
        length=duty_length,
        height=floor_height,
        room_type="service_area",
        metadata={
            "name": f"Duty Room Floor {floor_number}",
            "service_type": "duty_room",
        },
    )

    duty_room = Room(
        width=duty_width,
        length=duty_length,
        height=floor_height,
        room_type="service_area",
        name=f"Duty Room Floor {floor_number}",
        floor=floor_number,
        id=duty_room_id,
        metadata={"service_type": "duty_room"},
    )
    duty_room.position = (service_x, service_y, z_position)
    rooms.append(duty_room)

    # Linen room - use floor-specific ID
    linen_room_id = room_id_offset + floor_number * 1000 + 4
    linen_width = service_rooms["linen_room"]["width"]
    linen_length = service_rooms["linen_room"]["length"]
    linen_x = service_x
    linen_y = service_y + duty_length + 0.5  # 0.5m gap

    spatial_grid.place_room(
        room_id=linen_room_id,
        x=linen_x,
        y=linen_y,
        z=z_position,
        width=linen_width,
        length=linen_length,
        height=floor_height,
        room_type="service_area",
        metadata={
            "name": f"Linen Room Floor {floor_number}",
            "service_type": "linen_room",
        },
    )

    linen_room = Room(
        width=linen_width,
        length=linen_length,
        height=floor_height,
        room_type="service_area",
        name=f"Linen Room Floor {floor_number}",
        floor=floor_number,
        id=linen_room_id,
        metadata={"service_type": "linen_room"},
    )
    linen_room.position = (linen_x, linen_y, z_position)
    rooms.append(linen_room)

    # Staff bathroom
    bath_room_id = room_id_offset + 10005 + floor_number * 100
    bath_width = service_rooms["staff_bathroom"]["width"]
    bath_length = service_rooms["staff_bathroom"]["length"]
    bath_x = service_x
    bath_y = linen_y + linen_length + 0.5  # 0.5m gap

    spatial_grid.place_room(
        room_id=bath_room_id,
        x=bath_x,
        y=bath_y,
        z=z_position,
        width=bath_width,
        length=bath_length,
        height=floor_height,
        room_type="service_area",
        metadata={
            "name": f"Staff Bathroom Floor {floor_number}",
            "service_type": "staff_bathroom",
        },
    )

    bath_room = Room(
        width=bath_width,
        length=bath_length,
        height=floor_height,
        room_type="service_area",
        name=f"Staff Bathroom Floor {floor_number}",
        floor=floor_number,
        id=bath_room_id,
        metadata={"service_type": "staff_bathroom"},
    )
    bath_room.position = (bath_x, bath_y, z_position)
    rooms.append(bath_room)

    # Calculate usable width for guest rooms (adjust for circulation core)
    # The north side starts at elevator_x + elevator_width and extends to the end of the floor width
    usable_width_north = floor_width - (elevator_x - floor_position_x + elevator_width)

    # The south side starts at floor_position_x and extends to the service rooms
    # Need to account for the total width of circulation and service areas
    service_area_width = (
        elevator_x - floor_position_x
    ) + elevator_width  # Include elevator
    usable_width_south = floor_position_x + floor_width - service_area_width

    print(
        f"Available space - North side: {usable_width_north}m, South side: {usable_width_south}m"
    )

    # Place guest rooms on north side (right of core)
    north_start_x = elevator_x + elevator_width + 1.0

    # Calculate how many rooms can fit on north side
    print(f"North corridor: Starting at x={north_start_x}, width={usable_width_north}")
    room_count_north = generate_guest_rooms(
        spatial_grid=spatial_grid,
        rooms=rooms,
        start_id=room_id_offset + 10100 + floor_number * 100,
        start_x=north_start_x,
        start_y=corridor_y + corridor_width,  # North of corridor
        z=z_position,
        usable_width=usable_width_north,
        room_depth=room_depth,
        floor_height=floor_height,
        floor_number=floor_number,
        room_types_distribution=room_types_distribution,
        room_sizes=room_sizes,
        side="north",
    )

    # Place guest rooms on south side (left of core and service rooms)
    south_start_x = floor_position_x

    print(f"South corridor: Starting at x={south_start_x}, width={usable_width_south}")
    # Calculate how many rooms can fit on south side
    room_count_south = generate_guest_rooms(
        spatial_grid=spatial_grid,
        rooms=rooms,
        start_id=room_id_offset + 10200 + floor_number * 100,
        start_x=south_start_x,
        start_y=corridor_y - room_depth,  # South of corridor
        z=z_position,
        usable_width=usable_width_south,
        room_depth=room_depth,
        floor_height=floor_height,
        floor_number=floor_number,
        room_types_distribution=room_types_distribution,
        room_sizes=room_sizes,
        side="south",
    )

    # Print summary
    total_rooms = room_count_north + room_count_south
    print(
        f"Floor {floor_number}: Generated {total_rooms} guest rooms ({room_count_north} north + {room_count_south} south)"
    )

    return spatial_grid, rooms


def generate_guest_rooms(
    spatial_grid: SpatialGrid,
    rooms: List[Room],
    start_id: int,
    start_x: float,
    start_y: float,
    z: float,
    usable_width: float,
    room_depth: float,
    floor_height: float,
    floor_number: int,
    room_types_distribution: Dict[str, float],
    room_sizes: Dict[str, Dict[str, float]],
    side: str,
) -> int:
    """
    Generate guest rooms for one side of the corridor with proper type distribution.

    Args:
        spatial_grid: The spatial grid to place rooms in
        rooms: List to append created room objects to
        start_id: Starting room ID
        start_x, start_y, z: Starting coordinates
        usable_width: Available width for rooms
        room_depth: Depth of rooms
        floor_height: Height of floor
        floor_number: Floor number
        room_types_distribution: Distribution of room types
        room_sizes: Size information for each room type
        side: Which side of corridor ("north" or "south")

    Returns:
        int: Number of rooms placed
    """
    # Validate usable width - ensure we have meaningful space
    if usable_width <= 0:
        print(f"  No usable width available on {side} side")
        return 0

    print(f"  Generating rooms for {side} side - Available width: {usable_width}m")

    # Use a direct room spacing approach to maximize room count
    # For typical hotel design, we want minimal spacing between rooms
    room_spacing = 0.15  # 15cm between rooms for wall thickness

    # First, determine how many rooms of each type can fit
    # Start with smallest standard room width to calculate total possible rooms
    min_room_width = min(size_info["width"] for size_info in room_sizes.values())
    max_possible_rooms = int(
        (usable_width + room_spacing) / (min_room_width + room_spacing)
    )

    print(f"  Maximum possible rooms (if all minimum width): {max_possible_rooms}")

    if max_possible_rooms <= 0:
        print(f"  Cannot fit any rooms on {side} side")
        return 0

    # Determine target number for each room type based on distribution
    target_rooms = {}
    for room_type, percentage in room_types_distribution.items():
        target_rooms[room_type] = max(1, round(max_possible_rooms * percentage))

    # Adjust to ensure we have at least one of each type if space permits
    total_min_width = sum(
        room_sizes[room_type]["width"] for room_type in target_rooms.keys()
    )
    total_min_width += room_spacing * (len(target_rooms) - 1)  # Add spacing

    if total_min_width > usable_width:
        # Not enough space for one of each type, prioritize based on distribution
        sorted_types = sorted(
            room_types_distribution.items(), key=lambda x: x[1], reverse=True
        )
        target_rooms = {}
        remaining_width = usable_width

        for room_type, _ in sorted_types:
            width = room_sizes[room_type]["width"]
            if remaining_width >= width:
                target_rooms[room_type] = 1
                remaining_width -= width + room_spacing
            else:
                target_rooms[room_type] = 0

    # Calculate how many of each type we can actually fit
    remaining_width = usable_width
    room_counts = {room_type: 0 for room_type in target_rooms}

    # Prioritize distribution while ensuring we maximize room count
    while remaining_width > min_room_width:
        placed = False

        # Sort room types by how far behind target they are
        types_by_need = sorted(
            target_rooms.keys(),
            key=lambda rt: (
                room_counts[rt] / target_rooms[rt]
                if target_rooms[rt] > 0
                else float("inf")
            ),
        )

        for room_type in types_by_need:
            width = room_sizes[room_type]["width"]
            if remaining_width >= width + room_spacing:
                room_counts[room_type] += 1
                remaining_width -= width + room_spacing
                placed = True
                break

        if not placed:
            # Can't fit any more rooms
            break

    # Print distribution we achieved
    print(f"  Room distribution for {side} side:")
    total_rooms = sum(room_counts.values())
    for room_type, count in room_counts.items():
        if total_rooms > 0:
            percentage = count / total_rooms * 100
        else:
            percentage = 0
        print(f"    {room_type}: {count} rooms ({percentage:.1f}%)")

    # Generate the room placement sequence
    room_sequence = []
    for room_type, count in room_counts.items():
        room_sequence.extend([room_type] * count)

    # Arrange rooms strategically - suites near elevator, standard rooms distributed
    if "suite" in room_sequence:
        # Remove suites from sequence to place them strategically
        suites = room_sequence.count("suite")
        room_sequence = [rt for rt in room_sequence if rt != "suite"]

        # Put suites in strategic positions
        if side == "north":
            # On north side, put suites at beginning (near elevator)
            room_sequence = ["suite"] * suites + room_sequence
        else:
            # On south side, put suites at end (near elevator)
            room_sequence = room_sequence + ["suite"] * suites

    # Place the rooms
    current_x = start_x
    room_id = start_id

    for i, room_type in enumerate(room_sequence):
        room_width = room_sizes[room_type]["width"]
        room_length = room_sizes[room_type]["length"]

        # Double-check that this room fits in remaining space
        if current_x + room_width > start_x + usable_width:
            print(f"    Warning: Room {i+1} doesn't fit in available space - skipping")
            continue

        # Create unique room number based on floor and position
        room_number = (
            f"{floor_number}{100 + i:02d}"
            if side == "north"
            else f"{floor_number}{200 + i:02d}"
        )

        # Place room in spatial grid
        metadata = {"name": f"Room {room_number}", "room_type": room_type, "side": side}

        # Determine if this room has a view (exterior)
        has_exterior = (
            (side == "north" and start_y + room_depth >= spatial_grid.length)
            or (side == "south" and start_y <= 0)
            or (current_x <= 0)
            or (current_x + room_width >= spatial_grid.width)
        )

        metadata["has_exterior"] = has_exterior

        # Place the main room
        placement_success = spatial_grid.place_room(
            room_id=room_id,
            x=current_x,
            y=start_y,
            z=z,
            width=room_width,
            length=room_depth,  # Use standard depth for all rooms
            height=floor_height,
            room_type="guest_room",
            metadata=metadata,
        )

        if placement_success:
            # Create room object
            guest_room = Room(
                width=room_width,
                length=room_depth,
                height=floor_height,
                room_type="guest_room",
                name=metadata["name"],
                floor=floor_number,
                id=room_id,
                metadata=metadata,
            )
            guest_room.position = (current_x, start_y, z)
            rooms.append(guest_room)

            # Add bathroom to each room (implicit - could be separate objects)
            bathroom_area = 5.0  # 5m² per requirements

            # Update position for next room with proper spacing
            current_x += room_width + room_spacing
            room_id += 1
        else:
            print(
                f"    Failed to place room at ({current_x}, {start_y}) - possible collision"
            )

    # Return actual number of rooms placed
    return len(
        [
            r
            for r in rooms
            if r.room_type == "guest_room"
            and r.floor == floor_number
            and r.metadata.get("side") == side
        ]
    )


def generate_all_standard_floors(
    building_config: Dict[str, Any],
    template_path: Optional[str] = None,
    spatial_grid: Optional[SpatialGrid] = None,
    target_room_count: int = 380,  # From Chinese requirements: 380 rooms total
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate all standard floors for a hotel tower based on target room count.

    Args:
        building_config: Building configuration with standard floor parameters
        template_path: Optional path to a floor template JSON file
        spatial_grid: Optional existing spatial grid to add to
        target_room_count: Target number of guest rooms to generate

    Returns:
        Tuple containing the spatial grid and list of room objects
    """
    # Get standard floor range
    std_floor_config = building_config.get("standard_floor", {})
    start_floor = std_floor_config.get("start_floor", 5)
    end_floor = std_floor_config.get("end_floor", 20)

    # If no spatial grid provided, create a new one
    if spatial_grid is None:
        # Use the overall building parameters
        building_width = building_config.get("width", 60.0)
        building_length = building_config.get("length", 80.0)
        building_height = building_config.get("height", 100.0)
        grid_size = building_config.get("grid_size", 1.0)
        min_floor = building_config.get("min_floor", -2)
        floor_height = building_config.get("floor_height", 4.0)

        spatial_grid = SpatialGrid(
            width=building_width,
            length=building_length,
            height=building_height,
            grid_size=grid_size,
            min_floor=min_floor,
            floor_height=floor_height,
        )

    # Find highest room ID in existing layout to avoid conflicts
    room_id_offset = 0
    if hasattr(spatial_grid, "rooms") and spatial_grid.rooms:
        existing_ids = [int(room_id) for room_id in spatial_grid.rooms.keys()]
        if existing_ids:
            room_id_offset = max(existing_ids) + 1000

    print(f"Using room ID offset: {room_id_offset}")

    # Initialize list for all room objects
    all_rooms = []

    # Define a temporary spatial grid just for testing
    temp_building_width = building_config.get("width", 60.0)
    temp_building_length = building_config.get("length", 80.0)
    temp_building_height = building_config.get("height", 100.0)
    temp_grid_size = building_config.get("grid_size", 1.0)
    temp_min_floor = building_config.get("min_floor", -2)
    temp_floor_height = building_config.get("floor_height", 4.0)

    temp_grid = SpatialGrid(
        width=temp_building_width,
        length=temp_building_length,
        height=temp_building_height,
        grid_size=temp_grid_size,
        min_floor=temp_min_floor,
        floor_height=temp_floor_height,
    )

    # Generate a test floor to check how many rooms we can fit per floor
    print("\nGenerating test floor to calculate rooms per floor...")
    test_grid, test_rooms = generate_standard_floor(
        floor_number=start_floor,
        building_config=building_config,
        template_path=template_path,
        spatial_grid=temp_grid,
        room_id_offset=room_id_offset,
    )

    # Count guest rooms in the test floor
    guest_rooms_per_floor = sum(
        1 for room in test_rooms if room.room_type == "guest_room"
    )

    if guest_rooms_per_floor == 0:
        print("ERROR: Could not generate any guest rooms in test floor!")
        print(
            "Check building dimensions and room sizes - there may not be enough space"
        )
        # Return empty layout as fallback
        return spatial_grid, []

    # Calculate needed floor count to reach target room count
    required_floors = math.ceil(target_room_count / guest_rooms_per_floor)

    # Ensure we don't exceed the maximum available floors
    actual_end_floor = min(end_floor, start_floor + required_floors - 1)

    print(f"Guest rooms per floor: {guest_rooms_per_floor}")
    print(f"Required floors to reach {target_room_count} rooms: {required_floors}")
    print(f"Generating floors {start_floor} to {actual_end_floor}")

    # Generate actual floors in the final spatial grid
    for floor in range(start_floor, actual_end_floor + 1):
        print(f"\nGenerating floor {floor}...")
        _, floor_rooms = generate_standard_floor(
            floor_number=floor,
            building_config=building_config,
            template_path=template_path,
            spatial_grid=spatial_grid,
            room_id_offset=room_id_offset,
        )
        all_rooms.extend(floor_rooms)

        # Count how many guest rooms we have so far
        guest_rooms_so_far = sum(
            1 for room in all_rooms if room.room_type == "guest_room"
        )
        print(
            f"Progress: {guest_rooms_so_far}/{target_room_count} guest rooms generated ({guest_rooms_so_far/target_room_count*100:.1f}%)"
        )

    # Final count of total guest rooms generated
    total_guest_rooms = sum(1 for room in all_rooms if room.room_type == "guest_room")
    print(
        f"\nFinal tally: {total_guest_rooms} guest rooms generated across {actual_end_floor-start_floor+1} floors"
    )

    # Summary of room types
    room_types_count = {}
    for room in all_rooms:
        if (
            room.room_type == "guest_room"
            and hasattr(room, "metadata")
            and room.metadata
        ):
            room_subtype = room.metadata.get("room_type", "unknown")
            if room_subtype not in room_types_count:
                room_types_count[room_subtype] = 0
            room_types_count[room_subtype] += 1

    if total_guest_rooms > 0:
        print("\nGuest room types distribution:")
        for room_type, count in room_types_count.items():
            print(f"  {room_type}: {count} rooms ({count/total_guest_rooms*100:.1f}%)")

    # Calculate total area
    total_area = 0
    for room in all_rooms:
        if room.room_type == "guest_room":
            total_area += room.width * room.length

    print(f"\nTotal guest room area: {total_area:.1f} m²")
    if total_guest_rooms > 0:
        print(f"Average area per guest room: {total_area/total_guest_rooms:.1f} m²")

    return spatial_grid, all_rooms
