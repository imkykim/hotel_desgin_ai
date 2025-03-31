"""
Standard floor generator for hotel tower designs.
This module provides functions to generate standard floor layouts with guest rooms.
"""

from typing import Dict, List, Tuple, Any, Optional
import json
import os
import numpy as np

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


def generate_standard_floor(
    floor_number: int,
    building_config: Dict[str, Any],
    template_path: Optional[str] = None,
    spatial_grid: Optional[SpatialGrid] = None,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate a standard floor layout for a hotel tower.

    Args:
        floor_number: The floor number to generate
        building_config: Building configuration with standard floor parameters
        template_path: Optional path to a floor template JSON file
        spatial_grid: Optional existing spatial grid to add to

    Returns:
        Tuple containing the spatial grid and list of room objects
    """
    # Get standard floor parameters from building config with the new format
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

    # Initialize list for room objects
    rooms = []

    # Place corridor
    corridor_y = floor_position_y + floor_length / 2 - corridor_width / 2
    corridor_id = 10000 + floor_number * 100

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

    elevator_x = floor_position_x + floor_width / 2 - elevator_width / 2
    elevator_y = corridor_y - elevator_length / 2  # Center relative to corridor

    elevator_id = 10001 + floor_number * 100
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

    stairs_id = 10002 + floor_number * 100
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

    # Calculate number of rooms that can fit on each side of corridor
    # Standard room parameters
    room_width = 4.0  # Standard module width

    # Calculate available width after elevators and stairs
    usable_width = floor_width

    # Number of standard rooms that can fit
    num_rooms_per_side = int(usable_width / room_width)

    # Place rooms on north side of corridor
    for i in range(num_rooms_per_side):
        room_id = 10100 + floor_number * 100 + i
        room_x = floor_position_x + i * room_width
        room_y = corridor_y + corridor_width  # North of corridor

        # Determine if this should be a corner/special room
        is_corner = i == 0 or i == num_rooms_per_side - 1
        room_type = "guest_room"
        room_name = f"Room {floor_number}{i+1:02d}"

        # Corner rooms might be larger
        actual_room_width = room_width * 1.5 if is_corner else room_width

        # Skip if we'd exceed the floor width
        if room_x + actual_room_width > floor_position_x + floor_width:
            continue

        # Place room in spatial grid
        spatial_grid.place_room(
            room_id=room_id,
            x=room_x,
            y=room_y,
            z=z_position,
            width=actual_room_width,
            length=room_depth,
            height=floor_height,
            room_type=room_type,
            metadata={"name": room_name, "is_corner": is_corner, "side": "north"},
        )

        # Create room object
        guest_room = Room(
            width=actual_room_width,
            length=room_depth,
            height=floor_height,
            room_type=room_type,
            name=room_name,
            floor=floor_number,
            id=room_id,
            metadata={"is_corner": is_corner, "side": "north"},
        )
        guest_room.position = (room_x, room_y, z_position)
        rooms.append(guest_room)

    # Place rooms on south side of corridor
    for i in range(num_rooms_per_side):
        room_id = 10200 + floor_number * 100 + i
        room_x = floor_position_x + i * room_width
        room_y = corridor_y - room_depth  # South of corridor

        # Determine if this should be a corner/special room
        is_corner = i == 0 or i == num_rooms_per_side - 1
        room_type = "guest_room"
        room_name = f"Room {floor_number}{num_rooms_per_side+i+1:02d}"

        # Corner rooms might be larger
        actual_room_width = room_width * 1.5 if is_corner else room_width

        # Skip if we'd exceed the floor width
        if room_x + actual_room_width > floor_position_x + floor_width:
            continue

        # Place room in spatial grid
        spatial_grid.place_room(
            room_id=room_id,
            x=room_x,
            y=room_y,
            z=z_position,
            width=actual_room_width,
            length=room_depth,
            height=floor_height,
            room_type=room_type,
            metadata={"name": room_name, "is_corner": is_corner, "side": "south"},
        )

        # Create room object
        guest_room = Room(
            width=actual_room_width,
            length=room_depth,
            height=floor_height,
            room_type=room_type,
            name=room_name,
            floor=floor_number,
            id=room_id,
            metadata={"is_corner": is_corner, "side": "south"},
        )
        guest_room.position = (room_x, room_y, z_position)
        rooms.append(guest_room)

    return spatial_grid, rooms


def generate_all_standard_floors(
    building_config: Dict[str, Any],
    template_path: Optional[str] = None,
    spatial_grid: Optional[SpatialGrid] = None,
) -> Tuple[SpatialGrid, List[Room]]:
    """
    Generate all standard floors for a hotel tower.

    Args:
        building_config: Building configuration with standard floor parameters
        template_path: Optional path to a floor template JSON file
        spatial_grid: Optional existing spatial grid to add to

    Returns:
        Tuple containing the spatial grid and list of room objects
    """
    # Get standard floor range
    std_floor_config = building_config.get("standard_floor", {})
    start_floor = std_floor_config.get("start_floor", 5)
    end_floor = std_floor_config.get("end_floor", 20)

    # Get overall building parameters if needed
    if spatial_grid is None:
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

    # Initialize list for all rooms
    all_rooms = []

    # Generate each standard floor
    for floor in range(start_floor, end_floor + 1):
        _, floor_rooms = generate_standard_floor(
            floor_number=floor,
            building_config=building_config,
            template_path=template_path,
            spatial_grid=spatial_grid,
        )
        all_rooms.extend(floor_rooms)

    return spatial_grid, all_rooms
