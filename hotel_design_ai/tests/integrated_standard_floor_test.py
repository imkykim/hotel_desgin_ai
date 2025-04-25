#!/usr/bin/env python
"""
Example script for integrating the improved standard floor generator with the hotel_design_ai system.
This script demonstrates how to:
1. Generate a podium layout first
2. Find the vertical circulation core
3. Generate standard floors based on the core
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import required modules
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.core.rule_engine import RuleEngine
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import export_to_json
from hotel_design_ai.config.config_loader import (
    get_building_envelope,
    create_room_objects_from_program,
)
from hotel_design_ai.config.config_loader_grid import create_room_objects_for_section
from hotel_design_ai.models.room import Room
from hotel_design_ai.utils.metrics import LayoutMetrics

# Import the improved standard floor generator
from hotel_design_ai.core.standard_floor_generator import (
    find_vertical_circulation_core,
    generate_standard_floor,
    generate_all_standard_floors,
)


def convert_room_dicts_to_room_objects(room_dicts):
    """Convert room dictionaries to Room objects"""
    rooms = []

    for room_dict in room_dicts:
        # Create room metadata by combining all available metadata
        metadata = {"department": room_dict["department"], "id": room_dict["id"]}

        # Preserve original name if present
        if "metadata" in room_dict and room_dict["metadata"]:
            if "original_name" in room_dict["metadata"]:
                metadata["original_name"] = room_dict["metadata"]["original_name"]

        # If no original_name in metadata but has name, use it
        if "original_name" not in metadata and "name" in room_dict:
            metadata["original_name"] = room_dict["name"]

        room = Room(
            width=room_dict["width"],
            length=room_dict["length"],
            height=room_dict["height"],
            room_type=room_dict["room_type"],
            name=room_dict["name"],
            floor=room_dict.get("floor"),
            requires_natural_light=room_dict.get("requires_natural_light", False),
            requires_exterior_access=False,  # Default value
            preferred_adjacencies=room_dict.get("requires_adjacency", []),
            avoid_adjacencies=[],  # Default value
            metadata=metadata,  # Use the complete metadata
            id=room_dict["id"],
        )

        rooms.append(room)

    return rooms


def generate_podium_layout(building_config, program_config, fixed_positions=None):
    """
    Generate the podium (裙房) section of the hotel.

    Args:
        building_config: Building configuration
        program_config: Program configuration
        fixed_positions: Optional dictionary of fixed room positions

    Returns:
        Tuple of (spatial_grid, rooms)
    """
    print("\nGenerating podium (裙房) layout...")

    # Get podium floor range from the building config
    podium_config = building_config.get("podium", {})
    min_floor = podium_config.get("min_floor", building_config.get("min_floor", -2))
    max_floor = podium_config.get("max_floor", 4)

    # Ensure there's no overlap with standard floors
    std_floor_config = building_config.get("standard_floor", {})
    std_start_floor = std_floor_config.get("start_floor", 5)

    # Adjust podium max floor if needed
    if max_floor >= std_start_floor:
        max_floor = std_start_floor - 1
        print(
            f"Adjusted podium max floor to {max_floor} to avoid overlap with tower section"
        )

    # Get building parameters
    width = building_config["width"]
    length = building_config["length"]
    height = building_config["height"]
    grid_size = building_config["grid_size"]
    structural_grid = (
        building_config["structural_grid_x"],
        building_config["structural_grid_y"],
    )
    floor_height = building_config["floor_height"]

    # Create spatial grid
    spatial_grid = SpatialGrid(
        width=width,
        length=length,
        height=height,
        grid_size=grid_size,
        min_floor=min_floor,
        floor_height=floor_height,
    )

    # Create rule engine
    rule_engine = RuleEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid,
        building_config=building_config,
    )

    # Replace spatial grid to ensure proper support
    rule_engine.spatial_grid = spatial_grid

    # Filter room dicts for podium section
    podium_room_dicts = create_room_objects_for_section(
        program_config, building_config, section="podium"
    )

    # Convert to Room objects
    rooms = convert_room_dicts_to_room_objects(podium_room_dicts)

    # Mark core circulation elements in metadata
    for room in rooms:
        if room.room_type == "vertical_circulation":
            if not room.metadata:
                room.metadata = {}
            # Mark main vertical circulation cores as "is_core"
            if "main" in room.name.lower() or "core" in room.name.lower():
                room.metadata["is_core"] = True

    # Apply fixed positions if provided
    if fixed_positions:
        # Create a copy of rooms to avoid modifying the original list
        modified_rooms = []
        for room in rooms:
            room_copy = Room.from_dict(room.to_dict())
            if room.id in fixed_positions:
                room_copy.position = fixed_positions[room.id]
            modified_rooms.append(room_copy)
        rooms = modified_rooms

    # Generate layout
    start_time = time.time()
    layout = rule_engine.generate_layout(rooms)
    end_time = time.time()

    print(f"Podium layout generated in {end_time - start_time:.2f} seconds")
    print(f"Placed {len(layout.rooms)} rooms in podium section")

    return layout, rooms


def generate_complete_hotel_layout(args):
    """
    Generate a complete hotel layout with both podium and standard floor sections.
    Using the improved standard floor generator.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (combined_layout, all_rooms)
    """
    print("\nGenerating complete hotel layout...")

    # Get building configuration
    building_config = get_building_envelope(args.building_config)

    # Step 1: Generate podium section
    print("Step 1: Generating podium (裙房) section...")
    podium_layout, podium_rooms = generate_podium_layout(
        building_config, args.program_config, args.fixed_positions
    )

    # Print debug info about the podium layout
    podium_floors = {}
    for room_id, room_data in podium_layout.rooms.items():
        floor = int(room_data["position"][2] / building_config["floor_height"])
        if floor not in podium_floors:
            podium_floors[floor] = 0
        podium_floors[floor] += 1

    print("\nPodium layout room distribution by floor:")
    for floor in sorted(podium_floors.keys()):
        print(f"  Floor {floor}: {podium_floors[floor]} rooms")

    # Step 2: Find vertical circulation core
    print("\nStep 2: Finding vertical circulation core from podium...")
    circulation_core = find_vertical_circulation_core(podium_layout, building_config)

    if circulation_core:
        print(f"Found vertical circulation core to extend to standard floors:")
        print(f"  Position: {circulation_core['position']}")
        print(f"  Dimensions: {circulation_core['dimensions']}")
        if "metadata" in circulation_core and circulation_core["metadata"]:
            print(f"  Name: {circulation_core['metadata'].get('name', 'Unnamed')}")
    else:
        print("No suitable vertical circulation core found in podium")
        print("Will create a new core for standard floors")

    # Step 3: Generate standard floor section
    print("\nStep 3: Generating standard floor (tower) section...")

    # Set target room count
    target_room_count = args.target_room_count

    # Use our improved generator for standard floors
    standard_layout, standard_rooms = generate_all_standard_floors(
        building_config=building_config,
        spatial_grid=podium_layout,  # Use podium as starting point
        target_room_count=target_room_count,
    )

    # Print debug info after standard floor generation
    print(f"Generated {len(standard_rooms)} rooms on standard floors")
    print(f"Total rooms after standard floors: {len(standard_layout.rooms)}")

    # Count guest rooms
    guest_rooms = sum(1 for room in standard_rooms if room.room_type == "guest_room")
    print(f"Generated {guest_rooms} guest rooms on standard floors")

    # Combine all rooms (podium + standard floors)
    all_rooms = podium_rooms + standard_rooms

    print(f"\nComplete layout generated with {len(standard_layout.rooms)} rooms:")
    print(f"  - Podium section: {len(podium_layout.rooms)} rooms")
    print(f"  - Standard floor section: {len(standard_rooms)} rooms")

    # Calculate metrics
    room_types = {}
    for room_id, room_data in standard_layout.rooms.items():
        room_type = room_data["type"]
        if room_type not in room_types:
            room_types[room_type] = 0
        room_types[room_type] += 1

    print("\nRoom type distribution:")
    for room_type, count in sorted(room_types.items()):
        print(f"  {room_type}: {count} rooms")

    return standard_layout, all_rooms


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Hotel Design AI - Improved Standard Floor Generator Example"
    )

    # Basic arguments
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--building-config",
        type=str,
        default="default",
        help="Building configuration name",
    )
    parser.add_argument(
        "--program-config",
        type=str,
        default="default",
        help="Program configuration name",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualizations",
    )
    parser.add_argument(
        "--target-room-count",
        type=int,
        default=380,
        help="Target number of guest rooms",
    )
    parser.add_argument(
        "--fixed-positions",
        type=str,
        default=None,
        help="JSON file with fixed room positions",
    )

    args = parser.parse_args()

    # Load fixed positions if specified
    if args.fixed_positions and os.path.exists(args.fixed_positions):
        with open(args.fixed_positions, "r") as f:
            args.fixed_positions = json.load(f)
    else:
        args.fixed_positions = {}

    # Generate complete hotel layout (podium + standard floors)
    layout, all_rooms = generate_complete_hotel_layout(args)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save the layout to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"hotel_layout_{timestamp}.json")
    export_to_json(layout, output_file)
    print(f"\nLayout exported to: {output_file}")

    # Visualize the layout if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            # Get building parameters
            building_config = get_building_envelope(args.building_config)

            # Create renderer
            renderer = LayoutRenderer(layout, building_config=building_config)

            # Create 3D visualization
            fig1, ax1 = renderer.render_3d(show_labels=True)
            fig1.suptitle("Complete Hotel Layout - 3D View")

            # Get standard floor info
            std_floor_config = building_config.get("standard_floor", {})
            start_floor = std_floor_config.get("start_floor", 5)

            # Create floor plans for key floors
            # Ground floor (podium)
            fig2, ax2 = renderer.render_floor_plan(floor=0)
            fig2.suptitle("Hotel Layout - Ground Floor (Podium)")

            # First standard floor
            fig3, ax3 = renderer.render_floor_plan(floor=start_floor)
            fig3.suptitle(f"Hotel Layout - Floor {start_floor} (First Standard Floor)")

            plt.show()
        except ImportError:
            print("Could not visualize: matplotlib is required")
            print("Install matplotlib with: pip install matplotlib")


if __name__ == "__main__":
    main()
