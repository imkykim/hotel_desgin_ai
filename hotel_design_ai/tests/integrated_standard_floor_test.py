#!/usr/bin/env python
"""
Integrated test script for generating and visualizing a complete hotel design
with both podium (裙房) and standard floor (tower) sections.

This script demonstrates how to combine podium and tower portions
into a complete hotel layout design.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import hotel design AI modules
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.core.grid_rule_engine import RuleEngine
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import export_to_json
from hotel_design_ai.config.config_loader import (
    get_building_envelope,
    create_room_objects_from_program,
)
from hotel_design_ai.utils.metrics import LayoutMetrics

# Import standard floor module
# In a real implementation, this would be part of the hotel_design_ai package
from hotel_design_ai.core.standard_floor_generator import (
    generate_standard_floor,
    generate_all_standard_floors,
)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Integrated Hotel Design Generator")

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
        "--podium-only",
        action="store_true",
        help="Generate only the podium section (no tower)",
    )

    parser.add_argument(
        "--tower-only",
        action="store_true",
        help="Generate only the tower section (no podium)",
    )

    parser.add_argument("--visualize", action="store_true", help="Show visualizations")

    parser.add_argument(
        "--save-images", action="store_true", help="Save visualization images"
    )

    parser.add_argument(
        "--sample-floor", type=int, help="Render only a specific standard floor"
    )

    return parser.parse_args()


def generate_podium_layout(building_config, program_config):
    """
    Generate the podium (裙房) section of the hotel.

    Args:
        building_config: Building configuration
        program_config: Program configuration

    Returns:
        Tuple of (spatial_grid, rooms)
    """
    print("\nGenerating podium (裙房) layout...")

    # Get podium floor range from the new building config format
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

    # Replace spatial grid
    rule_engine.spatial_grid = spatial_grid

    # Create rooms from program requirements
    room_dicts = create_room_objects_from_program(program_config)

    # Filter for podium floors only
    podium_room_dicts = []
    for room_dict in room_dicts:
        # Check if room is in podium floors
        floor = room_dict.get("floor")
        if isinstance(floor, list):
            # If list of floors, check if any are in podium range
            floors_in_podium = [f for f in floor if min_floor <= f <= max_floor]
            if floors_in_podium:
                # Clone room_dict and set floor to first podium floor
                podium_room = room_dict.copy()
                podium_room["floor"] = floors_in_podium[0]
                podium_room_dicts.append(podium_room)
        elif floor is not None and min_floor <= floor <= max_floor:
            # Single floor value in podium range
            podium_room_dicts.append(room_dict)

    print(f"Found {len(podium_room_dicts)} rooms for podium section")

    # Convert to Room objects using helper function
    from main import convert_room_dicts_to_room_objects

    rooms = convert_room_dicts_to_room_objects(podium_room_dicts)

    # Generate layout
    start_time = time.time()
    podium_layout = rule_engine.generate_layout(rooms)
    end_time = time.time()

    print(f"Podium layout generated in {end_time - start_time:.2f} seconds")
    print(f"Placed {len(podium_layout.rooms)} rooms in podium section")

    return podium_layout, rooms


def generate_tower_layout(building_config, existing_layout=None):
    """
    Generate the tower section (standard floors) of the hotel.

    Args:
        building_config: Building configuration
        existing_layout: Optional existing layout to add to

    Returns:
        Tuple of (spatial_grid, rooms)
    """
    print("\nGenerating tower (standard floors) layout...")

    # Get building parameters
    width = building_config["width"]
    length = building_config["length"]
    height = building_config["height"]
    grid_size = building_config["grid_size"]
    min_floor = building_config.get("min_floor", -2)
    floor_height = building_config["floor_height"]

    # Use existing spatial grid or create new one
    if existing_layout:
        spatial_grid = existing_layout
    else:
        spatial_grid = SpatialGrid(
            width=width,
            length=length,
            height=height,
            grid_size=grid_size,
            min_floor=min_floor,
            floor_height=floor_height,
        )

    # Generate standard floors
    start_time = time.time()
    tower_layout, rooms = generate_all_standard_floors(
        building_config=building_config, spatial_grid=spatial_grid
    )
    end_time = time.time()

    print(f"Tower layout generated in {end_time - start_time:.2f} seconds")
    print(f"Added {len(rooms)} rooms to tower section")

    return tower_layout, rooms


def main():
    """Main function"""
    print("Integrated Hotel Design Generator")
    print("================================")

    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get building configuration
    building_config = get_building_envelope(args.building_config)

    # Set up spatial grid that will hold the complete design
    width = building_config["width"]
    length = building_config["length"]
    height = building_config["height"]
    grid_size = building_config["grid_size"]
    min_floor = building_config.get("min_floor", -2)
    floor_height = building_config["floor_height"]

    complete_layout = SpatialGrid(
        width=width,
        length=length,
        height=height,
        grid_size=grid_size,
        min_floor=min_floor,
        floor_height=floor_height,
    )

    all_rooms = []

    # Generate podium section if requested
    if not args.tower_only:
        podium_layout, podium_rooms = generate_podium_layout(
            building_config, args.program_config
        )

        # Copy podium rooms to complete layout
        for room_id, room_data in podium_layout.rooms.items():
            complete_layout.rooms[room_id] = room_data.copy()

        all_rooms.extend(podium_rooms)

    # Generate tower section if requested
    if not args.podium_only:
        tower_layout, tower_rooms = generate_tower_layout(
            building_config, complete_layout
        )

        # If using separate layouts, copy tower rooms to complete layout
        if args.tower_only:
            for room_id, room_data in tower_layout.rooms.items():
                complete_layout.rooms[room_id] = room_data.copy()

        all_rooms.extend(tower_rooms)

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export to JSON
    json_file = os.path.join(args.output, f"hotel_design_{timestamp}.json")
    export_to_json(complete_layout, json_file)
    print(f"\nExported complete layout to: {json_file}")

    # Print basic metrics
    print("\nLayout metrics:")
    metrics = LayoutMetrics(complete_layout, building_config=building_config)
    space_utilization = metrics.space_utilization() * 100
    print(f"Space utilization: {space_utilization:.1f}%")

    # Count rooms by type
    room_types = {}
    for room_id, room_data in complete_layout.rooms.items():
        room_type = room_data["type"]
        if room_type not in room_types:
            room_types[room_type] = 0
        room_types[room_type] += 1

    print("\nRoom counts by type:")
    for room_type, count in sorted(room_types.items()):
        print(f"  {room_type}: {count}")

    # Visualize layout
    if args.visualize or args.save_images:
        print("\nCreating visualizations...")

        # Create renderer
        renderer = LayoutRenderer(complete_layout, building_config=building_config)

        # Render 3D view
        fig_3d, ax_3d = renderer.render_3d(show_labels=True)
        fig_3d.suptitle("Complete Hotel Design - 3D View")

        # Determine which floors to render
        if args.sample_floor:
            # Render only the specified standard floor
            floors_to_render = [args.sample_floor]
        else:
            # Render one floor from each section using updated building config format
            podium_config = building_config.get("podium", {})
            podium_min = podium_config.get(
                "min_floor", building_config.get("min_floor", -2)
            )
            podium_max = podium_config.get("max_floor", 4)

            std_floor_config = building_config.get("standard_floor", {})
            std_min = std_floor_config.get("start_floor", 5)
            std_max = std_floor_config.get("end_floor", 20)

            # Ensure there's no overlap
            if podium_max >= std_min:
                podium_max = std_min - 1

            # Pick representative floors
            floors_to_render = [
                podium_min,  # Lowest basement
                0,  # Ground floor
                podium_max,  # Top of podium
                std_min,  # First standard floor
                (std_min + std_max) // 2,  # Middle standard floor
                std_max,  # Top floor
            ]

        # Render floor plans
        floor_figs = []
        for floor in floors_to_render:
            fig, ax = renderer.render_floor_plan(floor=floor)
            fig.suptitle(f"Hotel Design - Floor {floor}")
            floor_figs.append((fig, floor))

        # Save images if requested
        if args.save_images:
            # Save 3D view
            fig_3d.savefig(
                os.path.join(args.output, f"hotel_design_3d_{timestamp}.png"),
                dpi=300,
                bbox_inches="tight",
            )

            # Save floor plans
            for fig, floor in floor_figs:
                # Name basements and standard floors appropriately
                if floor < 0:
                    floor_name = f"basement{abs(floor)}"
                elif floor >= building_config.get("standard_floor", {}).get(
                    "start_floor", 5
                ):
                    floor_name = f"std_floor_{floor}"
                else:
                    floor_name = f"floor_{floor}"

                fig.savefig(
                    os.path.join(
                        args.output, f"hotel_design_{floor_name}_{timestamp}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

            print(f"Saved visualizations to {args.output}")

        # Show visualizations if requested
        if args.visualize:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            # Close figures to free memory
            import matplotlib.pyplot as plt

            plt.close(fig_3d)
            for fig, _ in floor_figs:
                plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
