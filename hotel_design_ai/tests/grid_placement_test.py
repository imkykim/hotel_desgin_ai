#!/usr/bin/env python3
"""
Example demonstrating the GridPlacementEngine for hotel layout generation.
This script creates a hotel layout using grid-aligned room placement.
"""

import os
import sys
import argparse
from datetime import datetime

# Add the parent directory to path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from hotel_design_ai.core.grid_engine import GridPlacementEngine
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import export_to_json
from hotel_design_ai.config.config_loader import create_room_objects_from_program
from hotel_design_ai.config.config_loader import get_building_envelope
from hotel_design_ai.config.config_loader_grid import adjust_room_list_to_grid
from hotel_design_ai.models.room import Room


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


def generate_hotel_layout(args):
    """Generate a hotel layout using the GridPlacementEngine"""
    print("\nGenerating layout with GridPlacementEngine...")

    # Get building envelope parameters
    building_config = get_building_envelope(args.building_config)
    width = building_config["width"]
    length = building_config["length"]
    height = building_config["height"]
    grid_size = building_config["grid_size"]
    structural_grid = (
        building_config["structural_grid_x"],
        building_config["structural_grid_y"],
    )

    # Create rooms from program requirements
    room_dicts = create_room_objects_from_program(args.program_config)

    # Adjust room dimensions to align with grid
    adjusted_room_dicts = adjust_room_list_to_grid(
        room_dicts, structural_grid[0], structural_grid[1], args.grid_fraction
    )

    # Convert to Room objects
    rooms = convert_room_dicts_to_room_objects(adjusted_room_dicts)

    # Initialize grid placement engine
    grid_engine = GridPlacementEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid,
        grid_fraction=args.grid_fraction,
        building_config=building_config,
    )

    # Generate layout
    layout = grid_engine.generate_layout(rooms)

    print(f"Generated layout with {len(layout.rooms)} rooms")

    # Calculate and print space utilization
    space_util = layout.calculate_space_utilization() * 100
    print(f"Space utilization: {space_util:.1f}%")

    return layout


def visualize_layout(layout, args):
    """Visualize the generated layout"""
    if not args.visualize:
        return

    print("\nCreating visualizations...")

    # Get building parameters
    building_config = get_building_envelope(args.building_config)

    # Create renderer
    renderer = LayoutRenderer(layout, building_config=building_config)

    # Render 3D view
    fig1, ax1 = renderer.render_3d(show_labels=True)
    fig1.suptitle("Hotel Layout (Grid-Aligned) - 3D View")

    # Render floor plans
    for floor in range(
        building_config.get("min_floor", -1), building_config.get("max_floor", 3) + 1
    ):
        fig, ax = renderer.render_floor_plan(floor=floor)
        floor_name = "Basement" if floor < 0 else f"Floor {floor}"
        fig.suptitle(f"Hotel Layout (Grid-Aligned) - {floor_name}")

    # Show all figures
    import matplotlib.pyplot as plt

    plt.show()


def save_output(layout, args):
    """Save the layout to output files"""
    print("\nSaving output files...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export to JSON
    json_file = os.path.join(args.output, f"grid_aligned_layout_{timestamp}.json")
    export_to_json(layout, json_file)
    print(f"Saved layout to {json_file}")

    # Save visualizations
    if args.save_images:
        print("Saving visualizations...")

        # Get building parameters
        building_config = get_building_envelope(args.building_config)

        # Create renderer
        renderer = LayoutRenderer(layout, building_config=building_config)

        # Save renders
        renderer.save_renders(
            output_dir=args.output,
            prefix=f"grid_aligned_{timestamp}",
            include_3d=True,
            include_floor_plans=True,
        )

        print(f"Saved visualizations to {args.output}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate hotel layout with grid alignment"
    )

    parser.add_argument("--output", type=str, default="output", help="Output directory")
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
        "--grid-fraction",
        type=float,
        default=0.5,
        help="Grid fraction for alignment (0.5 = half grid)",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visualizations")
    parser.add_argument(
        "--save-images", action="store_true", help="Save visualization images"
    )

    return parser.parse_args()


def main():
    """Main function"""
    print("Grid-Aligned Hotel Layout Generator")
    print("==================================")

    # Parse arguments
    args = parse_arguments()

    # Generate layout
    layout = generate_hotel_layout(args)

    # Visualize layout
    visualize_layout(layout, args)

    # Save output
    save_output(layout, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
