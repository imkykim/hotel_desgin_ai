#!/usr/bin/env python
"""
Test script for generating and visualizing standard floor layouts.
This script demonstrates the generation of standard guest room floors
in the tower portion of a hotel design.
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import hotel design AI modules
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import export_to_json
from hotel_design_ai.config.config_loader import get_building_envelope

# Import standard floor generator
from hotel_design_ai.core.standard_floor_generator import (
    generate_standard_floor,
    generate_all_standard_floors,
)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Standard Floor Generator")

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
        "--template", type=str, default=None, help="Path to floor template JSON file"
    )

    parser.add_argument(
        "--floor",
        type=int,
        default=5,
        help="Floor number to generate (if not generating all)",
    )

    parser.add_argument(
        "--all-floors", action="store_true", help="Generate all standard floors"
    )

    parser.add_argument("--visualize", action="store_true", help="Show visualizations")

    parser.add_argument(
        "--save-images", action="store_true", help="Save visualization images"
    )

    return parser.parse_args()


def main():
    """Main function"""
    print("Standard Floor Generator Test")
    print("===========================")

    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get building configuration
    building_config = get_building_envelope(args.building_config)

    # Generate standard floor(s)
    if args.all_floors:
        print(f"Generating all standard floors...")
        spatial_grid, rooms = generate_all_standard_floors(
            building_config=building_config, template_path=args.template
        )
        print(f"Generated {len(rooms)} rooms across all standard floors")
    else:
        print(f"Generating standard floor {args.floor}...")
        spatial_grid, rooms = generate_standard_floor(
            floor_number=args.floor,
            building_config=building_config,
            template_path=args.template,
        )
        print(f"Generated {len(rooms)} rooms on floor {args.floor}")

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export to JSON
    json_file = os.path.join(args.output, f"standard_floor_{timestamp}.json")
    export_to_json(spatial_grid, json_file)
    print(f"Exported layout to: {json_file}")

    # Visualize layout
    if args.visualize or args.save_images:
        print("Creating visualizations...")

        # Create renderer
        renderer = LayoutRenderer(spatial_grid, building_config=building_config)

        # Render 3D view
        fig_3d, ax_3d = renderer.render_3d(show_labels=True)
        fig_3d.suptitle("Standard Floor Layout - 3D View")

        # Determine floor range to render
        if args.all_floors:
            std_floor_config = building_config.get("standard_floor", {})
            start_floor = std_floor_config.get("start_floor", 5)
            end_floor = std_floor_config.get("end_floor", 20)
            floor_range = range(start_floor, end_floor + 1)
        else:
            floor_range = [args.floor]

        # Render floor plans
        floor_figs = []
        for floor in floor_range:
            fig, ax = renderer.render_floor_plan(floor=floor)
            fig.suptitle(f"Standard Floor Layout - Floor {floor}")
            floor_figs.append((fig, floor))

        # Save images if requested
        if args.save_images:
            # Save 3D view
            fig_3d.savefig(
                os.path.join(args.output, f"standard_floor_3d_{timestamp}.png"),
                dpi=300,
                bbox_inches="tight",
            )

            # Save floor plans
            for fig, floor in floor_figs:
                fig.savefig(
                    os.path.join(
                        args.output, f"standard_floor_{floor}_{timestamp}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

            print(f"Saved visualizations to {args.output}")

        # Show visualizations if requested
        if args.visualize:
            plt.show()
        else:
            # Close figures to free memory
            plt.close(fig_3d)
            for fig, _ in floor_figs:
                plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
