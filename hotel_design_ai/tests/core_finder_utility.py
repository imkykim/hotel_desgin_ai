#!/usr/bin/env python
"""
Utility script to find and analyze vertical circulation cores in a hotel layout.
This is useful for diagnosing issues with standard floor generation.

Run this script to analyze an existing layout and identify the circulation core:
python core_finder_utility.py --layout path/to/layout.json
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import required modules
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.config.config_loader import get_building_envelope

# Import the improved standard floor generator
from improved_standard_floor_generator import find_vertical_circulation_core


def load_layout(filepath: str) -> Optional[SpatialGrid]:
    """
    Load a layout from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Optional[SpatialGrid]: Loaded layout or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        if "spatial_grid" in data:
            # Layout format
            spatial_grid_data = data["spatial_grid"]
        else:
            # Direct SpatialGrid format
            spatial_grid_data = data

        # Create a new spatial grid from the data
        spatial_grid = SpatialGrid.from_dict(spatial_grid_data)
        return spatial_grid
    except Exception as e:
        print(f"Error loading layout: {e}")
        return None


def find_circulation_cores(
    layout: SpatialGrid, building_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Find all vertical circulation cores in the layout.

    Args:
        layout: The spatial grid to analyze
        building_config: Building configuration parameters

    Returns:
        List[Dict[str, Any]]: List of circulation core data
    """
    # Get floor height for floor calculation
    floor_height = building_config.get("floor_height", 4.0)

    # Look for vertical circulation elements
    cores = []

    for room_id, room_data in layout.rooms.items():
        # Check if it's a vertical circulation element
        if room_data["type"] == "vertical_circulation":
            # Get position and dimensions
            position = room_data["position"]
            dimensions = room_data["dimensions"]

            # Calculate the floor
            floor = int(position[2] / floor_height)

            # Extract metadata
            metadata = room_data.get("metadata", {})
            is_core = metadata.get("is_core", False)
            name = metadata.get("name", f"Circulation {room_id}")

            # Add to list of cores
            cores.append(
                {
                    "id": room_id,
                    "position": position,
                    "dimensions": dimensions,
                    "floor": floor,
                    "is_core": is_core,
                    "name": name,
                    "area": dimensions[0] * dimensions[1],
                }
            )

    # Sort by size (descending)
    cores.sort(key=lambda x: x["area"], reverse=True)

    return cores


def main():
    """Main function of the utility script"""
    parser = argparse.ArgumentParser(
        description="Find vertical circulation cores in a hotel layout"
    )
    parser.add_argument(
        "--layout", type=str, required=True, help="Path to layout JSON file"
    )
    parser.add_argument(
        "--building-config",
        type=str,
        default="default",
        help="Building configuration name",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show visualization of the layout"
    )

    args = parser.parse_args()

    # Get building configuration
    building_config = get_building_envelope(args.building_config)

    # Load layout
    layout = load_layout(args.layout)
    if not layout:
        return

    # Count total rooms
    print(f"Layout contains {len(layout.rooms)} rooms")

    # Find all circulation cores
    cores = find_circulation_cores(layout, building_config)

    # Print information about found cores
    print(f"Found {len(cores)} vertical circulation elements:")
    for i, core in enumerate(cores):
        print(f"  Core {i+1}:")
        print(f"    ID: {core['id']}")
        print(f"    Position: {core['position']}")
        print(f"    Dimensions: {core['dimensions']}")
        print(f"    Floor: {core['floor']}")
        print(f"    Area: {core['area']} m²")
        print(f"    Marked as core: {core['is_core']}")
        print(f"    Name: {core['name']}")

    # Use the standard floor generator algorithm to find the main core
    main_core = find_vertical_circulation_core(layout, building_config)

    if main_core:
        print("\nIdentified main vertical circulation core:")
        print(f"  ID: {main_core['id']}")
        print(f"  Position: {main_core['position']}")
        print(f"  Dimensions: {main_core['dimensions']}")
        print(f"  Type: {main_core['type']}")
    else:
        print("\nNo suitable main core found in the layout")

    # Visualize if requested
    if args.visualize and layout:
        try:
            import matplotlib.pyplot as plt
            from hotel_design_ai.visualization.renderer import LayoutRenderer

            # Create renderer
            renderer = LayoutRenderer(layout, building_config=building_config)

            # Render 3D view
            fig, ax = renderer.render_3d(show_labels=True)

            # Highlight all cores in red
            if cores:
                print("Highlighting cores in the visualization...")
                highlight_ids = [core["id"] for core in cores]
                fig, ax = renderer.render_3d(highlight_rooms=highlight_ids)

            plt.show()
        except ImportError:
            print("Could not visualize layout: matplotlib is required")
            print("Install matplotlib with: pip install matplotlib")


if __name__ == "__main__":
    main()
