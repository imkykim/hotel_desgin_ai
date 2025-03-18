#!/usr/bin/env python3
"""
Quick script to test floor utilization in a hotel layout.
This script directly creates a simple layout with rooms on every floor
and visualizes it to confirm rendering works.

Place this file in the 'tests' directory of your hotel_design_ai project.
Run it with: python -m tests.test_floors
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Ensure the parent directory is in the path
# This makes imports work correctly when running from the tests directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.visualization.renderer import LayoutRenderer


def create_test_layout():
    """Create a test layout with rooms on every floor"""
    print("Creating test layout with rooms on every floor...")

    # Building dimensions
    width = 60.0
    length = 80.0
    height = 20.0
    floor_height = 4.0

    # Initialize spatial grid
    grid = SpatialGrid(width=width, length=length, height=height, grid_size=1.0)

    # Create rooms on each floor
    room_id = 1

    # Vertical circulation core at the center
    core_width = 8.0
    core_length = 8.0
    core_x = (width - core_width) / 2
    core_y = (length - core_length) / 2

    # Place vertical circulation on all floors
    print("Placing vertical circulation elements...")
    for floor in range(-1, 4):  # -1 to 3
        z = floor * floor_height
        grid.place_room(
            room_id=10000 + floor,
            x=core_x,
            y=core_y,
            z=z,
            width=core_width,
            length=core_length,
            height=floor_height,
            room_type="vertical_circulation",
            metadata={"name": f"Circulation Core (Floor {floor})"},
        )

    # Place rooms on each floor
    print("Placing rooms on each floor...")
    # Floor -1 (Basement)
    grid.place_room(
        room_id=1,
        x=10,
        y=10,
        z=-1 * floor_height,
        width=15,
        length=20,
        height=floor_height,
        room_type="parking",
        metadata={"name": "Basement Parking"},
    )

    # Floor 0 (Ground)
    grid.place_room(
        room_id=2,
        x=10,
        y=40,
        z=0,
        width=20,
        length=30,
        height=floor_height,
        room_type="lobby",
        metadata={"name": "Main Lobby"},
    )

    # Floor 1
    grid.place_room(
        room_id=3,
        x=30,
        y=10,
        z=1 * floor_height,
        width=10,
        length=15,
        height=floor_height,
        room_type="guest_room",
        metadata={"name": "Guest Room (Floor 1)"},
    )

    # Floor 2
    grid.place_room(
        room_id=4,
        x=30,
        y=30,
        z=2 * floor_height,
        width=10,
        length=15,
        height=floor_height,
        room_type="guest_room",
        metadata={"name": "Guest Room (Floor 2)"},
    )

    # Floor 3
    grid.place_room(
        room_id=5,
        x=10,
        y=50,
        z=3 * floor_height,
        width=10,
        length=15,
        height=floor_height,
        room_type="guest_room",
        metadata={"name": "Guest Room (Floor 3)"},
    )

    # Count rooms by floor
    rooms_by_floor = {}
    for room_id, room_data in grid.rooms.items():
        z = room_data["position"][2]
        floor = int(z / floor_height)
        if floor not in rooms_by_floor:
            rooms_by_floor[floor] = 0
        rooms_by_floor[floor] += 1

    # Print summary
    print("Floor utilization in test layout:")
    for floor in sorted(rooms_by_floor.keys()):
        floor_name = "Basement" if floor < 0 else f"Floor {floor}"
        print(f"  {floor_name}: {rooms_by_floor[floor]} rooms")

    return grid


def visualize_layout(grid):
    """Visualize the layout"""
    print("\nVisualizing layout...")

    # Setup building config for renderer
    building_config = {"floor_height": 4.0, "min_floor": -1, "max_floor": 3}

    # Create renderer
    renderer = LayoutRenderer(grid, building_config=building_config)

    # Render 3D view
    fig1, ax1 = renderer.render_3d(show_labels=True)
    fig1.suptitle("Test Layout - 3D View")

    # Render floor plans
    for floor in range(-1, 4):  # -1 to 3
        fig, ax = renderer.render_floor_plan(floor=floor)
        floor_name = "Basement" if floor < 0 else f"Floor {floor}"
        fig.suptitle(f"Test Layout - {floor_name}")

    # Show all figures
    plt.show()


def main():
    """Main function"""
    print("Floor rendering test script")
    print("==========================")

    # Create test layout
    grid = create_test_layout()

    # Visualize layout
    visualize_layout(grid)

    print("\nTest completed!")


if __name__ == "__main__":
    main()
