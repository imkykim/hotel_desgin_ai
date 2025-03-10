#!/usr/bin/env python
"""
Script to generate an initial hotel layout based on the provided program requirements.
This uses the rule-based engine as a starting point before RL optimization.
"""

import os
import sys
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.core.rule_engine import RuleEngine
from hotel_design_ai.models.room import Room, RoomFactory
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import export_to_json, export_to_csv
from hotel_design_ai.config.ENV import BUILDING_ENVELOPE, get_all_rooms


def convert_room_dicts_to_room_objects(room_dicts):
    """Convert room dictionaries from ENV.py to Room objects"""
    rooms = []
    
    for room_dict in room_dicts:
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
            metadata={
                "department": room_dict["department"],
                "id": room_dict["id"]
            },
            id=room_dict["id"]
        )
        
        rooms.append(room)
    
    return rooms


def create_initial_layout():
    """Generate an initial hotel layout using the rule-based engine"""
    print("Hotel Design AI - Initial Layout Generator")
    print("------------------------------------------")
    
    # Get building envelope parameters
    width = BUILDING_ENVELOPE["width"]
    length = BUILDING_ENVELOPE["length"]
    height = BUILDING_ENVELOPE["height"]
    grid_size = BUILDING_ENVELOPE["grid_size"]
    structural_grid = (
        BUILDING_ENVELOPE["structural_grid_x"],
        BUILDING_ENVELOPE["structural_grid_y"]
    )
    
    # Initialize rule engine
    print(f"\nInitializing Rule Engine with building envelope: {width}m x {length}m x {height}m")
    rule_engine = RuleEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid
    )
    
    # Get room requirements from ENV
    room_dicts = get_all_rooms()
    print(f"Loaded {len(room_dicts)} rooms from program requirements")
    
    # Convert to Room objects
    rooms = convert_room_dicts_to_room_objects(room_dicts)
    
    # Display room summary by department
    departments = {}
    for room in rooms:
        dept = room.metadata.get("department", "unknown")
        if dept not in departments:
            departments[dept] = {"count": 0, "area": 0}
        
        departments[dept]["count"] += 1
        departments[dept]["area"] += room.area
    
    print("\nRoom Summary by Department:")
    print("---------------------------")
    for dept, stats in departments.items():
        print(f"{dept}: {stats['count']} rooms, {stats['area']:.1f} m²")
    
    # Generate layout
    print("\nGenerating layout using rule-based engine...")
    start_time = time.time()
    layout = rule_engine.generate_layout(rooms)
    end_time = time.time()
    
    # Print layout statistics
    print(f"Layout generated in {end_time - start_time:.2f} seconds")
    print(f"Total rooms placed: {len(layout.rooms)} / {len(rooms)}")
    space_utilization = layout.calculate_space_utilization() * 100
    print(f"Space utilization: {space_utilization:.1f}%")
    
    # Save outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to various formats
    print(f"\nSaving outputs to {output_dir} directory...")
    json_file = os.path.join(output_dir, f"hotel_layout_{timestamp}.json")
    csv_file = os.path.join(output_dir, f"hotel_layout_{timestamp}.csv")
    
    export_to_json(layout, json_file)
    export_to_csv(layout, csv_file)
    
    # Create visualization
    print("Creating visualizations...")
    renderer = LayoutRenderer(layout)
    renderer.save_renders(
        output_dir=output_dir,
        prefix=f"hotel_layout_{timestamp}",
        include_3d=True,
        include_floor_plans=True,
        num_floors=BUILDING_ENVELOPE["num_floors"] + 1  # +1 for basement
    )
    
    print(f"\nCompleted! Output files saved to '{output_dir}' directory.")
    return layout


def visualize_layout(layout):
    """Visualize the layout interactively"""
    renderer = LayoutRenderer(layout)
    
    # Create 3D visualization
    fig1, ax1 = renderer.render_3d(show_labels=True)
    fig1.suptitle("Hotel Layout - 3D View")
    
    # Create floor plans for each floor including basement
    num_floors = BUILDING_ENVELOPE["num_floors"] + 1  # +1 for basement
    for floor in range(-1, num_floors-1):
        fig, ax = renderer.render_floor_plan(floor=floor)
        floor_name = "Basement" if floor == -1 else f"Floor {floor}"
        fig.suptitle(f"Hotel Layout - {floor_name}")
    
    # Create room type legend
    fig_legend, ax_legend = renderer.create_room_legend()
    fig_legend.suptitle("Room Types")
    
    # Show all figures
    plt.show()


if __name__ == "__main__":
    # Generate the layout
    layout = create_initial_layout()
    
    # Optional: visualize interactively
    if "--visualize" in sys.argv:
        visualize_layout(layout)
