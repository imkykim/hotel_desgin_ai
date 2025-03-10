#!/usr/bin/env python
"""
Sample application to demonstrate the Hotel Design AI system.
This script creates a hotel layout and visualizes it.
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.core.rule_engine import RuleEngine
from hotel_design_ai.core.rl_engine import RLEngine
from hotel_design_ai.models.room import Room, RoomFactory
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import export_to_json, export_to_csv


def create_sample_hotel():
    """Create a sample hotel layout using the rule engine"""
    print("Creating sample hotel layout...")
    
    # Define bounding box (in meters)
    width = 50.0
    length = 60.0
    height = 40.0
    
    # Initialize rule engine
    rule_engine = RuleEngine(
        bounding_box=(width, length, height),
        grid_size=1.0,
        structural_grid=(8.0, 8.0)
    )
    
    # Create rooms
    rooms = [
        # Public areas
        RoomFactory.create_entrance(name="Main Entrance"),
        RoomFactory.create_lobby(name="Main Lobby"),
        RoomFactory.create_restaurant(name="Restaurant"),
        RoomFactory.create_vertical_circulation(name="Main Circulation Core"),
        
        # Meeting facilities
        RoomFactory.create_meeting_room(name="Meeting Room A"),
        RoomFactory.create_meeting_room(name="Meeting Room B"),
        RoomFactory.create_meeting_room(width=15.0, length=20.0, name="Ballroom"),
        
        # Back of house
        Room(width=8.0, length=10.0, height=4.0, room_type="kitchen", name="Kitchen"),
        Room(width=8.0, length=6.0, height=3.5, room_type="back_of_house", name="Storage"),
        Room(width=10.0, length=8.0, height=3.5, room_type="back_of_house", name="Staff Area"),
        Room(width=8.0, length=8.0, height=3.5, room_type="service_area", name="Mechanical Room"),
    ]
    
    # Add guest rooms - 10 per floor, 3 floors
    for floor in range(1, 4):  # Floors 1-3
        for i in range(1, 11):  # 10 rooms per floor
            rooms.append(
                RoomFactory.create_guest_room(
                    name=f"Room {floor}0{i}",
                    floor=floor
                )
            )
    
    # Generate layout using rule engine
    layout = rule_engine.generate_layout(rooms)
    
    print(f"Created layout with {len(layout.rooms)} rooms")
    space_utilization = layout.calculate_space_utilization() * 100
    print(f"Space utilization: {space_utilization:.1f}%")
    
    return layout


def demonstrate_rl_engine(layout):
    """Demonstrate RL engine by simulating user feedback"""
    print("\nDemonstrating RL engine with simulated user feedback...")
    
    # Extract bounding box from existing layout
    width = layout.width
    length = layout.length
    height = layout.height
    
    # Initialize RL engine
    rl_engine = RLEngine(
        bounding_box=(width, length, height),
        grid_size=1.0,
        structural_grid=(8.0, 8.0)
    )
    
    # Extract rooms from existing layout
    rooms = []
    for room_id, room_data in layout.rooms.items():
        room_type = room_data['type']
        w, l, h = room_data['dimensions']
        name = room_data.get('name', f"Room {room_id}")
        
        room = Room(
            width=w,
            length=l,
            height=h,
            room_type=room_type,
            name=name,
            id=room_id
        )
        rooms.append(room)
    
    # Fix position of some key rooms (simulating user interaction)
    fixed_positions = {}
    
    # Find entrance, lobby, and vertical circulation
    entrance_id = None
    lobby_id = None
    circulation_id = None
    
    for room_id, room_data in layout.rooms.items():
        if room_data['type'] == 'entrance':
            entrance_id = room_id
        elif room_data['type'] == 'lobby':
            lobby_id = room_id
        elif room_data['type'] == 'vertical_circulation':
            circulation_id = room_id
    
    # Fix positions of these key rooms
    if entrance_id and entrance_id in layout.rooms:
        fixed_positions[entrance_id] = layout.rooms[entrance_id]['position']
    
    if lobby_id and lobby_id in layout.rooms:
        fixed_positions[lobby_id] = layout.rooms[lobby_id]['position']
    
    if circulation_id and circulation_id in layout.rooms:
        fixed_positions[circulation_id] = layout.rooms[circulation_id]['position']
    
    # Update RL engine with fixed positions
    rl_engine.update_fixed_elements(fixed_positions)
    
    # Generate initial layout
    print("Generating initial layout with RL engine...")
    rl_layout = rl_engine.generate_layout(rooms)
    
    # Simulate user feedback and iteration
    print("Simulating user feedback and layout iterations...")
    for i in range(5):
        # Calculate reward based on architectural principles
        reward = rl_engine.calculate_reward(rl_layout)
        
        # Simulate user rating (0-10 scale), adding some variance
        user_rating = min(10, max(0, reward * 10 + np.random.normal(0, 1)))
        
        print(f"Iteration {i+1}: Calculated reward = {reward:.2f}, User rating = {user_rating:.1f}")
        
        # Update model with feedback
        rl_engine.update_model(user_rating)
        
        # Generate new layout
        rl_layout = rl_engine.generate_layout(rooms)
    
    print("\nRL engine training complete")
    return rl_layout


def main():
    parser = argparse.ArgumentParser(description='Generate and visualize a hotel layout')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for renderings and exports')
    parser.add_argument('--no-rl', action='store_true',
                        help='Skip RL engine demonstration')
    parser.add_argument('--export-formats', type=str, default='json,csv',
                        help='Comma-separated list of export formats (json, csv)')
                        
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create initial layout with rule engine
    layout = create_sample_hotel()
    
    # Export initial layout
    export_formats = args.export_formats.split(',')
    for fmt in export_formats:
        if fmt.strip().lower() == 'json':
            export_to_json(layout, os.path.join(args.output, f'hotel_layout_{timestamp}.json'))
        elif fmt.strip().lower() == 'csv':
            export_to_csv(layout, os.path.join(args.output, f'hotel_layout_{timestamp}.csv'))
    
    # Create renderer
    renderer = LayoutRenderer(layout)
    
    # Render and save initial layout
    print("\nRendering initial layout...")
    renderer.save_renders(
        output_dir=args.output,
        prefix=f'initial_layout_{timestamp}'
    )
    
    # Demonstrate RL engine unless disabled
    if not args.no_rl:
        rl_layout = demonstrate_rl_engine(layout)
        
        # Create renderer for RL layout
        rl_renderer = LayoutRenderer(rl_layout)
        
        # Render and save RL layout
        print("\nRendering RL-generated layout...")
        rl_renderer.save_renders(
            output_dir=args.output,
            prefix=f'rl_layout_{timestamp}'
        )
        
        # Export RL layout
        for fmt in export_formats:
            if fmt.strip().lower() == 'json':
                export_to_json(rl_layout, os.path.join(args.output, f'rl_hotel_layout_{timestamp}.json'))
            elif fmt.strip().lower() == 'csv':
                export_to_csv(rl_layout, os.path.join(args.output, f'rl_hotel_layout_{timestamp}.csv'))
    
    print(f"\nCompleted! Output files saved to '{args.output}' directory.")


if __name__ == '__main__':
    main()
