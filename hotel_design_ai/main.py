# python main.py --mode hybrid --visualize
# python main.py --mode rule  --program-config hotel_requirements_2 --visualize --fixed-rooms data/fix/fixed_rooms.json
"""
Main application for Hotel Design AI.
This integrates all components (rule engine, RL engine, constraints, visualization)
and provides a command-line interface for generating and evaluating hotel layouts.
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hotel_design_ai.core.spatial_grid import SpatialGrid

# from hotel_design_ai.core.grid_rule_engine import RuleEngine

from hotel_design_ai.core.rule_engine import RuleEngine

# from hotel_design_ai.core.grid_rl_engine import RLEngine

from hotel_design_ai.core.rl_engine import RLEngine
from hotel_design_ai.core.constraints import (
    Constraint,
    ConstraintSystem,
    create_default_constraints,
)
from hotel_design_ai.models.room import Room
from hotel_design_ai.models.layout import Layout
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.visualization.export import (
    export_to_json,
    export_to_csv,
    export_to_blender,
    export_to_rhino,
    export_for_three_js,
    export_metrics_to_json,
)

from hotel_design_ai.utils.metrics import LayoutMetrics

# from hotel_design_ai.utils.diagram_metrics import LayoutMetrics

from hotel_design_ai.config.config_loader import (
    get_building_envelope,
    get_program_requirements,
    get_adjacency_requirements,
    get_design_constraints,
    get_rl_parameters,
    create_room_objects_from_program,
    get_all_constraints,
    # Import the directory paths
    DATA_DIR,
    USER_DATA_DIR,
    LAYOUTS_DIR,
    MODELS_DIR,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hotel Design AI - Layout Generator")

    # Basic arguments
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rule", "rl", "hybrid"],
        default="rule",
        help="Generation mode (rule-based, RL, or hybrid)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualizations (requires matplotlib)",
    )

    # Building parameters
    parser.add_argument(
        "--building-config",
        type=str,
        default="default",
        help="Building configuration name (without .json extension)",
    )
    parser.add_argument(
        "--program-config",
        type=str,
        default="default",
        help="Program configuration name (without .json extension)",
    )

    # Export options
    parser.add_argument(
        "--export-formats",
        type=str,
        default="json,csv",
        help="Comma-separated list of export formats (json,csv,revit,threejs)",
    )

    # Advanced options
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of layout iterations to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--fixed-rooms",
        type=str,
        default=None,
        help="JSON file with fixed room positions",
    )
    parser.add_argument(
        "--constraints",
        action="store_true",
        help="Use constraints from data/constraints directory",
    )

    # RL specific options
    parser.add_argument(
        "--rl-model", type=str, default=None, help="Path to saved RL model"
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=20,
        help="Number of training iterations for RL",
    )
    parser.add_argument(
        "--simulate-feedback",
        action="store_true",
        help="Simulate user feedback for RL training",
    )
    parser.add_argument(
        "--standard-floor-zones",
        type=str,
        help="Path to standard floor zones JSON file",
    )
    parser.add_argument(
        "--modified-layout", type=str, help="Path to user-modified layout JSON file"
    )
    parser.add_argument(
        "--user-rating", type=float, help="User rating for the layout (0-10)"
    )
    parser.add_argument(
        "--reference-layout", type=str, help="Path to reference layout JSON file"
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Generate complete hotel with podium and standard floors",
    )

    return parser.parse_args()


# Add this code to main.py around line 150-170, in the load_fixed_rooms function
# or right before where fixed_positions is processed


def load_fixed_rooms(filepath: str) -> Dict[int, Any]:
    """
    Enhanced fixed room loader with strict position enforcement.
    """
    print(f"\nLOADING FIXED ROOMS WITH STRICT ENFORCEMENT from: {filepath}")

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Check if this is the enhanced format with identifiers
        if isinstance(data, dict) and "fixed_rooms" in data:
            print(f"Found {len(data['fixed_rooms'])} fixed room definitions")

            # Return the raw data for later exact matching
            return data

        # Original format with room IDs
        fixed_positions = {}
        for room_id, position in data.items():
            fixed_positions[int(room_id)] = tuple(position)
            print(f"Fixed position for room {room_id}: {position}")

        return fixed_positions

    except Exception as e:
        print(f"Error loading fixed rooms file: {e}")
        return {}


def match_fixed_rooms_to_actual(fixed_data, rooms, debug=True):
    """
    Completely rewritten matching function with strict position enforcement.
    """
    if debug:
        print(f"\nDEBUG: STRICT fixed position matcher")

    # If already in simple format (direct ID to position mapping), convert IDs to int
    if isinstance(fixed_data, dict) and not "fixed_rooms" in fixed_data:
        result = {}
        for room_id_str, position in fixed_data.items():
            room_id = int(room_id_str) if isinstance(room_id_str, str) else room_id_str
            result[room_id] = tuple(position)
            if debug:
                print(f"DEBUG: Direct mapping for room ID {room_id}: {position}")
        return result

    # Enhanced format with identifiers
    if not isinstance(fixed_data, dict) or not "fixed_rooms" in fixed_data:
        if debug:
            print(f"DEBUG: Unrecognized fixed data format: {type(fixed_data)}")
        return {}

    fixed_rooms = fixed_data["fixed_rooms"]
    result = {}
    matched_count = 0

    # Create direct lookup maps
    id_map = {room.id: room for room in rooms}
    type_map = {}
    for room in rooms:
        room_type = getattr(room, "room_type", "unknown")
        if room_type not in type_map:
            type_map[room_type] = []
        type_map[room_type].append(room)

    # First pass: exact identifier matches only
    for fixed_def in fixed_rooms:
        if "identifier" not in fixed_def or "position" not in fixed_def:
            continue

        identifier = fixed_def["identifier"]
        position = tuple(fixed_def["position"])
        id_type = identifier.get("type", "")

        # Most specific match: department + name
        if id_type == "department_with_name":
            dept = identifier.get("department", "")
            name = identifier.get("name", "")

            # Special case for circulation cores
            if dept == "circulation" and "core" in name:
                for room in rooms:
                    meta = getattr(room, "metadata", {}) or {}
                    r_type = getattr(room, "room_type", "")

                    # Match vertical circulation with core in name
                    if r_type == "vertical_circulation" and room.id not in result:
                        result[room.id] = position
                        if debug:
                            print(
                                f"DEBUG: STRICT CORE MATCH - room id {room.id} -> {position}"
                            )
                        matched_count += 1
                        break

        # Room type + name match
        elif id_type == "room_type_with_name":
            room_type = identifier.get("room_type", "")
            name = identifier.get("name", "")

            # Special case for lobby/reception
            if room_type == "lobby" and name == "reception":
                for room in rooms:
                    r_type = getattr(room, "room_type", "")
                    if r_type == "lobby" and room.id not in result:
                        result[room.id] = position
                        if debug:
                            print(
                                f"DEBUG: STRICT LOBBY MATCH - room id {room.id} -> {position}"
                            )
                        matched_count += 1
                        break

    if debug:
        print(f"DEBUG: Matched {matched_count} fixed positions with strict rules")

    return result


# def match_fixed_rooms_to_actual(
#     fixed_data: Any, rooms: List[Room]
# ) -> Dict[int, Tuple[float, float, float]]:
#     """
#     Match fixed rooms to actual rooms using various identifier types.

#     Args:
#         fixed_data: Either a dictionary mapping room IDs to positions,
#                     or a list of fixed room configurations with identifiers
#         rooms: List of Room objects

#     Returns:
#         Dictionary mapping room IDs to positions
#     """
#     # If already in simple format (direct ID to position mapping), return as is
#     if isinstance(fixed_data, dict):
#         return fixed_data

#     result = {}
#     matched_room_ids = set()
#     matched_count = 0

#     for fixed_room in fixed_data:
#         try:
#             identifier = fixed_room["identifier"]
#             position = tuple(fixed_room["position"])

#             # Find matching room based on identifier type
#             matching_room = None

#             if identifier["type"] == "department_with_name":
#                 # Match by department and name
#                 department = identifier["department"]
#                 name = identifier["name"]

#                 for room in rooms:
#                     if room.id in matched_room_ids:
#                         continue

#                     # Check department (either in metadata or direct attribute)
#                     has_dept = (
#                         hasattr(room, "metadata")
#                         and room.metadata
#                         and room.metadata.get("department") == department
#                     ) or (hasattr(room, "department") and room.department == department)

#                     # Check name matches directly or in metadata
#                     has_name = room.name == name or (
#                         hasattr(room, "metadata")
#                         and room.metadata
#                         and room.metadata.get("subspace_name") == name
#                     )

#                     if has_dept and has_name:
#                         matching_room = room
#                         break

#             elif identifier["type"] == "room_type_with_name":
#                 # Match by room type and name
#                 room_type = identifier["room_type"]
#                 name = identifier["name"]

#                 for room in rooms:
#                     if room.id in matched_room_ids:
#                         continue

#                     if room.room_type == room_type and room.name == name:
#                         matching_room = room
#                         break

#             elif identifier["type"] == "room_type" and "value" in identifier:
#                 # Match by room type only
#                 room_type = identifier["value"]

#                 for room in rooms:
#                     if room.id in matched_room_ids and room.room_type == room_type:
#                         matching_room = room
#                         break

#             # If found a match, add to results
#             if matching_room:
#                 result[matching_room.id] = position
#                 matched_room_ids.add(matching_room.id)
#                 matched_count += 1
#                 print(
#                     f"  ✓ Matched room: id={matching_room.id}, name={matching_room.name}"
#                 )
#             else:
#                 # Simplified error message
#                 print(f"  ✗ No match found for {identifier['type']} identifier")

#         except (KeyError, TypeError) as e:
#             print(f"  ✗ Error with fixed room definition: {e}")

#     print(
#         f"Successfully fixed positions for {matched_count} out of {len(fixed_data)} rooms"
#     )
#     return result


def convert_room_dicts_to_room_objects(room_dicts: List[Dict[str, Any]]) -> List[Room]:
    """Convert room dictionaries from config_loader to Room objects"""
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


def generate_rule_based_layout(
    args, rooms: List[Room], fixed_positions: Optional[Dict[int, Any]] = None
):
    """Generate a layout using the rule-based engine"""
    print("\nGenerating layout using rule-based engine...")

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
    min_floor = building_config.get("min_floor", -1)
    max_floor = building_config.get("max_floor", 3)
    floor_height = building_config.get("floor_height", 5.0)

    # Create a spatial grid that properly supports basements
    spatial_grid = SpatialGrid(
        width=width,
        length=length,
        height=height,
        grid_size=grid_size,
        min_floor=min_floor,
        floor_height=floor_height,
    )

    # Initialize rule engine
    rule_engine = RuleEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid,
        building_config=building_config,  # Pass the complete building config
    )

    # Replace the spatial grid to ensure basement support
    rule_engine.spatial_grid = spatial_grid

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

    print(f"Layout generated in {end_time - start_time:.2f} seconds")

    return layout


# Update similar patterns in generate_rl_layout and generate_hybrid_layout functions
def generate_rl_layout(
    args, rooms: List[Room], fixed_positions: Optional[Dict[int, Any]] = None
):
    """Generate a layout using the RL engine"""
    print("\nGenerating layout using RL engine...")

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
    min_floor = building_config.get("min_floor", -1)
    max_floor = building_config.get("max_floor", 3)
    floor_height = building_config.get("floor_height", 5.0)

    # Get RL parameters
    rl_params = get_rl_parameters()

    # Create a spatial grid that properly supports basements
    spatial_grid = SpatialGrid(
        width=width,
        length=length,
        height=height,
        grid_size=grid_size,
        min_floor=min_floor,
        floor_height=floor_height,
    )

    # Initialize RL engine
    rl_engine = RLEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid,
        exploration_rate=rl_params["training"]["exploration_rate"],
        learning_rate=rl_params["training"]["learning_rate"],
        discount_factor=rl_params["training"]["discount_factor"],
        building_config=building_config,  # Pass the complete building config
    )

    # Replace the spatial grid to ensure basement support
    rl_engine.spatial_grid = spatial_grid

    # Load pre-trained model if specified
    if args.rl_model and os.path.exists(args.rl_model):
        print(f"Loading RL model from {args.rl_model}")
        rl_engine.load_model(args.rl_model)

    # Update fixed elements
    if fixed_positions:
        rl_engine.update_fixed_elements(fixed_positions)

    # Train or simulate feedback if requested
    if args.train_iterations > 0 and args.simulate_feedback:
        print(f"Simulating user feedback for {args.train_iterations} iterations...")

        for i in range(args.train_iterations):
            # Generate a layout
            layout = rl_engine.generate_layout(rooms)

            # Calculate reward based on architectural principles
            reward = rl_engine.calculate_reward(layout)

            # Convert to user rating (0-10 scale)
            user_rating = min(10, max(0, reward * 10))

            print(
                f"  Iteration {i+1}: Calculated reward={reward:.2f}, Simulated rating={user_rating:.1f}"
            )

            # Update model with feedback
            rl_engine.update_model(user_rating)

    # Generate final layout
    start_time = time.time()
    layout = rl_engine.generate_layout(rooms)
    end_time = time.time()

    print(f"Layout generated in {end_time - start_time:.2f} seconds")

    # Save trained model if specified
    if args.rl_model:
        model_dir = os.path.dirname(args.rl_model)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        print(f"Saving RL model to {args.rl_model}")
        rl_engine.save_model(args.rl_model)

    return layout


def generate_hybrid_layout(
    args, rooms: List[Room], fixed_positions: Optional[Dict[int, Any]] = None
):
    """Generate a layout using a hybrid approach (rule-based + RL refinement)"""
    print("\nGenerating layout using hybrid approach...")

    # First, generate a layout using rule-based engine
    print("Step 1: Generating initial layout with rule-based engine...")
    rule_layout = generate_rule_based_layout(args, rooms, fixed_positions)

    # Extract important elements to fix for RL stage
    print("Step 2: Identifying key elements to fix for RL refinement...")
    key_room_types = ["entrance", "lobby", "vertical_circulation", "kitchen"]
    rl_fixed_positions = {}

    # Start with user-provided fixed positions
    if fixed_positions:
        rl_fixed_positions.update(fixed_positions)

    # Add key elements from rule-based layout
    for room_id, room_data in rule_layout.rooms.items():
        room_type = room_data["type"]
        if room_type in key_room_types and room_id not in rl_fixed_positions:
            rl_fixed_positions[room_id] = room_data["position"]

    # Create rooms list with positions for the RL engine
    rl_rooms = []
    for room in rooms:
        room_copy = Room.from_dict(room.to_dict())
        if room.id in rule_layout.rooms:
            room_copy.position = rule_layout.rooms[room.id]["position"]
        rl_rooms.append(room_copy)

    # Now refine using RL engine with fixed key elements
    print("Step 3: Refining layout with RL engine...")
    rl_layout = generate_rl_layout(args, rl_rooms, rl_fixed_positions)

    return rl_layout


def evaluate_layout(layout: SpatialGrid, rooms: List[Room]):
    """Evaluate a layout using various metrics"""
    print("\nEvaluating layout...")

    # Create Layout model wrapper
    layout_model = Layout(layout)

    # Get building parameters for metrics
    building_config = get_building_envelope()

    # Create metrics calculator
    metrics = LayoutMetrics(layout, building_config=building_config)

    # Calculate metrics
    space_utilization = metrics.space_utilization() * 100
    print(f"Space utilization: {space_utilization:.1f}%")

    # Create a simple room_id to department mapping for clustering metric
    room_departments = {
        room.id: room.metadata.get("department", "unknown") for room in rooms
    }

    # Get adjacency preferences from configuration
    adjacency_requirements = get_adjacency_requirements()
    adjacency_preferences = {}

    # Convert to the format expected by metrics
    for room_type1, room_type2 in adjacency_requirements.get(
        "required_adjacencies", []
    ):
        if room_type1 not in adjacency_preferences:
            adjacency_preferences[room_type1] = []
        adjacency_preferences[room_type1].append(room_type2)

    # Get structural grid for metrics
    structural_grid = (
        building_config["structural_grid_x"],
        building_config["structural_grid_y"],
    )

    # Get RL weights for metrics
    rl_params = get_rl_parameters()
    weights = rl_params.get("weights", {})

    # Evaluate all metrics
    all_metrics = metrics.evaluate_all(
        adjacency_preferences=adjacency_preferences,
        room_departments=room_departments,
        structural_grid=structural_grid,
        weights=weights,
    )

    # Display key metrics
    print(f"Overall score: {all_metrics['overall_score'] * 100:.1f}%")

    if "adjacency_satisfaction" in all_metrics:
        print(
            f"Adjacency satisfaction: {all_metrics['adjacency_satisfaction'] * 100:.1f}%"
        )

    if "natural_light_access" in all_metrics:
        print(f"Natural light access: {all_metrics['natural_light_access'] * 100:.1f}%")

    if "department_clustering" in all_metrics:
        print(
            f"Department clustering: {all_metrics['department_clustering'] * 100:.1f}%"
        )

    if "structural_alignment" in all_metrics:
        print(f"Structural alignment: {all_metrics['structural_alignment'] * 100:.1f}%")

    # Show areas by room type
    areas_by_type = layout_model.get_areas_by_type()
    print("\nAreas by room type (in final layout):")
    for room_type, area in sorted(areas_by_type.items()):
        print(f"  {room_type}: {area:.1f} m²")

    # Group areas by department in final layout
    print("\nAreas by department (in final layout):")
    areas_by_department = {}
    total_area = 0

    # Create a mapping of room_id to department
    room_id_to_dept = {
        room.id: room.metadata.get("department", "unknown") for room in rooms
    }

    # Calculate areas by department using actual layout
    for room_id, room_data in layout.rooms.items():
        dept = room_id_to_dept.get(room_id, "unknown")

        if dept not in areas_by_department:
            areas_by_department[dept] = 0

        width, length, _ = room_data["dimensions"]
        area = width * length
        areas_by_department[dept] += area
        total_area += area

    # Print department summaries
    for dept, area in sorted(areas_by_department.items()):
        print(f"  {dept}: {area:.1f} m²")

    # Print total area
    print(f"\nTotal area placed in layout: {total_area:.1f} m²")

    # Compare with programmed areas from the original requirements
    print("\nProgrammed areas from requirements (excluding logistics reserves):")
    program_areas = {
        "public": 680,  # reception 160 + retail 400 + service_areas 120
        "dining": 2250,  # kitchen 850 + restaurants 1300 + other 100
        "meeting": 1610,  # ballroom 500 + hall 650 + meeting_rooms 250 + vip 100 + other 110
        "recreational": 1400,  # pool 700 + gym 150 + beauty 80 + billiards 120 + ktv 350
        "administrative": 2040,  # offices 540 + staff 800 + misc 700
        "engineering": 2800,  # maintenance 315 + equipment 2485
        "parking": 3900,  # underground 3900
    }
    total_programmed = sum(program_areas.values())
    for dept, area in sorted(program_areas.items()):
        print(f"  {dept}: {area:.1f} m²")
    print(f"  Total programmed: {total_programmed:.1f} m²")

    # Return metrics for possible export
    return all_metrics


def visualize_layout(layout: SpatialGrid, args):
    """Visualize the layout if requested"""
    if not args.visualize:
        # Make sure to close any figures that might be created
        import matplotlib.pyplot as plt

        plt.close("all")
        return

    print("\nCreating visualizations...")

    # Get building parameters
    building_config = get_building_envelope(args.building_config)

    # Create renderer
    renderer = LayoutRenderer(layout, building_config=building_config)

    # Print rooms by floor for debugging
    rooms_by_floor = {}
    for room_id, room_data in layout.rooms.items():
        z = room_data["position"][2]
        floor = int(z / building_config["floor_height"])
        if floor not in rooms_by_floor:
            rooms_by_floor[floor] = []
        rooms_by_floor[floor].append(room_id)

    print("\nRoom distribution by floor:")
    for floor, rooms in sorted(rooms_by_floor.items()):
        print(f"Floor {floor}: {len(rooms)} rooms")

    # Get standard floor info
    std_floor_config = building_config.get("standard_floor", {})
    start_floor = std_floor_config.get("start_floor", 5)
    end_floor = std_floor_config.get("end_floor", 20)

    # Get all floors that have rooms
    occupied_floors = sorted(
        [floor for floor, rooms in rooms_by_floor.items() if len(rooms) > 0]
    )

    # Split into podium and standard floors
    podium_floors = [f for f in occupied_floors if f < start_floor]
    standard_floors = [f for f in occupied_floors if start_floor <= f <= end_floor]

    # For standard floors, only keep one representative floor if sample_standard=True
    if standard_floors and args.complete:
        # Keep only the first standard floor
        sample_standard_floor = min(standard_floors)
        standard_floors = [sample_standard_floor]

    # Combine and sort for final rendering
    floors_to_render = sorted(podium_floors + standard_floors)

    print(f"\nRendering floor plans for floors: {floors_to_render}")

    try:
        # Create 3D visualization
        fig1, ax1 = renderer.render_3d(show_labels=True)
        fig1.suptitle("Hotel Layout - 3D View")

        # Create floor plans ONLY for selected floors
        for floor in floors_to_render:
            fig, ax = renderer.render_floor_plan(floor=floor)

            # Create appropriate floor name
            if floor < 0:
                floor_name = f"Basement {abs(floor)}"
            elif start_floor <= floor <= end_floor:
                floor_name = f"Floor {floor} (Standard Floor)"
                # If this is a sample, indicate it represents all standard floors
                if len(standard_floors) == 1 and floor == standard_floors[0]:
                    floor_name += " (Representative)"
            else:
                floor_name = f"Floor {floor}"

            fig.suptitle(f"Hotel Layout - {floor_name}")

        # Show all figures
        import matplotlib.pyplot as plt

        plt.show()

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback

        traceback.print_exc()
        print("Make sure matplotlib is installed and properly configured.")


def save_outputs(layout: SpatialGrid, metrics: Dict[str, Any], args):
    """Save layout to various output formats"""
    print("\nSaving outputs...")

    # Create output directory within user_data/layouts if needed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subfolder = os.path.join(USER_DATA_DIR, "layouts", timestamp)
    os.makedirs(output_subfolder, exist_ok=True)

    # # Create output directory if needed
    # os.makedirs(args.output, exist_ok=True)

    # # Create timestamped subfolder
    # output_subfolder = os.path.join(args.output, timestamp)
    # os.makedirs(output_subfolder, exist_ok=True)

    prefix = f"hotel_layout"

    # Get building parameters
    building_config = get_building_envelope(args.building_config)

    # Determine export formats
    # Determine export formats
    export_formats = [f.strip().lower() for f in args.export_formats.split(",")]

    # Export to requested formats
    for export_format in export_formats:
        if export_format == "json":
            json_file = os.path.join(output_subfolder, f"{prefix}.json")
            export_to_json(layout, json_file)
            print(f"  Saved JSON to {json_file}")

        elif export_format == "csv":
            csv_file = os.path.join(output_subfolder, f"{prefix}.csv")
            export_to_csv(layout, csv_file)
            print(f"  Saved CSV to {csv_file}")

        elif export_format == "threejs":
            threejs_file = os.path.join(output_subfolder, f"{prefix}_threejs.json")
            export_for_three_js(layout, threejs_file)
            print(f"  Saved Three.js JSON to {threejs_file}")

        elif export_format == "blender":
            blender_file = os.path.join(output_subfolder, f"{prefix}_blender.py")
            export_to_blender(layout, blender_file)
            print(f"  Saved Blender Python script to {blender_file}")

        elif export_format == "rhino":
            rhino_file = os.path.join(output_subfolder, f"{prefix}_rhino.py")
            export_to_rhino(layout, rhino_file)
            print(f"  Saved Rhino Python script to {rhino_file}")

    metrics_file = os.path.join(output_subfolder, f"{prefix}_metrics.json")
    export_metrics_to_json(metrics, metrics_file)
    print(f"  Saved metrics to {metrics_file}")

    # Create visualizations
    renderer = LayoutRenderer(layout, building_config=building_config)

    try:
        # Use min_floor and max_floor from building_config
        min_floor = building_config.get("min_floor", -1)
        max_floor = building_config.get(
            "max_floor", building_config.get("num_floors", 4)
        )
        num_floors = max_floor - min_floor + 1

        # Save renders to output directory
        print("  Saving visualizations...")
        renderer.save_renders(
            output_dir=output_subfolder,
            prefix=prefix,
            include_3d=True,
            include_floor_plans=True,
            num_floors=num_floors,
            min_floor=min_floor,
            sample_standard=True,
        )

        # Close any open matplotlib figures to prevent hanging
        import matplotlib.pyplot as plt

        plt.close("all")

    except Exception as e:
        print(f"  Error saving visualizations: {e}")

    print(f"\nOutputs saved to: {output_subfolder}")


def generate_complete_hotel_layout(args, fixed_positions=None):
    """
    Generate a complete hotel layout with both podium and standard floor sections.
    """
    print("\nGenerating complete hotel layout...")

    from hotel_design_ai.config.config_loader_grid import (
        create_room_objects_for_section,
    )
    from hotel_design_ai.config.building_config_compatibility import (
        tag_rooms_with_section,
    )

    # Get building configuration
    building_config = get_building_envelope(args.building_config)

    # Step 1: Generate podium section
    print("Step 1: Generating podium (裙房) section...")

    # Filter rooms for podium section
    podium_room_dicts = create_room_objects_for_section(
        args.program_config, building_config, section="podium"
    )

    # Convert to Room objects
    podium_rooms = convert_room_dicts_to_room_objects(podium_room_dicts)

    # Tag rooms with section info
    podium_rooms = tag_rooms_with_section(podium_rooms, building_config)

    # FIXED: Re-match fixed positions to podium rooms if needed
    if fixed_positions:
        # If using enhanced format, we need to rematch based on new room objects
        if isinstance(fixed_positions, list):
            fixed_pos_for_podium = match_fixed_rooms_to_actual(
                fixed_positions, podium_rooms
            )
        else:
            # Direct mapping might not work with new IDs, but try anyway
            fixed_pos_for_podium = fixed_positions

        # Generate podium layout with fixed positions
        podium_layout = generate_rule_based_layout(
            args, podium_rooms, fixed_pos_for_podium
        )
    else:
        # Generate podium layout without fixed positions
        podium_layout = generate_rule_based_layout(args, podium_rooms)

    # Rest of the function remains the same...
    # Step 2: Generate standard floor section
    print("\nStep 2: Generating standard floor (tower) section...")

    # Print debug info before standard floor generation
    print(f"Layout before standard floors: {len(podium_layout.rooms)} rooms")

    # Determine which floors need standard floor layouts
    std_floor_config = building_config.get("standard_floor", {})
    start_floor = std_floor_config.get("start_floor", 5)
    end_floor = std_floor_config.get("end_floor", 20)

    print(f"Standard floor range: {start_floor} to {end_floor}")

    # Create a spatial grid that will hold the complete layout
    width = building_config["width"]
    length = building_config["length"]
    height = building_config["height"]
    grid_size = building_config["grid_size"]
    min_floor = building_config.get("min_floor", -2)
    floor_height = building_config["floor_height"]

    # Instead of creating a new grid, use the podium_layout as our base
    complete_layout = podium_layout

    # IMPORTANT: Make sure we're using the correct function to generate standard floors
    # Import the standard floor generator
    from hotel_design_ai.core.standard_floor_generator import (
        generate_all_standard_floors,
    )

    # Generate standard floors with explicit parameters
    print("Generating standard floors with explicit parameters...")
    standard_layout, standard_rooms = generate_all_standard_floors(
        building_config=building_config,
        spatial_grid=complete_layout,
        target_room_count=380,
    )

    # Print debug info after standard floor generation
    print(f"Generated {len(standard_rooms)} rooms on standard floors")
    print(f"Total rooms after standard floors: {len(complete_layout.rooms)}")

    # Tag standard floor rooms
    standard_rooms = tag_rooms_with_section(standard_rooms, building_config)

    # Combine all rooms
    all_rooms = podium_rooms + standard_rooms

    print(f"\nComplete layout generated with {len(complete_layout.rooms)} rooms:")
    print(f"  - Podium section: {len(podium_layout.rooms)} rooms")
    print(f"  - Standard floor section: {len(standard_rooms)} rooms")

    return complete_layout, all_rooms


def main():
    """Main entry point for the application"""
    print("Hotel Design AI - Layout Generator")
    print("=================================")

    # Parse command line arguments
    args = parse_arguments()

    print(f"Using building config: {args.building_config}")
    print(f"Using program config: {args.program_config}")

    # Set random seed if specified
    if args.seed is not None:
        import random
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- Standard floor zones loading ---
    standard_floor_zones = None
    fixed_positions = None
    # --- Fixed rooms loading ---
    if args.fixed_rooms:
        print(f"Loading fixed rooms from: {args.fixed_rooms}")
        fixed_data = load_fixed_rooms(args.fixed_rooms)
        # fixed_data can be a dict (id: pos) or a list (enhanced format)
        # Always match to actual Room objects (after rooms are created)
        # So defer matching until after rooms are created
    else:
        fixed_data = None

    # Create rooms from program requirements with the specified config
    from hotel_design_ai.config.config_loader import create_room_objects_from_program

    # Get room dictionaries from program config
    room_dicts = create_room_objects_from_program(args.program_config)

    # Convert to Room objects
    rooms = convert_room_dicts_to_room_objects(room_dicts)

    # --- Match fixed rooms to actual Room objects after rooms are created ---
    if args.fixed_rooms and fixed_data is not None:
        if isinstance(fixed_data, list):
            print("Matching enhanced fixed_rooms identifiers to actual rooms...")
            fixed_positions = match_fixed_rooms_to_actual(fixed_data, rooms)
        else:
            fixed_positions = fixed_data

        # Apply fixed positions to Room objects (set .position)
        for room in rooms:
            if fixed_positions and room.id in fixed_positions:
                room.position = fixed_positions[room.id]

    # --- Main layout generation ---
    building_config = get_building_envelope(args.building_config)

    # Generate podium/rule-based layout first
    if args.complete:
        print("\nGenerating complete hotel with podium and standard floors...")
        layout, all_rooms = generate_complete_hotel_layout(args, fixed_positions)
        rooms = all_rooms
    else:
        # Generate layout based on selected mode
        if args.mode == "rule":
            layout = generate_rule_based_layout(args, rooms, fixed_positions)
        elif args.mode == "rl":
            layout = generate_rl_layout(args, rooms, fixed_positions)
        elif args.mode == "hybrid":
            layout = generate_hybrid_layout(args, rooms, fixed_positions)
        else:
            print(f"Error: Unknown mode '{args.mode}'")
            return

    # --- Standard floor generation if needed ---
    # Only add standard floors if not already done in generate_complete_hotel_layout
    if not args.complete:
        # Check if standard floor config or zones exist
        std_floor_config = building_config.get("standard_floor")
        if (args.standard_floor_zones and standard_floor_zones) or std_floor_config:
            print("\nGenerating standard floors (tower section)...")
            from hotel_design_ai.core.standard_floor_generator import (
                generate_all_standard_floors,
            )

            # If standard_floor_zones provided, update config accordingly
            if standard_floor_zones:
                # Use the first zone as the standard floor boundary
                zone = standard_floor_zones["floor_zones"][0]
                std_floor_params = {
                    "start_floor": standard_floor_zones.get("start_floor", 2),
                    "end_floor": standard_floor_zones.get("end_floor", 20),
                    "width": zone["width"],
                    "length": zone["height"],
                    "position_x": zone["x"],
                    "position_y": zone["y"],
                    "corridor_width": (
                        std_floor_config.get("corridor_width", 4.0)
                        if std_floor_config
                        else 4.0
                    ),
                    "room_depth": (
                        std_floor_config.get("room_depth", 8.0)
                        if std_floor_config
                        else 8.0
                    ),
                }
                building_config["standard_floor"] = std_floor_params

            # Generate standard floors and merge into layout
            layout, standard_rooms = generate_all_standard_floors(
                building_config=building_config,
                spatial_grid=layout,
                target_room_count=building_config.get("target_room_count", 100),
            )
            # Optionally, extend rooms list if you want metrics to include standard rooms
            rooms.extend(standard_rooms)
            print(f"Added {len(standard_rooms)} standard floor rooms.")

    # Evaluate layout
    metrics = evaluate_layout(layout, rooms)

    # Visualize layout if requested
    visualize_layout(layout, args)

    # Save outputs
    save_outputs(layout, metrics, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
