# python main.py --mode hybrid --visualize
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
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the parent directory to path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.core.rule_engine import RuleEngine
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
    export_to_revit,
    export_for_three_js,
)
from hotel_design_ai.utils.metrics import LayoutMetrics

# Import the config loader instead of ENV.py
from hotel_design_ai.config.config_loader import (
    get_building_envelope,
    get_program_requirements,
    get_adjacency_requirements,
    get_design_constraints,
    get_rl_parameters,
    create_room_objects_from_program,
    get_all_constraints,
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
        default="hotel_requirements",  # Changed default from "default" to "hotel_requirements"
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
        default=10,
        help="Number of training iterations for RL",
    )
    parser.add_argument(
        "--simulate-feedback",
        action="store_true",
        help="Simulate user feedback for RL training",
    )

    return parser.parse_args()


def load_fixed_rooms(filepath: str) -> Dict[int, Any]:
    """Load fixed room positions from a JSON file"""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Convert positions from lists to tuples
    fixed_positions = {}
    for room_id, position in data.items():
        fixed_positions[int(room_id)] = tuple(position)

    return fixed_positions


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

    # Initialize rule engine
    rule_engine = RuleEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid,
    )

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

    # Get RL parameters
    rl_params = get_rl_parameters()

    # Initialize RL engine
    rl_engine = RLEngine(
        bounding_box=(width, length, height),
        grid_size=grid_size,
        structural_grid=structural_grid,
        exploration_rate=rl_params["training"]["exploration_rate"],
        learning_rate=rl_params["training"]["learning_rate"],
        discount_factor=rl_params["training"]["discount_factor"],
    )

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

    # Create metrics calculator
    metrics = LayoutMetrics(layout)

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

    # Get building parameters for metrics
    building_config = get_building_envelope()
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

    # Create renderer
    renderer = LayoutRenderer(layout)

    try:
        # Create 3D visualization
        fig1, ax1 = renderer.render_3d(show_labels=True)
        fig1.suptitle("Hotel Layout - 3D View")

        # Get building parameters
        building_config = get_building_envelope(args.building_config)
        num_floors = building_config["num_floors"] + 1  # +1 for basement

        # Create floor plans for each level
        for floor in range(-1, num_floors - 1):
            fig, ax = renderer.render_floor_plan(floor=floor)
            floor_name = "Basement" if floor == -1 else f"Floor {floor}"
            fig.suptitle(f"Hotel Layout - {floor_name}")

        # Show all figures
        import matplotlib.pyplot as plt

        plt.show()

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure matplotlib is installed and properly configured.")


def save_outputs(layout: SpatialGrid, metrics: Dict[str, Any], args):
    """Save layout to various output formats"""
    print("\nSaving outputs...")

    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Create timestamp for unique filenames and subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped subfolder
    output_subfolder = os.path.join(args.output, timestamp)
    os.makedirs(output_subfolder, exist_ok=True)

    prefix = f"hotel_layout"

    # Determine export formats
    export_formats = [f.strip().lower() for f in args.export_formats.split(",")]

    # Export to requested formats
    for export_format in export_formats:
        if export_format == "json":
            json_file = os.path.join(output_subfolder, f"{prefix}.json")
            export_to_json(layout, json_file)
            print(f"  Saved JSON to {json_file}")

            # Also save metrics
            metrics_file = os.path.join(output_subfolder, f"{prefix}_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved metrics to {metrics_file}")

        # Other export formats...

    # Create visualizations
    renderer = LayoutRenderer(layout)

    try:
        # Get building parameters
        building_config = get_building_envelope(args.building_config)
        num_floors = building_config["num_floors"] + 1  # +1 for basement

        # Save renders to output directory
        print("  Saving visualizations...")
        renderer.save_renders(
            output_dir=output_subfolder,
            prefix=prefix,
            include_3d=True,
            include_floor_plans=True,
            num_floors=num_floors,
        )

        # Close any open matplotlib figures to prevent hanging
        import matplotlib.pyplot as plt

        plt.close("all")

    except Exception as e:
        print(f"  Error saving visualizations: {e}")

    print(f"\nOutputs saved to: {output_subfolder}")


def main():
    """Main entry point for the application"""
    print("Hotel Design AI - Layout Generator")
    print("=================================")

    # Parse command line arguments
    args = parse_arguments()

    # Set random seed if specified
    if args.seed is not None:
        import random
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create rooms from program requirements - pass the correct program config
    from hotel_design_ai.config.config_loader import (
        create_room_objects_from_program,
        get_program_requirements,
    )

    # Override the default to use hotel_requirements.json
    original_get_program_requirements = get_program_requirements

    def patched_get_program_requirements(name=args.program_config):
        return original_get_program_requirements(name)

    # Monkey patch the function to use our specified config
    import hotel_design_ai.config.config_loader

    hotel_design_ai.config.config_loader.get_program_requirements = (
        patched_get_program_requirements
    )

    # Now create the rooms
    room_dicts = create_room_objects_from_program()
    rooms = convert_room_dicts_to_room_objects(room_dicts)

    print(f"Created {len(rooms)} rooms from program requirements")
    # In the main function, after creating rooms:
    print("Room Configuration Check:")
    if rooms and len(rooms) > 0:
        for i in range(min(3, len(rooms))):
            print(
                f"Room {i}: {rooms[i].name}, Type: {rooms[i].room_type}, Metadata: {rooms[i].metadata}"
            )

    # Load fixed room positions if specified
    fixed_positions = None
    if args.fixed_rooms:
        try:
            fixed_positions = load_fixed_rooms(args.fixed_rooms)
            print(f"Loaded {len(fixed_positions)} fixed room positions")
        except Exception as e:
            print(f"Error loading fixed rooms: {e}")

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

    # Evaluate layout
    metrics = evaluate_layout(layout, rooms)

    # Visualize layout if requested
    visualize_layout(layout, args)

    # Save outputs
    save_outputs(layout, metrics, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
