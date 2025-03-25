"""
Configuration loader for Hotel Design AI.
This module loads all configuration from data files and provides
a clean API for accessing configuration throughout the project.
"""

import os
import json
import glob
import math
from typing import Dict, List, Any, Optional, Tuple, Union

# Define paths to data directories
# Move up two directories from this file to get to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")

# Sub-directories for configuration data
BUILDING_DIR = os.path.join(DATA_DIR, "building")
CONSTRAINTS_DIR = os.path.join(DATA_DIR, "constraints")
PROGRAM_DIR = os.path.join(DATA_DIR, "program")
ROOM_TYPES_DIR = os.path.join(DATA_DIR, "room_types")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
RL_DIR = os.path.join(DATA_DIR, "rl")

# Sub-directories for user data
LAYOUTS_DIR = os.path.join(USER_DATA_DIR, "layouts")
FEEDBACK_DIR = os.path.join(USER_DATA_DIR, "feedback")
MODELS_DIR = os.path.join(USER_DATA_DIR, "models")

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    BUILDING_DIR,
    CONSTRAINTS_DIR,
    PROGRAM_DIR,
    ROOM_TYPES_DIR,
    TEMPLATES_DIR,
    RL_DIR,
    USER_DATA_DIR,
    LAYOUTS_DIR,
    FEEDBACK_DIR,
    MODELS_DIR,
]:
    os.makedirs(directory, exist_ok=True)


def _load_json_file(filepath: str, default: Any = None) -> Any:
    """
    Load a JSON file with error handling.

    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist or has errors

    Returns:
        Loaded JSON data or default value
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return default
    except Exception as e:
        return default


def get_building_envelope(name: str = "default") -> Dict[str, Any]:
    """
    Get building envelope parameters.

    Args:
        name: Name of the building configuration

    Returns:
        Dictionary of building parameters
    """
    filepath = os.path.join(BUILDING_DIR, f"{name}.json")

    # Default building envelope if file doesn't exist
    default_envelope = {
        "width": 60.0,
        "length": 80.0,
        "height": 20.0,
        "num_floors": 4,
        "min_floor": -1,
        "max_floor": 3,
        "floor_height": 5.0,
        "structural_grid_x": 8.4,
        "structural_grid_y": 8.4,
        "grid_size": 1.0,
        "main_entry": "flexible",
    }

    result = _load_json_file(filepath, default_envelope)

    # Ensure essential parameters exist
    for key in default_envelope:
        if key not in result:
            result[key] = default_envelope[key]

    return result


def get_program_requirements(name: str = "default") -> Dict[str, Any]:
    """
    Get program requirements.

    Args:
        name: Name of the program configuration

    Returns:
        Dictionary of program requirements
    """
    filepath = os.path.join(PROGRAM_DIR, f"{name}.json")

    # Try to load the program file
    program = _load_json_file(filepath, {})

    # If specific program file doesn't exist, try to load hotel_requirements.json
    if not program:
        alt_path = os.path.join(PROGRAM_DIR, "hotel_requirements.json")
        if os.path.exists(alt_path):
            program = _load_json_file(alt_path, {})

    # If still no program, try to assemble from department files
    if not program:
        program = {}
        department_files = glob.glob(os.path.join(PROGRAM_DIR, "*.json"))

        for dept_file in department_files:
            if os.path.basename(dept_file) != "default.json":
                dept_name = os.path.basename(dept_file).replace(".json", "")
                dept_data = _load_json_file(dept_file, {})
                if dept_data:
                    program[dept_name] = dept_data

    return program


def get_constraints_by_type(constraint_type: str) -> List[Dict[str, Any]]:
    """
    Get constraints of a specific type.

    Args:
        constraint_type: Type of constraints to get (e.g., "adjacency")

    Returns:
        List of constraints of the specified type
    """
    # Try specific file first
    filepath = os.path.join(CONSTRAINTS_DIR, f"{constraint_type}.json")
    type_constraints = _load_json_file(filepath, [])

    # If no specific file, filter from all constraints
    if not type_constraints:
        all_constraints = get_all_constraints()
        type_constraints = [
            c for c in all_constraints if c.get("type") == constraint_type
        ]

    return type_constraints


def get_all_constraints() -> List[Dict[str, Any]]:
    """
    Get all constraints from all constraint files.

    Returns:
        List of constraint definitions
    """
    constraints = []
    constraint_files = glob.glob(os.path.join(CONSTRAINTS_DIR, "*.json"))

    for constraint_file in constraint_files:
        file_constraints = _load_json_file(constraint_file, [])
        if isinstance(file_constraints, list):
            constraints.extend(file_constraints)

    return constraints


def get_adjacency_requirements() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get adjacency requirements in the format expected by the engines.

    Returns:
        Dictionary with required_adjacencies and separation_requirements
    """
    # Get adjacency constraints
    adjacency_constraints = get_constraints_by_type("adjacency")
    separation_constraints = get_constraints_by_type("separation")

    # Convert to tuples
    required_adjacencies = []
    for constraint in adjacency_constraints:
        if "room_type1" in constraint and "room_type2" in constraint:
            pair = (constraint["room_type1"], constraint["room_type2"])
            if pair not in required_adjacencies:
                required_adjacencies.append(pair)

    separation_requirements = []
    for constraint in separation_constraints:
        if "room_type1" in constraint and "room_type2" in constraint:
            pair = (constraint["room_type1"], constraint["room_type2"])
            if pair not in separation_requirements:
                separation_requirements.append(pair)

    return {
        "required_adjacencies": required_adjacencies,
        "separation_requirements": separation_requirements,
    }


def get_room_type(room_type: str) -> Optional[Dict[str, Any]]:
    """
    Get a room type definition.

    Args:
        room_type: Name of the room type

    Returns:
        Room type definition or None if not found
    """
    filepath = os.path.join(ROOM_TYPES_DIR, f"{room_type}.json")
    return _load_json_file(filepath)


def get_all_room_types() -> Dict[str, Dict[str, Any]]:
    """
    Get all room type definitions.

    Returns:
        Dictionary mapping room type names to definitions
    """
    room_types = {}
    type_files = glob.glob(os.path.join(ROOM_TYPES_DIR, "*.json"))

    for type_file in type_files:
        type_name = os.path.basename(type_file).replace(".json", "")
        room_def = _load_json_file(type_file)
        if room_def:
            room_types[type_name] = room_def

    return room_types


def get_template(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a layout template.

    Args:
        template_name: Name of the template

    Returns:
        Template definition or None if not found
    """
    filepath = os.path.join(TEMPLATES_DIR, f"{template_name}.json")
    return _load_json_file(filepath)


def get_rl_parameters() -> Dict[str, Any]:
    """
    Get reinforcement learning parameters.

    Returns:
        Dictionary of RL parameters
    """
    filepath = os.path.join(RL_DIR, "parameters.json")

    # Default RL parameters
    default_params = {
        "weights": {
            "space_efficiency": 1.0,
            "adjacency_satisfaction": 1.5,
            "circulation_quality": 1.2,
            "natural_light": 0.8,
            "structural_alignment": 1.0,
        },
        "training": {
            "learning_rate": 0.001,
            "exploration_rate": 0.7,
            "discount_factor": 0.99,
            "batch_size": 32,
        },
    }

    return _load_json_file(filepath, default_params)


def get_design_constraints() -> Dict[str, Any]:
    """
    Get design constraints.

    Returns:
        Dictionary of design constraints
    """
    filepath = os.path.join(CONSTRAINTS_DIR, "design.json")

    # Default design constraints
    default_constraints = {
        "zoning": {
            "public_areas": {"preferred_floors": [0, 1]},
            "back_of_house": {"preferred_floors": [2, 3, -1]},
            "parking": {"preferred_floors": [-1]},
        },
        "circulation": {
            "public_corridor_width": 2.4,
            "service_corridor_width": 1.8,
            "min_exit_distance": 30.0,
        },
        "structural": {
            "column_free_spaces": ["ballroom", "pool"],
            "large_span_adjustment": 1.2,
        },
        "access": {
            "guest_entry": {"preferred_sides": ["front"]},
            "service_entry": {"preferred_sides": ["back", "side"]},
            "parking_entry": {"preferred_sides": ["side"]},
        },
        "floor_heights": {"public_spaces": 5.0, "back_of_house": 3.5, "parking": 3.0},
    }

    return _load_json_file(filepath, default_constraints)


def create_room_objects_from_program(program_config="default") -> List[Dict[str, Any]]:
    """
    Create room objects from program requirements with improved flexibility.
    Adapts to changes in building configuration and program requirements.
    Now supports a list of preferred floors instead of just a single floor.

    Args:
        program_config: Name of the program configuration file to use

    Returns:
        List of room dictionaries
    """
    # Get program requirements
    program = get_program_requirements(program_config)

    # Get room type definitions for reference
    room_types = get_all_room_types()

    # Get building envelope for reference
    building = get_building_envelope()
    structural_grid_x = building.get("structural_grid_x", 8.0)
    structural_grid_y = building.get("structural_grid_y", 8.0)
    min_floor = building.get("min_floor", -1)
    max_floor = building.get("max_floor", 3)
    floor_capacity = {}

    # Building area for floor capacity calculations
    building_area = building.get("width", 80.0) * building.get("length", 100.0)

    # Create floor capacity dict dynamically based on the floor range
    for floor in range(min_floor, max_floor + 1):
        # Adjust capacity based on floor type (reduced for basement, etc.)
        if floor < 0:
            # Basements typically can use more of the floor area
            floor_capacity[floor] = building_area * 0.85
        else:
            # Above-ground floors have standard capacity
            floor_capacity[floor] = building_area * 0.7

    floor_area_used = {floor: 0.0 for floor in range(min_floor, max_floor + 1)}

    # Create rooms from program requirements
    all_rooms = []
    room_id = 1

    for department_key, department in program.items():
        for space_key, space in department.items():
            # Allow configuration to specify whether to skip logistics reserves
            skip_logistics = building.get("skip_logistics_reserve", True)
            if skip_logistics and "logistics_reserve" in space_key:
                continue

            area = space["area"]
            room_type = space["room_type"]

            # Try to get dimensions from room type definition
            room_type_def = room_types.get(room_type, {})
            dimensions = room_type_def.get("dimensions", {})

            # Get minimum dimensions from either the space or the room type definition
            min_width = space.get("min_width", dimensions.get("width", 5.0))
            min_height = space.get("min_height", dimensions.get("height", 3.5))

            # Determine floor assignment - respect the floor specified in the program if available
            # Support both a single floor integer and a list of floors
            floor = space.get("floor")
            # Ensure preferred_floors is available for the rule engine
            if isinstance(floor, list):
                preferred_floors = floor
            else:
                preferred_floors = [floor] if floor is not None else None

            # Create a more complete metadata dictionary
            metadata = {
                "original_name": space_key,
                "department": department_key,
            }

            # Add any additional metadata from the space
            for key, value in space.items():
                if key not in [
                    "area",
                    "room_type",
                    "min_width",
                    "min_height",
                    "floor",
                    "details",
                    "requires_natural_light",
                    "requires_adjacency",
                    "requires_separation",
                ]:
                    metadata[key] = value

            # Always include preferred_floors in the metadata
            metadata["preferred_floors"] = preferred_floors

            # Check if details are provided
            if "details" in space:
                # Create rooms for each detailed space
                for detail_key, detail_area in space["details"].items():
                    # Skip very small details or use a minimum area
                    if detail_area < 5:
                        continue

                    # Calculate appropriate dimensions based on area and constrains
                    detail_width, detail_length = _calculate_room_dimensions(
                        detail_area, min_width, structural_grid_x, structural_grid_y
                    )

                    # Create detailed metadata
                    detail_metadata = metadata.copy()
                    detail_metadata["subspace_name"] = detail_key
                    detail_metadata["original_name"] = f"{space_key} - {detail_key}"

                    # Create room dictionary
                    room = {
                        "id": room_id,
                        "name": f"{detail_key}",
                        "width": detail_width,
                        "length": detail_length,
                        "height": min_height,
                        "room_type": room_type,
                        "department": department_key,
                        "requires_natural_light": space.get(
                            "requires_natural_light", False
                        ),
                        "requires_adjacency": space.get("requires_adjacency", []),
                        "requires_separation": space.get("requires_separation", []),
                        "floor": floor,  # Keep original floor value (can be int or list)
                        "preferred_floors": preferred_floors,  # Add preferred_floors
                        "metadata": detail_metadata,
                    }
                    all_rooms.append(room)
                    room_id += 1

                    # Update floor area tracking if floor is specified as a single int
                    if isinstance(floor, int):
                        floor_area_used[floor] += detail_area
                    elif isinstance(floor, list) and len(floor) > 0:
                        # Just track on the first floor if it's a list
                        first_floor = floor[0]
                        if first_floor in floor_area_used:
                            floor_area_used[first_floor] += detail_area
            else:
                # Calculate appropriate dimensions
                width, length = _calculate_room_dimensions(
                    area, min_width, structural_grid_x, structural_grid_y
                )

                # Create room dictionary
                room = {
                    "id": room_id,
                    "name": f"{space_key}",
                    "width": width,
                    "length": length,
                    "height": min_height,
                    "room_type": room_type,
                    "department": department_key,
                    "requires_natural_light": space.get(
                        "requires_natural_light", False
                    ),
                    "requires_adjacency": space.get("requires_adjacency", []),
                    "requires_separation": space.get("requires_separation", []),
                    "floor": floor,  # Keep original floor value (can be int or list)
                    "preferred_floors": preferred_floors,  # Add preferred_floors
                    "metadata": metadata,
                }
                all_rooms.append(room)
                room_id += 1

                # Update floor area tracking if floor is specified as a single int
                if isinstance(floor, int):
                    floor_area_used[floor] += area
                elif isinstance(floor, list) and len(floor) > 0:
                    # Just track on the first floor if it's a list
                    first_floor = floor[0]
                    if first_floor in floor_area_used:
                        floor_area_used[first_floor] += area

    # Print floor usage summary to help with debugging
    print("\nFloor utilization in program requirements:")
    for floor in sorted(floor_area_used.keys()):
        capacity = floor_capacity.get(floor, 0)
        usage = floor_area_used.get(floor, 0)
        percentage = (usage / capacity * 100) if capacity > 0 else 0
        floor_name = "Basement" if floor < 0 else f"Floor {floor}"
        print(
            f"  {floor_name}: {usage:.1f} m² used of {capacity:.1f} m² ({percentage:.1f}%)"
        )

    return all_rooms


def _calculate_room_dimensions(area, min_width, grid_x, grid_y):
    """
    Calculate optimal room dimensions based on area and constraints.
    Enhanced to better align with structural grid.

    Args:
        area: Required area in square meters
        min_width: Minimum width in meters
        grid_x: X structural grid spacing
        grid_y: Y structural grid spacing

    Returns:
        Tuple of (width, length) in meters
    """
    # Ensure area is positive
    area = max(1.0, area)

    # For very large areas, align with structural grid
    if area > 300:
        # Find dimensions that are multiples of structural grid
        grid_multiplier_w = max(1, round(min_width / grid_x))
        width = grid_multiplier_w * grid_x

        # Calculate length based on required area
        length = math.ceil(area / width)

        # Adjust length to grid if possible
        grid_multiplier_l = max(1, round(length / grid_y))
        grid_length = grid_multiplier_l * grid_y

        # If grid_length is reasonably close to the required length, use it
        if 0.8 <= (grid_length / length) <= 1.2:
            length = grid_length
    else:
        # For smaller rooms, use more flexible dimensions
        width = max(min_width, math.sqrt(area / 2))  # Avoid excessively narrow rooms
        length = area / width

        # Round to nearest half meter for better placement
        width = round(width * 2) / 2
        length = round(length * 2) / 2

        # Ensure minimum width
        if width < min_width:
            width = min_width
            length = area / width

        # If length is too long compared to width, adjust
        if length > 3 * width:
            width = math.sqrt(area)
            length = area / width

            # Round again
            width = round(width * 2) / 2
            length = round(length * 2) / 2

    # Ensure width >= min_width
    width = max(width, min_width)

    # Recalculate length to ensure we have at least the required area
    actual_area = width * length
    if actual_area < area:
        # Increase length to meet area requirement
        length = area / width

    return width, length


def save_default_files():
    """
    Create default data files if they don't exist.
    This is useful for first-time setup.
    """
    # Default building envelope
    building_file = os.path.join(BUILDING_DIR, "default.json")
    if not os.path.exists(building_file):
        with open(building_file, "w", encoding="utf-8") as f:
            json.dump(get_building_envelope(), f, indent=2)

    # Default RL parameters
    rl_file = os.path.join(RL_DIR, "parameters.json")
    if not os.path.exists(rl_file):
        with open(rl_file, "w", encoding="utf-8") as f:
            json.dump(get_rl_parameters(), f, indent=2)

    # Default design constraints
    design_file = os.path.join(CONSTRAINTS_DIR, "design.json")
    if not os.path.exists(design_file):
        with open(design_file, "w", encoding="utf-8") as f:
            json.dump(get_design_constraints(), f, indent=2)


# Create default files on module import
save_default_files()
