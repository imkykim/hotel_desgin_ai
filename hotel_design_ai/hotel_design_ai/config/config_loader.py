"""
Configuration loader for Hotel Design AI.
This module loads all configuration from data files and provides
a clean API for accessing configuration throughout the project.
"""

import os
import json
import glob
import math
from typing import Dict, List, Any, Optional, Tuple

# Define paths to data directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Sub-directories
BUILDING_DIR = os.path.join(DATA_DIR, "building")
CONSTRAINTS_DIR = os.path.join(DATA_DIR, "constraints")
PROGRAM_DIR = os.path.join(DATA_DIR, "program")
ROOM_TYPES_DIR = os.path.join(DATA_DIR, "room_types")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
RL_DIR = os.path.join(DATA_DIR, "rl")

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    BUILDING_DIR,
    CONSTRAINTS_DIR,
    PROGRAM_DIR,
    ROOM_TYPES_DIR,
    TEMPLATES_DIR,
    RL_DIR,
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
    Create room objects from program requirements with improved floor distribution.

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

    # Define floor mapping for various room types
    default_floor_mapping = {
        # Ground floor (public access)
        "entrance": 0,
        "lobby": 0,
        "restaurant": 0,
        "retail": 0,
        "service": 0,
        "ballroom": 0,
        "meeting_room": 0,
        "food_service": 0,
        "pool": 0,
        # Upper floors
        "guest_room": 1,
        "office": 1,
        "staff_area": 1,
        # Basement
        "parking": -1,
        "mechanical": -1,
        "maintenance": -1,
        "back_of_house": -1,
        "kitchen": 0,  # could be basement or ground
    }

    # Building area for floor capacity calculations
    building_area = building.get("width", 80.0) * building.get("length", 100.0)
    floor_capacity = {
        floor: building_area * 0.7 for floor in range(min_floor, max_floor + 1)
    }
    floor_area_used = {floor: 0.0 for floor in range(min_floor, max_floor + 1)}

    # Create rooms from program requirements
    all_rooms = []
    room_id = 1

    for department_key, department in program.items():
        for space_key, space in department.items():
            # Skip logistics reserves for initial design
            if "logistics_reserve" in space_key:
                continue

            area = space["area"]
            room_type = space["room_type"]

            # Try to get dimensions from room type definition
            room_type_def = room_types.get(room_type, {})
            dimensions = room_type_def.get("dimensions", {})

            min_width = space.get("min_width", dimensions.get("width", 5.0))
            min_height = space.get("min_height", dimensions.get("height", 3.5))

            # Determine floor assignment
            floor = space.get("floor")
            if floor is None:
                # Use default mapping with balancing
                default_floor = default_floor_mapping.get(room_type)

                if default_floor is not None:
                    # Check if default floor is over capacity
                    if (
                        floor_area_used[default_floor] + area
                        > floor_capacity[default_floor]
                    ):
                        # Find alternative floor with capacity
                        alternative_floors = _get_alternative_floors(
                            room_type, min_floor, max_floor
                        )

                        # Choose best alternative
                        for alt_floor in sorted(
                            alternative_floors,
                            key=lambda f: floor_area_used[f] / floor_capacity[f],
                        ):
                            if (
                                floor_area_used[alt_floor] + area
                                <= floor_capacity[alt_floor]
                            ):
                                floor = alt_floor
                                break
                        else:
                            # No good alternative, use default
                            floor = default_floor
                    else:
                        floor = default_floor
                else:
                    # If no default mapping, distribute evenly
                    floor_usage = [
                        (f, floor_area_used[f] / floor_capacity[f])
                        for f in range(min_floor, max_floor + 1)
                    ]
                    floor_usage.sort(key=lambda x: x[1])  # Sort by usage ratio
                    floor = floor_usage[0][0]  # Take least used floor

            # Calculate dimensions - try to align with structural grid
            width, length = _calculate_room_dimensions(
                area, min_width, structural_grid_x, structural_grid_y
            )
            height = min_height

            # Update floor area tracking
            floor_area_used[floor] += area

            # Check if details are provided
            if "details" in space:
                # Create rooms for each detailed space
                for detail_key, detail_area in space["details"].items():
                    # Skip very small details
                    if detail_area < 10:
                        continue

                    detail_width, detail_length = _calculate_room_dimensions(
                        detail_area, min_width, structural_grid_x, structural_grid_y
                    )

                    # Create room dictionary
                    room = {
                        "id": room_id,
                        "name": f"{detail_key}",
                        "width": detail_width,
                        "length": detail_length,
                        "height": height,
                        "room_type": room_type,
                        "department": department_key,
                        "requires_natural_light": space.get(
                            "requires_natural_light", False
                        ),
                        "requires_adjacency": space.get("requires_adjacency", []),
                        "floor": floor,
                        "metadata": {
                            "original_name": detail_key,
                            "department": department_key,
                        },
                    }
                    all_rooms.append(room)
                    room_id += 1
            else:
                # Create single room for this space
                room = {
                    "id": room_id,
                    "name": f"{space_key}",
                    "width": width,
                    "length": length,
                    "height": height,
                    "room_type": room_type,
                    "department": department_key,
                    "requires_natural_light": space.get(
                        "requires_natural_light", False
                    ),
                    "requires_adjacency": space.get("requires_adjacency", []),
                    "floor": floor,
                    "metadata": {
                        "original_name": space_key,
                        "department": department_key,
                    },
                }
                all_rooms.append(room)
                room_id += 1

    return all_rooms


def _get_alternative_floors(room_type, min_floor, max_floor):
    """Get alternative floors for a room type when default floor is full"""
    if room_type == "guest_room":
        # Guest rooms can go on any upper floor
        return list(range(1, max_floor + 1))
    elif room_type in ["office", "staff_area"]:
        # Office spaces can go on any upper floor
        return list(range(1, max_floor + 1))
    elif room_type in ["mechanical", "maintenance", "back_of_house"]:
        # Service spaces can go in basement or upper floors
        return [min_floor] + list(range(2, max_floor + 1))
    elif room_type in ["meeting_room", "food_service"]:
        # These can go on ground or 1st floor
        return [0, 1]
    else:
        # For other types, check all floors
        return list(range(min_floor, max_floor + 1))


def _calculate_room_dimensions(area, min_width, grid_x, grid_y):
    """Calculate optimal room dimensions based on area and constraints"""
    if area > 300:
        # Find dimensions that are multiples of structural grid
        grid_multiplier_w = max(1, round(min_width / grid_x))
        width = grid_multiplier_w * grid_x
        length = math.ceil(area / width)

        # Adjust length to grid if possible
        grid_multiplier_l = max(1, round(length / grid_y))
        grid_length = grid_multiplier_l * grid_y

        # If grid_length is reasonably close, use it
        if 0.8 <= (grid_length / length) <= 1.2:
            length = grid_length
    else:
        # For smaller rooms, use more flexible dimensions
        width = min_width
        length = area / width

        # If length is too long compared to width, adjust
        if length > 2 * width:
            width = (area / 2) ** 0.5
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
