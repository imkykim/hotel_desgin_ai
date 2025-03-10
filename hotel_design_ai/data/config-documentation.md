# Hotel Design AI Configuration Guide

This guide explains how to configure the Hotel Design AI system using the data-driven configuration approach.

## Overview

The Hotel Design AI project uses a data-driven configuration system where all settings are stored in JSON files in the `data/` directory. This approach offers several benefits:

- **Clear organization**: Different types of configurations are in separate files
- **Easy editing**: JSON files can be edited without programming knowledge
- **Version control**: Configuration changes can be tracked with Git
- **Reusability**: Settings can be reused across different projects

## Configuration Directory Structure

```
data/
├── building/          # Building envelope configurations
│   └── default.json   # Default building parameters
├── constraints/       # Design constraints 
│   ├── adjacency.json # Adjacency requirements
│   ├── design.json    # General design constraints
│   ├── exterior.json  # Exterior access requirements
│   ├── floor.json     # Floor assignment constraints
│   └── separation.json# Separation requirements
├── program/           # Program requirements
│   └── default.json   # Full program specification
├── room_types/        # Room type definitions
│   ├── guest_room.json# Detailed guest room specifications
│   └── ...
├── templates/         # Layout templates
│   └── standard_floor.json
└── rl/                # Reinforcement learning parameters
    └── parameters.json
```

## Building Configuration

The building envelope is defined in `data/building/default.json`:

```json
{
  "width": 60.0,
  "length": 80.0,
  "height": 20.0,
  "num_floors": 4,
  "floor_height": 5.0,
  "structural_grid_x": 8.4,
  "structural_grid_y": 8.4,
  "grid_size": 1.0,
  "main_entry": "flexible"
}
```

You can create alternative building configurations by creating new files (e.g., `compact.json`, `tower.json`) in the same directory.

## Program Requirements

Program requirements are defined in `data/program/default.json`. This file has a hierarchical structure:

1. **Departments**: Top-level categories (dining, administrative, meeting, etc.)
2. **Spaces**: Individual functional spaces within each department
3. **Details**: Sub-spaces within a space (optional)

Example:

```json
{
  "dining": {
    "kitchen": {
      "area": 1300,
      "details": {
        "main_kitchen": 500,
        "chinese_kitchen": 300,
        ...
      },
      "room_type": "kitchen",
      "min_width": 20.0,
      "min_height": 4.0,
      "requires_exhaust": true,
      "requires_adjacency": ["restaurants"]
    },
    ...
  },
  ...
}
```

### Required Space Properties

- `area`: Total area in square meters
- `room_type`: Type of room (must match one defined in room_types or be a standard type)

### Optional Space Properties

- `details`: Sub-spaces with areas that add up to the total area
- `min_width`: Minimum width constraint
- `min_height`: Minimum height constraint
- `requires_natural_light`: Whether the space needs exterior access (true/false)
- `requires_adjacency`: List of room types that should be adjacent
- `requires_separation`: List of room types that should not be adjacent
- `floor`: Preferred floor level (0 = ground floor, -1 = basement)

## Constraints

Constraints are divided into several categories, each in its own file:

### Adjacency Constraints (`adjacency.json`)

```json
[
  {
    "type": "adjacency",
    "room_type1": "lobby",
    "room_type2": "entrance",
    "weight": 2.0,
    "description": "Main entrance should be adjacent to lobby"
  },
  ...
]
```

### Separation Constraints (`separation.json`)

```json
[
  {
    "type": "separation",
    "room_type1": "meeting_room",
    "room_type2": "mechanical",
    "weight": 1.5,
    "description": "Mechanical spaces should be separated from meeting rooms"
  },
  ...
]
```

### Floor Constraints (`floor.json`)

```json
[
  {
    "type": "floor",
    "room_type": "lobby",
    "floor": 0,
    "weight": 2.0,
    "description": "Lobby should be on ground floor"
  },
  ...
]
```

### Exterior Access Constraints (`exterior.json`)

```json
[
  {
    "type": "exterior",
    "room_type": "guest_room",
    "weight": 1.5,
    "description": "Guest rooms should have exterior access"
  },
  ...
]
```

### Design Constraints (`design.json`)

General design constraints are defined in `design.json`:

```json
{
  "zoning": {
    "public_areas": {
      "preferred_floors": [0, 1]
    },
    ...
  },
  "circulation": {
    "public_corridor_width": 2.4,
    "service_corridor_width": 1.8,
    ...
  },
  ...
}
```

## Room Types

Room type definitions provide detailed specifications for each type of room:

```json
{
  "name": "Standard Guest Room",
  "type": "guest_room",
  "dimensions": {
    "width": 4.0,
    "length": 8.0,
    "height": 3.0
  },
  "requirements": {
    "natural_light": true,
    "exterior_access": true,
    "min_area": 32.0,
    "preferred_adjacencies": ["vertical_circulation"],
    "avoid_adjacencies": ["service_area", "back_of_house"]
  },
  "variants": [
    {
      "name": "King Room",
      "dimensions": {
        "width": 4.5,
        "length": 9.0,
        "height": 3.0
      },
      "min_area": 40.0
    },
    ...
  ],
  ...
}
```

## RL Parameters

Reinforcement learning parameters are defined in `data/rl/parameters.json`:

```json
{
  "weights": {
    "space_efficiency": 1.0,
    "adjacency_satisfaction": 1.5,
    "circulation_quality": 1.2,
    ...
  },
  "training": {
    "learning_rate": 0.001,
    "exploration_rate": 0.7,
    ...
  },
  ...
}
```

## Using Configuration in the Application

The configuration is loaded through the `config_loader.py` module:

```python
from hotel_design_ai.config.config_loader import (
    get_building_envelope, get_program_requirements, get_adjacency_requirements,
    get_design_constraints, get_rl_parameters, create_room_objects_from_program
)

# Get building parameters
building = get_building_envelope()  # Or specify config: get_building_envelope("compact")

# Get program
program = get_program_requirements()

# Get constraints
adjacency = get_adjacency_requirements()

# Create rooms from program
rooms = create_room_objects_from_program()
```

## Command Line Options

The main application accepts configuration parameters through command-line arguments:

```bash
# Use a specific building configuration
python main.py --building-config compact

# Use a specific program configuration
python main.py --program-config hotel_small
```

## Adding New Configuration Files

To add new configuration files:

1. Create a new JSON file in the appropriate directory
2. Follow the schema of existing files
3. Use the new configuration file name in command-line arguments

## Configuration Best Practices

1. **Maintain Consistency**: Use consistent naming conventions
2. **Document Constraints**: Add descriptions to all constraints
3. **Separate Concerns**: Keep different types of constraints in different files
4. **Use Weights**: Adjust constraint weights to prioritize requirements
5. **Create Templates**: For frequently used configurations, create template files

## Advanced Configuration

For advanced users, you can:

1. **Create Department-Specific Files**: Split program into multiple files
2. **Create Project Templates**: Store complete project configurations
3. **Parameterize Configurations**: Use scripts to generate configurations
4. **Version Control**: Track configuration changes with Git
