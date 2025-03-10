# Hotel Design AI: Data Files Reference

This document provides a detailed explanation of all data files used in the Hotel Design AI system, including available options, constraints, and how to modify them.

## Table of Contents

1. [Building Configuration](#1-building-configuration)
2. [Program Requirements](#2-program-requirements)
3. [Constraints](#3-constraints)
   - [Adjacency Constraints](#31-adjacency-constraints)
   - [Separation Constraints](#32-separation-constraints)
   - [Floor Constraints](#33-floor-constraints)
   - [Exterior Constraints](#34-exterior-constraints)
   - [Design Constraints](#35-design-constraints)
4. [Room Types Reference](#4-room-types-reference)
5. [RL Parameters](#5-reinforcement-learning-parameters)
6. [Templates](#6-templates)
7. [Common Modifications](#7-common-modifications)

---

## 1. Building Configuration

Located in `data/building/default.json` and other custom configurations.

### Schema

| Field | Type | Description | Available Options |
|-------|------|-------------|-------------------|
| `width` | float | Building width in meters | Any positive value |
| `length` | float | Building length in meters | Any positive value |
| `height` | float | Building total height in meters | Any positive value |
| `num_floors` | integer | Number of floors above ground | Typically 1-50 |
| `floor_height` | float | Height of each floor in meters | Typically 3.0-5.0 |
| `structural_grid_x` | float | Column spacing in X direction | Typically 6.0-12.0 |
| `structural_grid_y` | float | Column spacing in Y direction | Typically 6.0-12.0 |
| `grid_size` | float | Resolution for spatial grid | Smaller = more precise but slower (0.5-2.0) |
| `main_entry` | string | Location of main entry | "north", "south", "east", "west", "flexible" |

### Example

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
  "main_entry": "flexible",
  "description": "Standard hotel building envelope"
}
```

---

## 2. Program Requirements

Located in `data/program/default.json` or department-specific files.

### Schema

Program requirements have a hierarchical structure:

1. **Departments**: Top-level categories
2. **Spaces**: Functional spaces within departments
3. **Details** (optional): Sub-spaces within a space

#### Department Schema

Each department is a top-level key in the JSON:

```json
{
  "dining": { ... },
  "administrative": { ... },
  "meeting": { ... }
}
```

#### Space Schema

| Field | Type | Required | Description | Available Options |
|-------|------|----------|-------------|-------------------|
| `area` | float | Yes | Total area in square meters | Any positive value |
| `room_type` | string | Yes | Type of room | See [Room Types Reference](#4-room-types-reference) |
| `details` | object | No | Sub-spaces with areas | Key-value pairs of names and areas |
| `min_width` | float | No | Minimum width constraint | Typically 3.0+ |
| `min_height` | float | No | Minimum height | Typically 2.4+ |
| `min_length` | float | No | Minimum length | Any positive value |
| `requires_natural_light` | boolean | No | Needs exterior access | true/false |
| `requires_adjacency` | array | No | Room types that should be adjacent | Array of room types |
| `requires_separation` | array | No | Room types to separate from | Array of room types |
| `requires_column_free` | boolean | No | Needs column-free space | true/false |
| `floor` | integer | No | Preferred floor | -1 (basement), 0 (ground), 1+ (upper) |

### Example

```json
"kitchen": {
  "area": 1300,
  "details": {
    "main_kitchen": 500,
    "chinese_kitchen": 300,
    "regional_kitchens": 200,
    "western_kitchen": 200,
    "pastry_kitchen": 100
  },
  "room_type": "kitchen",
  "min_width": 20.0,
  "min_height": 4.0,
  "requires_exhaust": true,
  "requires_adjacency": ["restaurants"]
}
```

---

## 3. Constraints

Constraints are divided into several categories, each in its own file.

### 3.1 Adjacency Constraints

Located in `data/constraints/adjacency.json`.

#### Schema

| Field | Type | Required | Description | Available Options |
|-------|------|----------|-------------|-------------------|
| `type` | string | Yes | Must be "adjacency" | "adjacency" |
| `room_type1` | string | Yes | First room type | Any room type |
| `room_type2` | string | Yes | Second room type | Any room type |
| `weight` | float | Yes | Importance weight | 0.1-3.0 (higher = more important) |
| `description` | string | No | Human-readable description | Any text |

#### Example

```json
{
  "type": "adjacency",
  "room_type1": "lobby",
  "room_type2": "entrance",
  "weight": 2.0,
  "description": "Main entrance should be adjacent to lobby"
}
```

### 3.2 Separation Constraints

Located in `data/constraints/separation.json`.

#### Schema

| Field | Type | Required | Description | Available Options |
|-------|------|----------|-------------|-------------------|
| `type` | string | Yes | Must be "separation" | "separation" |
| `room_type1` | string | Yes | First room type | Any room type |
| `room_type2` | string | Yes | Second room type | Any room type |
| `weight` | float | Yes | Importance weight | 0.1-3.0 (higher = more important) |
| `description` | string | No | Human-readable description | Any text |

#### Example

```json
{
  "type": "separation",
  "room_type1": "meeting_room",
  "room_type2": "mechanical",
  "weight": 1.5,
  "description": "Mechanical spaces should be separated from meeting rooms"
}
```

### 3.3 Floor Constraints

Located in `data/constraints/floor.json`.

#### Schema

| Field | Type | Required | Description | Available Options |
|-------|------|----------|-------------|-------------------|
| `type` | string | Yes | Must be "floor" | "floor" |
| `room_type` | string | Yes | Room type | Any room type |
| `floor` | integer | Yes | Preferred floor | -1 (basement), 0 (ground), 1+ (upper) |
| `weight` | float | Yes | Importance weight | 0.1-3.0 (higher = more important) |
| `description` | string | No | Human-readable description | Any text |

#### Example

```json
{
  "type": "floor",
  "room_type": "lobby",
  "floor": 0,
  "weight": 2.0,
  "description": "Lobby should be on ground floor"
}
```

### 3.4 Exterior Constraints

Located in `data/constraints/exterior.json`.

#### Schema

| Field | Type | Required | Description | Available Options |
|-------|------|----------|-------------|-------------------|
| `type` | string | Yes | Must be "exterior" | "exterior" |
| `room_type` | string | Yes | Room type | Any room type |
| `weight` | float | Yes | Importance weight | 0.1-3.0 (higher = more important) |
| `description` | string | No | Human-readable description | Any text |

#### Example

```json
{
  "type": "exterior",
  "room_type": "guest_room",
  "weight": 1.5,
  "description": "Guest rooms should have exterior access"
}
```

### 3.5 Design Constraints

Located in `data/constraints/design.json`.

This file contains multiple categories of design constraints.

#### Zoning Schema

Controls which areas should be on which floors.

```json
"zoning": {
  "public_areas": {
    "preferred_floors": [0, 1],
    "description": "Public areas should be on ground and first floors"
  },
  "back_of_house": {
    "preferred_floors": [2, 3, -1],
    "description": "Back of house areas can be on upper floors or basement"
  },
  "parking": {
    "preferred_floors": [-1],
    "description": "Parking should be in basement"
  }
}
```

#### Circulation Schema

Controls corridor widths and travel distances.

| Field | Type | Description | Available Options |
|-------|------|-------------|-------------------|
| `public_corridor_width` | float | Width of public corridors | Typically 1.8-3.0 |
| `service_corridor_width` | float | Width of service corridors | Typically 1.5-2.4 |
| `min_exit_distance` | float | Minimum distance between exits | Based on code requirements |
| `max_travel_distance` | float | Maximum travel distance to exit | Based on code requirements |

#### Structural Schema

Defines structural requirements.

| Field | Type | Description | Available Options |
|-------|------|-------------|-------------------|
| `column_free_spaces` | array | Room types that need no columns | Array of room types |
| `large_span_adjustment` | float | Factor for large spans | Typically 1.1-1.5 |

#### Access Schema

Controls building access points.

```json
"access": {
  "guest_entry": {
    "preferred_sides": ["front"],
    "description": "Guest entry should be on front of building"
  },
  "service_entry": {
    "preferred_sides": ["back", "side"],
    "description": "Service entry should be on back or side"
  }
}
```

#### Floor Heights Schema

Sets height requirements for different space types.

| Field | Type | Description | Available Options |
|-------|------|-------------|-------------------|
| `public_spaces` | float | Height for public spaces | Typically 4.0-6.0 |
| `back_of_house` | float | Height for back of house | Typically 3.0-4.0 |
| `parking` | float | Height for parking | Typically 2.5-3.5 |

---

## 4. Room Types Reference

This is a complete list of all available room types in the system, their descriptions, and typical characteristics.

| Room Type | Description | Natural Light | Typical Width (m) | Typical Height (m) | Typical Area (m²) |
|-----------|-------------|---------------|-------------------|-------------------|------------------|
| `entrance` | Main building entrance | Required | 8.0 | 4.5 | 48-80 |
| `lobby` | Main hotel lobby | Preferred | 15.0 | 4.5 | 200-1000 |
| `vertical_circulation` | Elevators, stairs, escalators | Not required | 8.0 | Building height | 50-100 per floor |
| `restaurant` | Dining spaces | Preferred | 15.0 | 4.0 | 150-500 |
| `kitchen` | Food preparation areas | Not required | 15.0 | 4.0 | 200-1500 |
| `meeting_room` | Conference and meeting spaces | Preferred | 8.0-15.0 | 3.5-4.5 | 50-200 |
| `ballroom` | Large event space | Not required | 20.0+ | 5.0+ | 400-1500 |
| `pre_function` | Space outside ballroom/meeting rooms | Not required | Variable | 3.5-4.5 | 30-50% of event space |
| `guest_room` | Hotel rooms for guests | Required | 4.0-6.0 | 3.0 | 30-80 |
| `service_area` | Back of house service spaces | Not required | Variable | 3.5 | Variable |
| `back_of_house` | Staff and service areas | Not required | Variable | 3.5 | Variable |
| `mechanical` | Equipment and mechanical spaces | Not required | Variable | 3.5-4.5 | Variable |
| `office` | Administrative offices | Preferred | 3.0-6.0 | 3.0-3.5 | 9-15 per person |
| `staff_area` | Staff facilities | Not required | Variable | 3.0-3.5 | Variable |
| `pool` | Swimming pool area | Preferred | 25.0+ | 4.5-6.0 | 300-700 |
| `fitness` | Exercise and gym area | Preferred | 10.0+ | 3.5-4.5 | 100-300 |
| `retail` | Shops and retail spaces | Preferred | 6.0-15.0 | 3.5-4.5 | 50-500 |
| `recreation` | Recreational facilities | Not required | 10.0+ | 3.5 | 50-250 |
| `entertainment` | Entertainment venues | Not required | 15.0+ | 3.5-5.0 | 200-800 |
| `food_service` | Bars, cafes, lounges | Preferred | 8.0-15.0 | 3.5-4.5 | 50-250 |
| `parking` | Car parking areas | Not required | 30.0+ | 2.5-3.0 | 20-25 per space |
| `circulation` | Corridors and hallways | Not required | 1.8-3.0 | Same as adjacent spaces | Variable |
| `lounge` | Seating and waiting areas | Preferred | 8.0+ | 3.5 | 50-200 |
| `maintenance` | Maintenance workshops and storage | Not required | 5.0+ | 3.5 | 20-150 |

### Adding New Room Types

To add a new room type:

1. Create a room type definition in `data/room_types/your_new_type.json`
2. Use the new room type in program requirements
3. Add necessary constraints for the new room type

---

## 5. Reinforcement Learning Parameters

Located in `data/rl/parameters.json`.

### Weights Schema

Importance weights for different optimization objectives.

| Field | Type | Description | Available Range |
|-------|------|-------------|-----------------|
| `space_efficiency` | float | Weight for space utilization | 0.1-3.0 |
| `adjacency_satisfaction` | float | Weight for adjacency constraints | 0.1-3.0 |
| `circulation_quality` | float | Weight for circulation efficiency | 0.1-3.0 |
| `natural_light` | float | Weight for natural light access | 0.1-3.0 |
| `structural_alignment` | float | Weight for structural grid alignment | 0.1-3.0 |
| `department_clustering` | float | Weight for department clustering | 0.1-3.0 |
| `vertical_stacking` | float | Weight for vertical consistency | 0.1-3.0 |

### Training Schema

Parameters controlling the RL training process.

| Field | Type | Description | Available Range |
|-------|------|-------------|-----------------|
| `learning_rate` | float | Rate of model updates | 0.0001-0.01 |
| `exploration_rate` | float | Initial exploration rate | 0.1-1.0 |
| `exploration_min` | float | Minimum exploration rate | 0.01-0.1 |
| `exploration_decay` | float | Rate of exploration decay | 0.9-0.999 |
| `discount_factor` | float | Weight of future rewards | 0.9-0.999 |
| `batch_size` | integer | Training batch size | 16-256 |
| `max_iterations` | integer | Maximum training iterations | 10-1000 |

### Example

```json
{
  "weights": {
    "space_efficiency": 1.0,
    "adjacency_satisfaction": 1.5,
    "circulation_quality": 1.2,
    "natural_light": 0.8,
    "structural_alignment": 1.0
  },
  "training": {
    "learning_rate": 0.001,
    "exploration_rate": 0.7,
    "discount_factor": 0.99,
    "batch_size": 32
  }
}
```

---

## 6. Templates

Located in `data/templates/`.

Templates define pre-configured arrangements of rooms that can be used as starting points.

### Standard Floor Template

For a typical guest room floor with double-loaded corridor.

#### Fixed Elements Schema

Elements that must be in specific positions.

| Field | Type | Description | Available Options |
|-------|------|-------------|-------------------|
| `type` | string | Room type | Any room type |
| `position_type` | string | How position is defined | "absolute" or "derived" |
| `position` | array | [x, y, z] coordinates | Any valid coordinates |
| `dimensions` | array | [width, length, height] | Any valid dimensions |
| `name` | string | Element name | Any text |
| `required` | boolean | Whether element is required | true/false |

#### Room Arrangements Schema

Patterns for arranging multiple rooms.

| Field | Type | Description | Available Options |
|-------|------|-------------|-------------------|
| `type` | string | Arrangement pattern | "double_loaded_corridor", "single_element", "corner_suite" |
| `reference_element` | string | Element to position relative to | Any fixed element name |
| `room_type` | string | Type of rooms to arrange | Any room type |
| `room_count` | integer | Number of rooms | Any positive integer |
| `room_dimensions` | array | [width, length, height] | Any valid dimensions |
| `spacing` | float | Space between rooms | 0.0+ |
| `orientation` | string | How rooms are oriented | "parallel", "perpendicular" |

#### Example

```json
{
  "name": "Standard Guest Room Floor",
  "floor": 1,
  "description": "Typical floor layout with guest rooms and circulation",
  "fixed_elements": [
    {
      "type": "vertical_circulation",
      "position_type": "absolute",
      "position": [30.0, 30.0, 4.0],
      "dimensions": [8.0, 8.0, 4.0],
      "name": "Main Core",
      "required": true
    }
  ],
  "room_arrangements": [
    {
      "type": "double_loaded_corridor",
      "reference_element": "Main Corridor",
      "room_type": "guest_room",
      "room_count": 10,
      "room_dimensions": [4.0, 8.0, 3.0],
      "spacing": 0.0,
      "orientation": "perpendicular"
    }
  ]
}
```

---

## 7. Common Modifications

Here are examples of common modifications you might want to make:

### Adding a New Room Type

1. Create a room type definition in `data/room_types/`:

```json
{
  "name": "Executive Office",
  "type": "executive_office",
  "dimensions": {
    "width": 5.0,
    "length": 7.0,
    "height": 3.5
  },
  "requirements": {
    "natural_light": true,
    "exterior_access": true,
    "min_area": 35.0,
    "preferred_adjacencies": ["office"],
    "avoid_adjacencies": ["kitchen", "service_area"]
  }
}
```

2. Use it in program requirements:

```json
"executive_suite": {
  "area": 35,
  "room_type": "executive_office",
  "min_width": 5.0,
  "min_height": 3.5,
  "requires_natural_light": true,
  "requires_adjacency": ["office"]
}
```

3. Add constraints for it:

```json
{
  "type": "adjacency",
  "room_type1": "executive_office",
  "room_type2": "office",
  "weight": 1.5,
  "description": "Executive offices should be near regular offices"
}
```

### Changing Building Configuration

To create a compact version of the hotel:

```json
{
  "width": 40.0,
  "length": 50.0,
  "height": 30.0,
  "num_floors": 6,
  "floor_height": 5.0,
  "structural_grid_x": 8.0,
  "structural_grid_y": 8.0,
  "grid_size": 1.0,
  "main_entry": "south"
}
```

Run with this configuration:

```bash
python main.py --building-config compact
```

### Prioritizing Constraints

To emphasize natural light more than other factors:

1. Edit weight in exterior constraints:

```json
{
  "type": "exterior",
  "room_type": "guest_room",
  "weight": 2.5,  // Increased from 1.5
  "description": "Guest rooms should have exterior access"
}
```

2. Adjust RL weights:

```json
"weights": {
  "space_efficiency": 1.0,
  "adjacency_satisfaction": 1.0,
  "circulation_quality": 1.0,
  "natural_light": 2.0,  // Increased from 0.8
  "structural_alignment": 0.8
}
```

### Creating a Custom Program

Create a smaller hotel program:

1. Create `data/program/small_hotel.json`:

```json
{
  "public": {
    "lobby": {
      "area": 300,
      "room_type": "lobby",
      "min_width": 15.0,
      "min_height": 4.5,
      "requires_natural_light": true
    },
    "restaurant": {
      "area": 200,
      "room_type": "restaurant",
      "min_width": 12.0,
      "min_height": 4.0,
      "requires_natural_light": true
    }
  },
  // Add other departments...
}
```

2. Run with this program:

```bash
python main.py --program-config small_hotel
```

---

For any other modifications not covered here, refer to the schema documentation above or check the example files in the `data/` directory.
