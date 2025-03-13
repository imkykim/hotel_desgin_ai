# Hotel Program Requirements Input Guide

This guide explains how to format and customize hotel program requirements for the Hotel Design AI system.

## Basic Structure

The program requirements should be provided in JSON format with this high-level structure:

```json
{
  "department_name": {
    "space_name": {
      // Space specifications
    },
    "another_space": {
      // Space specifications
    }
  },
  "another_department": {
    // More spaces
  }
}
```

## Space Specification Reference

Each space in your program needs the following properties:

### Required Properties

| Property | Type | Description |
|----------|------|-------------|
| `area` | float | Total area in square meters |
| `room_type` | string | Type of room (see Room Types section) |

### Common Optional Properties

| Property | Type | Description |
|----------|------|-------------|
| `details` | object | Sub-spaces with areas that add up to the total area |
| `min_width` | float | Minimum width constraint in meters |
| `min_length` | float | Minimum length constraint in meters |
| `min_height` | float | Minimum height constraint in meters |
| `floor` | integer | Preferred floor (0=ground, -1=basement, 1+=upper floors) |
| `requires_natural_light` | boolean | Whether the space needs exterior access |
| `requires_adjacency` | array | List of room types that should be adjacent |
| `requires_separation` | array | List of room types that should be separated |

### Extended Properties

| Property | Type | Description |
|----------|------|-------------|
| `requires_ventilation` | boolean | Whether the space needs mechanical ventilation |
| `requires_exhaust` | boolean | Whether the space needs direct exhaust systems |
| `can_be_divided` | boolean | Whether the space should be divisible with partitions |
| `requires_column_free` | boolean | Whether the space needs to be free of structural columns |
| `capacity` | integer | Number of people/units the space should accommodate |

## Example Space Definition

```json
"ballroom": {
  "area": 500,
  "details": {
    "main_hall": 400,
    "stage": 70,
    "av_control": 30
  },
  "room_type": "assembly",
  "min_width": 15.0,
  "min_height": 6.0,
  "requires_natural_light": false,
  "can_be_divided": true,
  "requires_column_free": true,
  "floor": 0,
  "requires_adjacency": ["pre_function", "back_of_house"]
}
```

## Room Types Reference

These are the standard room types supported by the system:

### Public Areas
- `entrance`: Main building entrances
- `lobby`: Main hotel lobby and reception areas
- `retail`: Shops, stores, and retail outlets
- `circulation`: Corridors, hallways, and general circulation

### Dining and Food Service
- `restaurant`: Dining spaces and restaurants
- `kitchen`: Food preparation areas
- `food_service`: Bars, cafes, and other food service areas

### Meeting and Events
- `ballroom`: Large event spaces
- `assembly`: Divisible event spaces and halls
- `meeting_room`: Conference and meeting spaces
- `pre_function`: Space outside ballroom/meeting rooms

### Guest Rooms
- `guest_room`: Standard hotel rooms for guests
- `lounge`: Seating and waiting areas

### Back of House
- `service_area`: Back of house service spaces
- `back_of_house`: Staff and service areas
- `staff_area`: Staff facilities and areas
- `maintenance`: Maintenance workshops and storage 
- `workshop`: Specialized workshops for repair and fabrication
- `mechanical`: Equipment and mechanical spaces
- `office`: Administrative offices

### Recreation
- `pool`: Swimming pool area
- `fitness`: Exercise and gym area
- `entertainment`: Entertainment venues (KTV, game rooms, etc.)

### Transportation
- `parking`: Car parking areas

## Floor Assignments

For optimal layout results, specify floor assignments for these critical spaces:

- Ground Floor (0)
  - Entrance
  - Lobby/Reception
  - Retail
  - Restaurants
  - Ballroom/Assembly spaces

- Upper Floors (1+)
  - Guest rooms
  - Meeting rooms
  - Offices

- Basement (-1)
  - Parking
  - Mechanical
  - Back of house
  - Kitchen (if not on ground floor)

## Department Organization

Organize your spaces into logical departments:

1. Public Areas
2. Guest Rooms
3. Food & Beverage
4. Meeting & Events
5. Recreation & Wellness
6. Back of House
7. Administration
8. Engineering

## Tips for Optimal Results

1. **Be specific with adjacencies**: List the most critical adjacency relationships.
2. **Specify floors**: Always specify preferred floor levels for key spaces.
3. **Include details**: Break down large spaces into sub-spaces with the `details` property.
4. **Respect proportions**: For large spaces, ensure min_width and min_height are appropriate.
5. **Use standard room types**: Stick to standard room types where possible.

## Example of a Complete Department

```json
"dining": {
  "restaurants": {
    "area": 1300,
    "details": {
      "chinese_restaurant_main": 300,
      "chinese_private_rooms": 200,
      "western_restaurant": 200,
      "all_day_dining": 300,
      "coffee_shop": 200,
      "bar": 100
    },
    "room_type": "restaurant",
    "min_width": 10.0,
    "min_height": 4.0,
    "requires_natural_light": true,
    "floor": 0,
    "requires_adjacency": ["kitchen", "lobby"]
  },
  "kitchen": {
    "area": 850,
    "details": {
      "chinese_kitchen": 500,
      "western_kitchen": 150,
      "all_day_dining_kitchen": 200
    },
    "room_type": "kitchen",
    "min_width": 8.0,
    "min_height": 4.0,
    "requires_exhaust": true,
    "requires_ventilation": true,
    "floor": 0,
    "requires_adjacency": ["restaurants", "service_elevators"]
  }
}
```

## Handling Room Type Mapping

If you have custom room types, map them to standard types using this reference:

| Custom Type    | Standard Type        |
|----------------|----------------------|
| "reception"    | "lobby"              |
| "assembly"     | "ballroom"           |
| "meeting"      | "meeting_room"       |
| "staff"        | "staff_area"         |
| "workshop"     | "maintenance"        |
