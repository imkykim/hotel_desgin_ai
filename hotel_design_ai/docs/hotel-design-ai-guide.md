# Hotel Design AI - Project Guide

This guide provides instructions for using the Hotel Design AI system, which automates the generation of hotel architectural layouts based on constraints and user feedback.

## Project Overview

The Hotel Design AI system is a Python-based application that combines rule-based architectural principles with reinforcement learning to generate optimized hotel layouts. It provides:

1. Automated generation of initial hotel layouts based on architectural best practices
2. Interactive refinement through user feedback
3. 3D visualization of generated designs
4. Exports to various industry formats

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hotel-design-ai.git
   cd hotel-design-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install optional dependencies for advanced features:
   ```
   pip install ifcopenshell pygltflib
   ```
   Note: ifcopenshell installation may be complex on some platforms.

## Quick Start

Try the sample hotel layout generator:

```
python sample_hotel_layout.py --output ./renders
```

This will generate a sample hotel design, visualize it, and save the results to the `./renders` directory.

## Required Input Information

To use the system effectively, you'll need to provide:

### 1. Building Envelope Information

- **Bounding box dimensions** (width, length, height in meters)
- **Structural grid spacing** (typical column spacing in x and y directions)
- **Grid size** for spatial representation (smaller = more detailed but slower)

### 2. Room Requirements

For each functional space/room:

- **Dimensions** (width, length, height in meters)
- **Room type** (predefined types include: entrance, lobby, guest_room, restaurant, meeting_room, vertical_circulation, service_area, back_of_house)
- **Special requirements:**
  - Natural light requirements
  - Exterior access needs
  - Preferred floor level

### 3. Adjacency Constraints

Specify room relationships:

- **Preferred adjacencies** (which rooms should be next to each other)
- **Avoid adjacencies** (which rooms should not be adjacent)

### 4. Additional Constraints (Optional)

- **Fixed positions** for specific rooms
- **Entrance location** preferences
- **Circulation requirements**

## Example Usage

Here's a minimal example showing how to set up and use the system:

```python
from hotel_design_ai.core.rule_engine import RuleEngine
from hotel_design_ai.models.room import Room, RoomFactory
from hotel_design_ai.visualization.renderer import LayoutRenderer

# Define bounding box
bounding_box = (60.0, 40.0, 30.0)  # width, length, height in meters

# Initialize rule engine
rule_engine = RuleEngine(
    bounding_box=bounding_box,
    grid_size=1.0,
    structural_grid=(8.0, 8.0)
)

# Create rooms
rooms = [
    RoomFactory.create_entrance(),
    RoomFactory.create_lobby(),
    RoomFactory.create_restaurant(),
    RoomFactory.create_vertical_circulation(),
    RoomFactory.create_guest_room(),
    # Add more rooms as needed
]

# Generate layout
layout = rule_engine.generate_layout(rooms)

# Visualize
renderer = LayoutRenderer(layout)
fig, ax = renderer.render_3d()
fig.savefig("hotel_layout_3d.png")
```

## Interactive Design Process

1. **Generate Initial Layout**: Start with the rule-based engine to create a baseline design
2. **User Feedback**: Examine and provide ratings or modify room positions
3. **Regenerate**: The system learns from feedback and generates improved layouts
4. **Iterate**: Continue the feedback loop until satisfied with the design

## Web API

The system includes a FastAPI-based web API for integration with frontend applications:

1. Start the API server:
   ```
   uvicorn api.main:app --reload
   ```

2. API endpoints:
   - `POST /api/sessions` - Create a new design session
   - `GET /api/sessions/{session_id}` - Get session details
   - `POST /api/sessions/{session_id}/update` - Update with user modifications
   - `POST /api/sessions/{session_id}/feedback` - Provide feedback for RL training

## Visualization and Export

The system supports various visualization and export options:

- 3D renderings of the complete layout
- 2D floor plans for each level
- Exports to:
  - JSON (for web applications)
  - CSV (for data analysis)
  - OBJ (for 3D modeling)
  - SketchUp-compatible format
  - Revit-compatible format
  - IFC (with optional dependencies)

## Advancing the System

As you work with the system:

1. **Training Data**: Each user interaction improves the RL agent's performance
2. **Custom Constraints**: Add domain-specific constraints by extending the core systems
3. **Visualization**: Enhance visualization with sector-specific requirements
4. **Integration**: Connect with other architectural and engineering tools

## Troubleshooting

Common issues:

- **Memory errors**: Reduce grid size or bounding box dimensions
- **Placement failures**: Check room dimensions and constraints compatibility
- **Visualization errors**: Ensure matplotlib and numpy are correctly installed

## Getting Help

If you need assistance:
- Check documentation in the `/docs` directory
- File issues on the GitHub repository
- Reach out via [contact information]
