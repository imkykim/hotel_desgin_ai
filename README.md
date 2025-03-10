# Hotel Design AI

An intelligent system for automated generation and optimization of hotel architectural layouts using rule-based algorithms and reinforcement learning.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Hotel Design AI is a powerful tool that combines architectural principles with machine learning to generate optimal hotel layouts. The system takes building requirements, constraints, and room specifications as input, then automatically creates layouts that maximize efficiency, adjacency relationships, and other design metrics.

**Key Features:**
- 🏗️ Rule-based generation of architectural layouts following industry best practices
- 🧠 Reinforcement learning for layout optimization based on user feedback
- 🔍 Constraint-based evaluation framework for measuring layout quality
- 📊 3D and 2D visualization of generated designs
- 📤 Export to various formats (JSON, CSV, Revit, Three.js, etc.)
- ⚙️ Highly configurable through data-driven parameters

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hotel-design-ai.git
cd hotel-design-ai

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for advanced features
# pip install ifcopenshell pygltflib
```

## Quick Start

Generate a sample hotel layout:

```bash
# Generate and visualize a sample hotel layout
python simple_example.py --visualize

# Export to multiple formats
python simple_example.py --export-formats json,csv,threejs

# Generate a layout with RL optimization
python main.py --mode rl --visualize
```

## Architecture

The system consists of multiple components working together:

- **Spatial Grid** - 3D representation system for room placement and collision detection
- **Rule Engine** - Generates initial layouts based on architectural knowledge and constraints
- **RL Engine** - Optimizes layouts based on user feedback and design metrics
- **Constraint System** - Evaluates layouts against architectural requirements
- **Visualization** - Renders 3D and 2D visualizations of the generated layouts

## Input Requirements

To generate a hotel layout, the system needs:

1. **Building Envelope** - Overall dimensions and structural grid
2. **Room Program** - Types, dimensions, and quantities of rooms
3. **Constraints** - Adjacency requirements, floor assignments, etc.

Example configuration (from `data/building/default.json`):
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

## Usage Examples

### Generating a Layout with the Rule Engine

```python
from hotel_design_ai.core.rule_engine import RuleEngine
from hotel_design_ai.models.room import Room, RoomFactory
from hotel_design_ai.visualization.renderer import LayoutRenderer

# Define bounding box (width, length, height in meters)
bounding_box = (60.0, 80.0, 40.0)

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
    # Add more rooms as needed
]

# Generate layout
layout = rule_engine.generate_layout(rooms)

# Visualize
renderer = LayoutRenderer(layout)
renderer.save_renders(output_dir="output", prefix="hotel_layout")
```

### Using the Reinforcement Learning Engine

```python
from hotel_design_ai.core.rl_engine import RLEngine

# Initialize RL engine with the same bounding box
rl_engine = RLEngine(
    bounding_box=bounding_box,
    grid_size=1.0,
    structural_grid=(8.0, 8.0)
)

# Generate layout with RL engine
layout = rl_engine.generate_layout(rooms)

# Provide feedback for learning
user_rating = 8.5  # Scale 0-10
rl_engine.update_model(user_rating)

# Generate improved layout
improved_layout = rl_engine.generate_layout(rooms)
```

## Command Line Interface

The system provides a comprehensive command-line interface:

```bash
# Basic usage
python main.py --output ./output --mode rule --visualize

# Advanced usage with custom parameters
python main.py --building-config compact --program-config hotel_small --mode hybrid --iterations 5 --seed 42
```

Command line options:
- `--output` - Output directory for generated files
- `--mode` - Generation mode (`rule`, `rl`, or `hybrid`)
- `--visualize` - Show visualizations
- `--building-config` - Building configuration name
- `--program-config` - Program configuration name
- `--export-formats` - Export formats (comma-separated)
- `--iterations` - Number of layout iterations
- `--seed` - Random seed for reproducibility
- `--fixed-rooms` - JSON file with fixed room positions
- `--rl-model` - Path to saved RL model

## Documentation

More detailed documentation is available in the `docs` directory:

- [Design Pipeline](docs/hotel-pipeline.md) - Development and optimization workflow
- [Data Files Reference](data/data-files-documentation.md) - Configuration options
- [User Guide](docs/hotel-design-ai-guide.md) - Comprehensive usage guide

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hotel design principles based on architectural best practices
- Optimization algorithms inspired by research in automated layout generation
