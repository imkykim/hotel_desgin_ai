"""
Main FastAPI application for Hotel Design AI web backend.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from routes import files
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import httpx
import json
import logging
from pathlib import Path
import sys
import uuid
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to system path
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

# Import core functionality
from hotel_design_ai.core.rule_engine import RuleEngine
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room
from hotel_design_ai.models.layout import Layout
from hotel_design_ai.visualization.renderer import LayoutRenderer
from hotel_design_ai.config.config_loader import (
    get_building_envelope,
    get_program_requirements,
    get_adjacency_requirements,
    create_room_objects_from_program,
)
from hotel_design_ai.utils.metrics import LayoutMetrics

# Import routes
from hotel_design_ai.web.backend.routes import files

# Initialize FastAPI app
app = FastAPI(title="Hotel Design AI Configuration Generator")

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths to save generated configurations
DATA_DIR = Path("./data")
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"

PROJECT_ROOT = Path(__file__).parents[3].absolute()
USER_DATA_DIR = PROJECT_ROOT / "user_data"
LAYOUTS_DIR = USER_DATA_DIR / "layouts"


# Ensure directories exist
for dir_path in [BUILDING_DIR, PROGRAM_DIR, USER_DATA_DIR, LAYOUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Mount with explicit absolute path
print(f"Mounting static files from: {LAYOUTS_DIR}")
app.mount("/layouts", StaticFiles(directory=str(LAYOUTS_DIR)), name="layouts")
try:
    frontend_dir = PROJECT_ROOT / "hotel_design_ai" / "web" / "frontend" / "build"
    if frontend_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend"
        )
        logger.info(f"Frontend static files mounted from: {frontend_dir}")
    else:
        logger.warning(f"Frontend build directory not found at {frontend_dir}")
except Exception as e:
    logger.error(f"Failed to mount frontend static files: {e}")


# Pydantic models for validation
class UserInput(BaseModel):
    # Basic hotel information
    hotel_name: str = Field(..., description="Name of the hotel")
    hotel_type: str = Field(
        ..., description="Type of hotel (luxury, business, resort, boutique, etc.)"
    )
    num_rooms: int = Field(..., description="Total number of guest rooms", gt=0)

    # Building envelope parameters
    building_width: Optional[float] = Field(
        None, description="Width of the building in meters"
    )
    building_length: Optional[float] = Field(
        None, description="Length of the building in meters"
    )
    building_height: Optional[float] = Field(
        None, description="Height of the building in meters"
    )
    num_floors: Optional[int] = Field(
        None, description="Number of floors (excluding basement)"
    )
    num_basement_floors: Optional[int] = Field(
        None, description="Number of basement floors"
    )
    floor_height: Optional[float] = Field(
        None, description="Height of each floor in meters"
    )

    # Program requirements
    has_restaurant: bool = Field(
        True, description="Whether the hotel has restaurant facilities"
    )
    has_meeting_rooms: bool = Field(
        True, description="Whether the hotel has meeting facilities"
    )
    has_ballroom: bool = Field(False, description="Whether the hotel has a ballroom")
    has_pool: bool = Field(False, description="Whether the hotel has a swimming pool")
    has_gym: bool = Field(True, description="Whether the hotel has a fitness center")
    has_spa: bool = Field(False, description="Whether the hotel has spa facilities")

    # Custom requirements as free text
    special_requirements: Optional[str] = Field(
        None, description="Any special requirements or constraints"
    )


class DesignGenerationInput(BaseModel):
    building_config: str = Field(..., description="Building configuration name")
    program_config: str = Field(..., description="Program configuration name")
    mode: str = Field("rule", description="Generation mode (rule, rl, hybrid)")
    fixed_positions: Optional[Dict[str, List[float]]] = Field(
        None, description="Fixed room positions {room_id: [x, y, z]}"
    )
    include_standard_floors: bool = Field(
        False, description="Whether to include standard floors"
    )


class DesignModificationInput(BaseModel):
    layout_id: str = Field(..., description="ID of the layout to modify")
    room_id: int = Field(..., description="ID of the room to modify")
    new_position: List[float] = Field(..., description="New position [x, y, z]")


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


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Hotel Design AI API is running"}


@app.post("/generate-configs")
async def generate_configs(user_input: UserInput = Body(...)):
    """Generate building envelope and hotel requirements configurations based on user input."""
    try:
        # In a real implementation, you would call the LLM API here
        # For now, we'll create a simple mock response

        # Create a simplified building envelope
        building_envelope = {
            "width": user_input.building_width or 60.0,
            "length": user_input.building_length or 80.0,
            "height": user_input.building_height or 30.0,
            "min_floor": -(user_input.num_basement_floors or 1),
            "max_floor": (user_input.num_floors or 5) - 1,
            "floor_height": user_input.floor_height or 4.5,
            "structural_grid_x": 8.0,
            "structural_grid_y": 8.0,
            "grid_size": 1.0,
            "main_entry": "front",
            "description": f"{user_input.hotel_name} building configuration",
            "units": "meters",
        }

        # Create a simplified hotel requirements
        # This is a placeholder - in a real implementation, this would be generated
        # based on the user input and hotel type
        hotel_requirements = {
            "public": {
                "reception": {
                    "area": 160,
                    "room_type": "lobby",
                    "min_width": 12.0,
                    "min_height": 4.5,
                    "requires_natural_light": True,
                    "requires_adjacency": ["entrance"],
                    "floor": [0],
                }
            },
            "circulation": {
                "main_core": {
                    "area": 64,
                    "room_type": "vertical_circulation",
                    "min_width": 8.0,
                    "min_height": 20.0,
                    "requires_natural_light": False,
                    "floor": [-2, -1, 0, 1, 2],
                    "metadata": {
                        "original_name": "Main Circulation Core",
                        "is_core": True,
                    },
                }
            },
        }

        # Generate filenames based on hotel name
        safe_name = user_input.hotel_name.lower().replace(" ", "_")
        building_filename = f"{safe_name}_building.json"
        requirements_filename = f"{safe_name}_requirements.json"

        # Save to files
        with open(BUILDING_DIR / building_filename, "w") as f:
            json.dump(building_envelope, f, indent=2)

        with open(PROGRAM_DIR / requirements_filename, "w") as f:
            json.dump(hotel_requirements, f, indent=2)

        return {
            "success": True,
            "building_envelope": {
                "filename": building_filename,
                "path": str(BUILDING_DIR / building_filename),
                "data": building_envelope,
            },
            "hotel_requirements": {
                "filename": requirements_filename,
                "path": str(PROGRAM_DIR / requirements_filename),
                "data": hotel_requirements,
            },
        }
    except Exception as e:
        logger.error(f"Error generating configurations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating configurations: {str(e)}"
        )


@app.post("/generate-layout")
async def generate_layout(input_data: DesignGenerationInput = Body(...)):
    """Generate a hotel layout based on configurations."""
    try:
        # Convert string fixed positions to int keys with float value tuples
        fixed_positions = None
        if input_data.fixed_positions:
            fixed_positions = {
                int(k): tuple(v) for k, v in input_data.fixed_positions.items()
            }

        # Get building envelope parameters
        building_config = get_building_envelope(input_data.building_config)
        width = building_config["width"]
        length = building_config["length"]
        height = building_config["height"]
        grid_size = building_config["grid_size"]
        structural_grid = (
            building_config["structural_grid_x"],
            building_config["structural_grid_y"],
        )
        min_floor = building_config.get("min_floor", -1)
        floor_height = building_config.get("floor_height", 5.0)

        # Create a spatial grid
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
            building_config=building_config,
        )

        # Replace spatial grid to ensure basement support
        rule_engine.spatial_grid = spatial_grid

        # Get room dictionaries from program config
        room_dicts = create_room_objects_from_program(input_data.program_config)

        # Convert to Room objects
        rooms = convert_room_dicts_to_room_objects(room_dicts)

        # Apply fixed positions if provided
        if fixed_positions:
            modified_rooms = []
            for room in rooms:
                room_copy = Room.from_dict(room.to_dict())
                if room.id in fixed_positions:
                    room_copy.position = fixed_positions[room.id]
                modified_rooms.append(room_copy)
            rooms = modified_rooms

        # Generate layout
        layout = rule_engine.generate_layout(rooms)

        # Add standard floors if requested
        standard_rooms = []
        if input_data.include_standard_floors:
            try:
                from hotel_design_ai.core.standard_floor_generator import (
                    generate_all_standard_floors,
                )

                layout, standard_rooms = generate_all_standard_floors(
                    building_config=building_config,
                    spatial_grid=layout,
                    target_room_count=building_config.get("target_room_count", 100),
                )
                # Add standard rooms to the room list
                rooms.extend(standard_rooms)
            except ImportError:
                logger.warning(
                    "Could not import standard_floor_generator, skipping standard floors"
                )

        # Evaluate layout
        layout_model = Layout(layout)
        metrics = LayoutMetrics(layout, building_config=building_config)

        # Calculate adjacency preferences
        adjacency_requirements = get_adjacency_requirements()
        adjacency_preferences = {}
        for room_type1, room_type2 in adjacency_requirements.get(
            "required_adjacencies", []
        ):
            if room_type1 not in adjacency_preferences:
                adjacency_preferences[room_type1] = []
            adjacency_preferences[room_type1].append(room_type2)

        # Create room_departments dictionary
        room_departments = {
            room.id: room.metadata.get("department", "unknown") for room in rooms
        }

        # Evaluate metrics
        all_metrics = metrics.evaluate_all(
            adjacency_preferences=adjacency_preferences,
            room_departments=room_departments,
            structural_grid=structural_grid,
        )

        # Create a unique layout ID
        layout_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        )

        # Create output directory
        output_dir = LAYOUTS_DIR / layout_id
        output_dir.mkdir(exist_ok=True)

        # Save layout as JSON
        json_file = output_dir / "hotel_layout.json"

        # Convert layout to serializable format
        layout_dict = {
            "rooms": {
                str(room_id): {
                    "id": room_id,
                    "type": room_data["type"],
                    "position": room_data["position"],
                    "dimensions": room_data["dimensions"],
                    "metadata": room_data.get("metadata", {}),
                }
                for room_id, room_data in layout.rooms.items()
            },
            "grid_size": layout.grid_size,
            "width": layout.width,
            "length": layout.length,
            "height": layout.height,
            "metrics": all_metrics,
        }

        # Save to file
        with open(json_file, "w") as f:
            json.dump(layout_dict, f, indent=2)

        # Save visualizations
        try:
            renderer = LayoutRenderer(layout, building_config=building_config)
            renderer.save_renders(
                output_dir=str(output_dir),
                prefix="hotel_layout",
                include_3d=True,
                include_floor_plans=True,
                sample_standard=True,
            )

            # Close any open matplotlib figures
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")

        # Return success with layout data
        return {
            "success": True,
            "layout_id": layout_id,
            "rooms": layout_dict["rooms"],
            "building_dimensions": {
                "width": layout.width,
                "length": layout.length,
                "height": layout.height,
            },
            "metrics": all_metrics,
            "image_urls": {
                "3d": f"/layouts/{layout_id}/hotel_layout_3d.png",
                "floor_plans": [
                    f"/layouts/{layout_id}/hotel_layout_floor{i}.png"
                    for i in range(-2, 6)
                ],
            },
        }

    except Exception as e:
        logger.error(f"Error generating layout: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error generating layout: {str(e)}"
        )


@app.post("/modify-layout")
async def modify_layout(input_data: DesignModificationInput = Body(...)):
    """Modify a room position in an existing layout."""
    try:
        # Check if layout exists
        layout_dir = LAYOUTS_DIR / input_data.layout_id
        layout_file = layout_dir / "hotel_layout.json"

        if not layout_file.exists():
            raise HTTPException(status_code=404, detail="Layout not found")

        # Load existing layout
        with open(layout_file, "r") as f:
            layout_data = json.load(f)

        # Check if room exists
        room_id_str = str(input_data.room_id)
        if room_id_str not in layout_data["rooms"]:
            raise HTTPException(status_code=404, detail="Room not found in layout")

        # Update room position
        layout_data["rooms"][room_id_str]["position"] = input_data.new_position

        # Save modified layout
        with open(layout_file, "w") as f:
            json.dump(layout_data, f, indent=2)

        # Create modified layout filename
        modified_file = layout_dir / "modified_layout.json"
        with open(modified_file, "w") as f:
            json.dump(layout_data, f, indent=2)

        return {
            "success": True,
            "layout_id": input_data.layout_id,
            "room_id": input_data.room_id,
            "new_position": input_data.new_position,
            "modified_layout_path": str(modified_file),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying layout: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error modifying layout: {str(e)}")


# Include the routes from files.py
app.include_router(files.router)


@app.get("/list-configurations")
async def list_configurations():
    """List all available configurations."""
    try:
        return await files.list_configurations()
    except Exception as e:
        logger.error(f"Error listing configurations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing configurations: {str(e)}"
        )


@app.get("/configuration/{config_type}/{config_id}")
async def get_configuration(config_type: str, config_id: str):
    """Get a specific configuration by type and ID."""
    try:
        return await files.get_configuration(config_type, config_id)
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting configuration: {str(e)}"
        )


@app.get("/layouts/{layout_id}")
async def get_layout(layout_id: str):
    """Get a specific layout by ID - forwards to the files router implementation."""
    try:
        # Forward to the implementation in files router
        return await files.get_layout_detail(layout_id)
    except Exception as e:
        logger.error(f"Error forwarding to get_layout_detail: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting layout: {str(e)}")


@app.get("/{path:path}")
async def serve_frontend(path: str):
    """
    Catch-all route to serve the frontend application.
    This allows client-side routing to work correctly when refreshing the page or accessing URLs directly.

    In a production environment, this would typically be handled by a reverse proxy like Nginx.
    """
    # In development, we'll just return a simple message suggesting to use the client-side router
    return {
        "message": "This is a client-side route that should be handled by the frontend application.",
        "requested_path": path,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
