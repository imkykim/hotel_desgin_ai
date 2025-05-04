"""
Main FastAPI application for Hotel Design AI web backend.
"""

import os
import sys
import json
import logging
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define project paths
PROJECT_ROOT = Path(__file__).parents[3].absolute()
sys.path.append(str(PROJECT_ROOT))

# Define data directories
DATA_DIR = PROJECT_ROOT / "data"
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"
FIX_DIR = DATA_DIR / "fix"

USER_DATA_DIR = PROJECT_ROOT / "user_data"
LAYOUTS_DIR = USER_DATA_DIR / "layouts"
VISUALIZATIONS_DIR = USER_DATA_DIR / "visualizations"

# Ensure all directories exist
for dir_path in [
    BUILDING_DIR,
    PROGRAM_DIR,
    FIX_DIR,
    USER_DATA_DIR,
    LAYOUTS_DIR,
    VISUALIZATIONS_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Import core functionality
from hotel_design_ai.core.grid_rule_engine import RuleEngine
from hotel_design_ai.core.grid_rl_engine import RLEngine
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

# Import route modules
from routes import files
from routes import visualization_routes
from routes import configuration_routes
from routes import layout_visualization_routes
from routes import chat2plan_routes
from routes import engine  # <-- Import the engine router

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

# Mount static file directories
logger.info(f"Mounting static files from: {LAYOUTS_DIR}")
app.mount("/layouts", StaticFiles(directory=str(LAYOUTS_DIR)), name="layouts")

logger.info(f"Mounting visualizations from: {VISUALIZATIONS_DIR}")
app.mount(
    "/visualizations",
    StaticFiles(directory=str(VISUALIZATIONS_DIR)),
    name="visualizations",
)

# Include router modules
app.include_router(files.router)
app.include_router(visualization_routes.router)
app.include_router(configuration_routes.router)
app.include_router(layout_visualization_routes.router)
app.include_router(chat2plan_routes.router)
app.include_router(engine.router)  # <-- Make engine endpoints available


# Pydantic models for validation
class UserInput(BaseModel):
    hotel_name: str = Field(..., description="Name of the hotel project")
    hotel_type: str = Field(..., description="Type of hotel (luxury, business, etc.)")
    num_rooms: int = Field(..., description="Number of guest rooms")
    building_width: Optional[float] = Field(
        None, description="Building width in meters"
    )
    building_length: Optional[float] = Field(
        None, description="Building length in meters"
    )
    building_height: Optional[float] = Field(
        None, description="Building height in meters"
    )
    min_floor: Optional[int] = Field(None, description="Lowest floor (e.g. -2)")
    max_floor: Optional[int] = Field(None, description="Highest floor (e.g. 20)")
    floor_height: float = Field(4.5, description="Height of each floor in meters")
    structural_grid_x: Optional[float] = Field(8.0, description="Structural grid X (m)")
    structural_grid_y: Optional[float] = Field(8.0, description="Structural grid Y (m)")
    grid_size: Optional[float] = Field(1.0, description="Grid size (m)")
    podium_min_floor: Optional[int] = Field(None, description="Podium min floor")
    podium_max_floor: Optional[int] = Field(None, description="Podium max floor")
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
    fixed_elements_file: Optional[str] = Field(
        None, description="Path to fixed elements JSON file"
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


def match_fixed_rooms_to_actual(fixed_rooms: list, rooms: list) -> dict:
    """
    Match fixed room identifiers to actual Room objects by id or name.
    Returns a dict {room.id: position}.
    """
    id_map = {room.id: room for room in rooms}
    name_map = {getattr(room, "name", None): room for room in rooms}
    result = {}
    for entry in fixed_rooms:
        # Each entry should have at least an 'id' or 'name' and a 'position'
        room_id = entry.get("id")
        room_name = entry.get("name")
        position = entry.get("position")
        matched_room = None
        if room_id is not None and room_id in id_map:
            matched_room = id_map[room_id]
        elif room_name is not None and room_name in name_map:
            matched_room = name_map[room_name]
        if matched_room is not None and position is not None:
            result[matched_room.id] = position
    return result


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Hotel Design AI API is running"}


@app.post("/generate-building-config")
async def generate_building_config(request: Dict = Body(...)):
    """Generate only building envelope configuration."""
    try:
        building_envelope = request.get("building_envelope")
        filename = request.get("filename")

        if not building_envelope:
            raise HTTPException(
                status_code=400, detail="Building envelope data is required"
            )

        if not filename:
            # Generate a default filename
            safe_name = "building"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{timestamp}.json"

        # Ensure the filename has .json extension
        if not filename.endswith(".json"):
            filename += ".json"

        # Save to file
        filepath = BUILDING_DIR / filename
        with open(filepath, "w") as f:
            json.dump(building_envelope, f, indent=2)

        logger.info(f"Building configuration saved to: {filepath}")

        return {
            "success": True,
            "building_id": filename.replace(".json", ""),
            "filename": filename,
            "path": str(filepath),
        }
    except Exception as e:
        logger.error(f"Error generating building config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating building configuration: {str(e)}"
        )


@app.post("/generate-configs")
async def generate_configs(user_input: UserInput = Body(...)):
    """Generate building envelope and hotel requirements configurations based on user input."""
    try:
        # In a real implementation, you would call the LLM API here
        # For now, we'll create a simple mock response

        # Build the building envelope dictionary using new fields
        building_envelope = {
            "width": user_input.building_width if user_input.building_width else 60.0,
            "length": (
                user_input.building_length if user_input.building_length else 80.0
            ),
            "height": (
                user_input.building_height if user_input.building_height else 100.0
            ),
            "min_floor": (
                user_input.min_floor if user_input.min_floor is not None else -2
            ),
            "max_floor": (
                user_input.max_floor if user_input.max_floor is not None else 20
            ),
            "floor_height": user_input.floor_height,
            "structural_grid_x": (
                user_input.structural_grid_x
                if user_input.structural_grid_x is not None
                else 8.0
            ),
            "structural_grid_y": (
                user_input.structural_grid_y
                if user_input.structural_grid_y is not None
                else 8.0
            ),
            "grid_size": (
                user_input.grid_size if user_input.grid_size is not None else 1.0
            ),
            "podium": {
                "min_floor": (
                    user_input.podium_min_floor
                    if user_input.podium_min_floor is not None
                    else -2
                ),
                "max_floor": (
                    user_input.podium_max_floor
                    if user_input.podium_max_floor is not None
                    else 1
                ),
                "description": "Podium section (裙房) of the building",
            },
            # ...other fields (standard_floor, etc.) can be added as needed...
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
        building_filepath = BUILDING_DIR / building_filename
        with open(building_filepath, "w") as f:
            json.dump(building_envelope, f, indent=2)

        program_filepath = PROGRAM_DIR / requirements_filename
        with open(program_filepath, "w") as f:
            json.dump(hotel_requirements, f, indent=2)

        logger.info(f"Building configuration saved to: {building_filepath}")
        logger.info(f"Program configuration saved to: {program_filepath}")

        return {
            "success": True,
            "building_envelope": {
                "filename": building_filename,
                "path": str(building_filepath),
                "data": building_envelope,
            },
            "hotel_requirements": {
                "filename": requirements_filename,
                "path": str(program_filepath),
                "data": hotel_requirements,
            },
        }
    except Exception as e:
        logger.error(f"Error generating configurations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating configurations: {str(e)}"
        )


@app.post("/update-building-config")
async def update_building_config(request: Dict = Body(...)):
    """Update an existing building configuration."""
    try:
        building_id = request.get("building_id")
        updated_config = request.get("building_config")

        if not building_id or not updated_config:
            raise HTTPException(
                status_code=400,
                detail="Building ID and configuration data are required",
            )

        # Construct the filepath
        filepath = BUILDING_DIR / f"{building_id}.json"

        logger.info(f"Updating building configuration at: {filepath}")

        # Log the standard_floor section specifically for debugging
        if "standard_floor" in updated_config:
            std_floor = updated_config["standard_floor"]
            logger.info(
                f"Standard floor parameters: width={std_floor.get('width')}, length={std_floor.get('length')}, "
                f"position_x={std_floor.get('position_x')}, position_y={std_floor.get('position_y')}"
            )
        else:
            logger.warning("No standard_floor section in updated config")

        # Full config logging at debug level
        logger.debug(f"Updated config: {updated_config}")

        # Check if the file exists
        if not filepath.exists():
            logger.warning(
                f"Building configuration {building_id} not found, creating new file"
            )

        # Make sure we have a standard_floor section to update
        if "standard_floor" not in updated_config:
            logger.warning("Adding default standard_floor section")
            updated_config["standard_floor"] = {
                "start_floor": 2,
                "end_floor": 20,
                "width": 56.0,
                "length": 20.0,
                "position_x": 0.0,
                "position_y": 32.0,
                "corridor_width": 4.0,
                "room_depth": 8.0,
            }

        # Write the updated configuration
        with open(filepath, "w") as f:
            json.dump(updated_config, f, indent=2)

        logger.info(f"Successfully updated building configuration at: {filepath}")

        return {
            "success": True,
            "message": "Building configuration updated successfully",
            "building_id": building_id,
            "filepath": str(filepath),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating building config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error updating building configuration: {str(e)}"
        )


@app.post("/save-fixed-elements")
async def save_fixed_elements(data: Dict[str, Any] = Body(...)):
    """Save fixed elements to a JSON file without modifying the reference file."""
    try:
        building_id = data.get("building_id")
        fixed_elements = data.get("fixed_elements")

        if not building_id or not fixed_elements:
            raise HTTPException(
                status_code=400,
                detail="Building ID and fixed elements data are required",
            )

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{building_id}_{timestamp}_fixed_rooms.json"

        # Save to file in the fix directory
        filepath = FIX_DIR / filename

        with open(filepath, "w") as f:
            json.dump(fixed_elements, f, indent=2)

        logger.info(f"Fixed elements saved to: {filepath}")

        # Return the filepath so the frontend can use it for layout generation
        return {
            "success": True,
            "message": "Fixed elements saved successfully",
            "filepath": str(filepath),
            "filename": filename,
        }
    except Exception as e:
        logger.error(f"Error saving fixed elements: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error saving fixed elements: {str(e)}"
        )


@app.post("/generate-layout")
async def generate_layout(input_data: DesignGenerationInput = Body(...)):
    """Generate a hotel layout based on configurations."""
    try:
        # Use the specified fixed elements file if provided
        fixed_positions = None
        if input_data.fixed_elements_file:
            # Load the fixed elements file
            try:
                fixed_file_path = Path(input_data.fixed_elements_file)
                if fixed_file_path.exists():
                    logger.info(f"Loading fixed elements from: {fixed_file_path}")
                    with open(fixed_file_path, "r") as f:
                        fixed_data = json.load(f)

                    # Extract fixed rooms data
                    if "fixed_rooms" in fixed_data:
                        # This is the enhanced format with identifiers, which will be handled
                        # by match_fixed_rooms_to_actual later in the process
                        fixed_positions = fixed_data["fixed_rooms"]
                    else:
                        # Direct format with room ID to position mapping
                        fixed_positions = {
                            int(k): tuple(v) for k, v in fixed_data.items()
                        }
                else:
                    logger.warning(f"Fixed elements file not found: {fixed_file_path}")
            except Exception as e:
                logger.error(f"Error loading fixed elements file: {e}")
                # Continue without fixed positions if there was an error
        elif input_data.fixed_positions:
            # If no file is provided but direct positions are, use those
            fixed_positions = {
                int(k): tuple(v) for k, v in input_data.fixed_positions.items()
            }

        # --- Ensure correct hotel requirements config is used ---
        program_config = input_data.program_config
        if program_config == "default":
            program_config = "hotel_requirements_3"

        # --- Load building and program configs ---
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

        # --- Always use the correct program config for room creation ---
        room_dicts = create_room_objects_from_program(program_config)
        rooms = convert_room_dicts_to_room_objects(room_dicts)

        # --- Apply fixed positions (including enhanced format) ---
        if fixed_positions:
            # If fixed_positions is a list, it's in the enhanced format with identifiers
            if isinstance(fixed_positions, list):
                fixed_positions = match_fixed_rooms_to_actual(fixed_positions, rooms)

            # Apply positions to rooms
            modified_rooms = []
            for room in rooms:
                room_copy = Room.from_dict(room.to_dict())
                if room.id in fixed_positions:
                    room_copy.position = fixed_positions[room.id]
                modified_rooms.append(room_copy)
            rooms = modified_rooms

        # --- Generate layout ---
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
