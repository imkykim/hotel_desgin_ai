import os
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
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
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

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

# Environment variables for LLM API
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.anthropic.com/v1/messages")

# Define paths to save generated configurations
DATA_DIR = Path("./data")
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"
USER_DATA_DIR = Path("./user_data")
LAYOUTS_DIR = USER_DATA_DIR / "layouts"

# Ensure directories exist
for dir_path in [BUILDING_DIR, PROGRAM_DIR, USER_DATA_DIR, LAYOUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


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


# Function to call the LLM API
async def call_llm_api(prompt: str) -> str:
    """Call the LLM API with the given prompt."""
    try:
        # Using Anthropic Claude API as an example
        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}],
            }

            response = await client.post(LLM_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            # Extract text from Claude response
            return response_data["content"][0]["text"]
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")


def format_building_envelope_prompt(user_input: UserInput) -> str:
    """Format the prompt for generating building envelope configuration."""
    prompt = f"""
    Generate a building envelope configuration JSON for a hotel design AI system based on the following requirements:
    
    Hotel Name: {user_input.hotel_name}
    Hotel Type: {user_input.hotel_type}
    Number of Guest Rooms: {user_input.num_rooms}
    
    Building Width: {user_input.building_width if user_input.building_width else "Calculate appropriate width"}
    Building Length: {user_input.building_length if user_input.building_length else "Calculate appropriate length"}
    Building Height: {user_input.building_height if user_input.building_height else "Calculate appropriate height"}
    Number of Floors: {user_input.num_floors if user_input.num_floors else "Calculate appropriate number"}
    Number of Basement Floors: {user_input.num_basement_floors if user_input.num_basement_floors else "Calculate appropriate number"}
    Floor Height: {user_input.floor_height if user_input.floor_height else "Use standard height (e.g., 4.0-5.0 meters)"}
    
    Special Requirements: {user_input.special_requirements if user_input.special_requirements else "None"}

    The JSON should follow this format:
    {{
      "width": [width in meters],
      "length": [length in meters],
      "height": [height in meters],
      "min_floor": [lowest floor number, e.g., -2 for two basement levels],
      "max_floor": [highest floor number, e.g., 5 for a 6-story building (0-5)],
      "floor_height": [height of each floor in meters],
      "structural_grid_x": [structural grid spacing in x-direction, typically 8.0-9.0 meters],
      "structural_grid_y": [structural grid spacing in y-direction, typically 8.0-9.0 meters],
      "grid_size": [calculation grid size, typically 1.0 meter],
      "main_entry": [main entry location, e.g., "front", "side", or "flexible"],
      "description": [brief description of the building envelope],
      "units": "meters"
    }}
    
    Calculate appropriate values based on industry standards for a {user_input.hotel_type} hotel with {user_input.num_rooms} rooms.
    If specific dimensions weren't provided, make reasonable assumptions based on hotel type and number of rooms.
    Use a structural grid that makes sense for the hotel type (luxury hotels might need larger spans).
    
    Return ONLY the valid JSON object without any explanations, markdown formatting, or code blocks.
    """
    return prompt


def format_hotel_requirements_prompt(user_input: UserInput) -> str:
    """Format the prompt for generating hotel requirements configuration."""
    # Calculate some basic parameters based on hotel type and size
    guest_room_area = 32  # Default square meters per room

    if user_input.hotel_type.lower() in ["luxury", "resort"]:
        guest_room_area = 45
    elif user_input.hotel_type.lower() in ["boutique", "upscale"]:
        guest_room_area = 38
    elif user_input.hotel_type.lower() in ["budget", "economy"]:
        guest_room_area = 26

    total_guest_room_area = user_input.num_rooms * guest_room_area

    # Typical ratios for different hotel components
    public_ratio = 0.10
    dining_ratio = 0.15 if user_input.has_restaurant else 0.05
    meeting_ratio = 0.20 if user_input.has_meeting_rooms else 0.05
    recreational_ratio = (
        0.15
        if (user_input.has_pool or user_input.has_gym or user_input.has_spa)
        else 0.05
    )

    # Calculate approximate areas based on ratios
    public_area = int(total_guest_room_area * public_ratio)
    dining_area = int(total_guest_room_area * dining_ratio)
    meeting_area = int(total_guest_room_area * meeting_ratio)
    recreational_area = int(total_guest_room_area * recreational_ratio)

    prompt = f"""
    Generate a hotel program requirements JSON for a hotel design AI system based on the following inputs:
    
    Hotel Name: {user_input.hotel_name}
    Hotel Type: {user_input.hotel_type}
    Number of Guest Rooms: {user_input.num_rooms}
    
    Facilities:
    - Restaurant: {"Yes" if user_input.has_restaurant else "No"}
    - Meeting Rooms: {"Yes" if user_input.has_meeting_rooms else "No"}
    - Ballroom: {"Yes" if user_input.has_ballroom else "No"}
    - Swimming Pool: {"Yes" if user_input.has_pool else "No"}
    - Fitness Center: {"Yes" if user_input.has_gym else "No"}
    - Spa: {"Yes" if user_input.has_spa else "No"}
    
    Estimated Areas:
    - Total Guest Room Area: ~{total_guest_room_area} m²
    - Public Areas: ~{public_area} m²
    - Dining Areas: ~{dining_area} m²
    - Meeting Areas: ~{meeting_area} m²
    - Recreational Areas: ~{recreational_area} m²
    
    Special Requirements: {user_input.special_requirements if user_input.special_requirements else "None"}

    The JSON should follow the format of hotel_requirements.json with these main sections:
    - public (reception, retail, service_areas)
    - dining (kitchen, restaurants, other facilities)
    - meeting (grand_ballroom if applicable, meeting rooms)
    - recreational (swimming_pool if applicable, gym if applicable)
    - administrative (offices, staff facilities)
    - engineering (maintenance, equipment_rooms)
    - circulation (main_core, secondary_core)
    - parking (underground_parking)

    Each area should include:
    - area (in square meters)
    - details (sub-spaces and their areas)
    - room_type (e.g., "lobby", "kitchen", "meeting_room", etc.)
    - min_width and min_height
    - specific requirements like requires_natural_light, requires_adjacency, etc.
    - preferred floor(s) as an array (e.g., [0] for ground floor or [-1, -2] for basement)
    
    Scale the areas appropriately for a {user_input.hotel_type} hotel with {user_input.num_rooms} rooms.
    Skip facilities that the hotel doesn't have, but include all necessary support spaces.
    
    Return ONLY the valid JSON object without any explanations, markdown formatting, or code blocks.
    """
    return prompt


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
    building_config_name: str,
    program_config_name: str,
    fixed_positions: Optional[Dict[int, Any]] = None,
    include_standard_floors: bool = False,
) -> Tuple[SpatialGrid, List[Room]]:
    """Generate a layout using the rule-based engine"""
    logger.info("Generating layout using rule-based engine...")

    # Get building envelope parameters
    building_config = get_building_envelope(building_config_name)
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

    # Create a spatial grid that properly supports basements
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

    # Replace the spatial grid to ensure basement support
    rule_engine.spatial_grid = spatial_grid

    # Get room dictionaries from program config
    room_dicts = create_room_objects_from_program(program_config_name)

    # Convert to Room objects
    rooms = convert_room_dicts_to_room_objects(room_dicts)

    # Apply fixed positions if provided
    if fixed_positions:
        # Create a copy of rooms to avoid modifying the original list
        modified_rooms = []

        for room in rooms:
            room_copy = Room.from_dict(room.to_dict())
            if room.id in fixed_positions:
                room_copy.position = tuple(fixed_positions[room.id])
            modified_rooms.append(room_copy)

        rooms = modified_rooms

    # Generate layout
    layout = rule_engine.generate_layout(rooms)
    logger.info(f"Layout generated with {len(layout.rooms)} rooms")

    # Add standard floors if requested
    if include_standard_floors:
        from hotel_design_ai.core.standard_floor_generator import (
            generate_all_standard_floors,
            find_vertical_circulation_core,
        )

        # Find circulation core to extend to standard floors
        circulation_core = find_vertical_circulation_core(layout, building_config)
        logger.info("Generating standard floors...")

        # Generate all standard floors
        layout, standard_rooms = generate_all_standard_floors(
            building_config=building_config,
            spatial_grid=layout,
            target_room_count=building_config.get("target_room_count", 100),
        )

        # Add standard rooms to the room list
        rooms.extend(standard_rooms)
        logger.info(f"Added {len(standard_rooms)} rooms in standard floors")

    return layout, rooms


def evaluate_layout(
    layout: SpatialGrid, rooms: List[Room], building_config_name: str
) -> Dict[str, Any]:
    """Evaluate a layout using various metrics"""
    logger.info("Evaluating layout...")

    # Create Layout model wrapper
    layout_model = Layout(layout)

    # Get building parameters for metrics
    building_config = get_building_envelope(building_config_name)

    # Create metrics calculator
    metrics = LayoutMetrics(layout, building_config=building_config)

    # Calculate metrics
    space_utilization = metrics.space_utilization() * 100
    logger.info(f"Space utilization: {space_utilization:.1f}%")

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

    # Get structural grid for metrics
    structural_grid = (
        building_config["structural_grid_x"],
        building_config["structural_grid_y"],
    )

    # Evaluate all metrics
    all_metrics = metrics.evaluate_all(
        adjacency_preferences=adjacency_preferences,
        room_departments=room_departments,
        structural_grid=structural_grid,
    )

    # Log key metrics
    logger.info(f"Overall score: {all_metrics['overall_score'] * 100:.1f}%")

    # Return metrics for presentation to user
    return all_metrics


def save_layout(layout: SpatialGrid, metrics: Dict[str, Any]) -> str:
    """Save layout to JSON file and return the layout ID"""
    # Create a unique layout ID
    layout_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

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
        "metrics": metrics,
    }

    # Save to file
    with open(json_file, "w") as f:
        json.dump(layout_dict, f, indent=2)

    # Save visualizations if matplotlib is available
    try:
        from hotel_design_ai.visualization.renderer import LayoutRenderer

        # Get building parameters
        building_config = get_building_envelope("default")

        # Create renderer
        renderer = LayoutRenderer(layout, building_config=building_config)

        # Save renders to output directory
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

    return layout_id


@app.post("/generate-configs")
async def generate_configs(user_input: UserInput = Body(...)):
    """Generate building envelope and hotel requirements configurations based on user input."""
    try:
        # Generate building envelope configuration
        building_prompt = format_building_envelope_prompt(user_input)
        building_envelope_text = await call_llm_api(building_prompt)

        # Remove any markdown formatting if present
        building_envelope_text = building_envelope_text.strip()
        if building_envelope_text.startswith("```json"):
            building_envelope_text = building_envelope_text.replace("```json", "", 1)
        if building_envelope_text.endswith("```"):
            building_envelope_text = building_envelope_text[:-3]
        building_envelope_text = building_envelope_text.strip()

        # Parse JSON to validate it
        building_envelope_json = json.loads(building_envelope_text)

        # Generate hotel requirements configuration
        hotel_prompt = format_hotel_requirements_prompt(user_input)
        hotel_requirements_text = await call_llm_api(hotel_prompt)

        # Remove any markdown formatting if present
        hotel_requirements_text = hotel_requirements_text.strip()
        if hotel_requirements_text.startswith("```json"):
            hotel_requirements_text = hotel_requirements_text.replace("```json", "", 1)
        if hotel_requirements_text.endswith("```"):
            hotel_requirements_text = hotel_requirements_text[:-3]
        hotel_requirements_text = hotel_requirements_text.strip()

        # Parse JSON to validate it
        hotel_requirements_json = json.loads(hotel_requirements_text)

        # Generate filenames based on hotel name
        safe_name = user_input.hotel_name.lower().replace(" ", "_")
        building_filename = f"{safe_name}_building.json"
        requirements_filename = f"{safe_name}_requirements.json"

        # Save to files
        with open(BUILDING_DIR / building_filename, "w") as f:
            json.dump(building_envelope_json, f, indent=2)

        with open(PROGRAM_DIR / requirements_filename, "w") as f:
            json.dump(hotel_requirements_json, f, indent=2)

        return {
            "success": True,
            "building_envelope": {
                "filename": building_filename,
                "path": str(BUILDING_DIR / building_filename),
                "data": building_envelope_json,
            },
            "hotel_requirements": {
                "filename": requirements_filename,
                "path": str(PROGRAM_DIR / requirements_filename),
                "data": hotel_requirements_json,
            },
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid JSON generated: {str(e)}")
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

        # Generate layout using rule-based engine
        layout, rooms = generate_rule_based_layout(
            input_data.building_config,
            input_data.program_config,
            fixed_positions,
            input_data.include_standard_floors,
        )

        # Evaluate layout
        metrics = evaluate_layout(layout, rooms, input_data.building_config)

        # Save layout to disk
        layout_id = save_layout(layout, metrics)

        # Prepare room data for response
        rooms_data = {
            str(room_id): {
                "id": room_id,
                "type": room_data["type"],
                "position": room_data["position"],
                "dimensions": room_data["dimensions"],
                "metadata": room_data.get("metadata", {}),
            }
            for room_id, room_data in layout.rooms.items()
        }

        # Return success with layout data
        return {
            "success": True,
            "layout_id": layout_id,
            "rooms": rooms_data,
            "building_dimensions": {
                "width": layout.width,
                "length": layout.length,
                "height": layout.height,
            },
            "metrics": metrics,
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


@app.get("/layouts/{layout_id}")
async def get_layout(layout_id: str):
    """Get a specific layout by ID."""
    try:
        # Check if layout exists
        layout_dir = LAYOUTS_DIR / layout_id
        layout_file = layout_dir / "hotel_layout.json"

        if not layout_file.exists():
            raise HTTPException(status_code=404, detail="Layout not found")

        # Load layout
        with open(layout_file, "r") as f:
            layout_data = json.load(f)

        # Return layout data
        return {
            "success": True,
            "layout_id": layout_id,
            "layout_data": layout_data,
            "image_urls": {
                "3d": f"/layouts/{layout_id}/hotel_layout_3d.png",
                "floor_plans": [
                    f"/layouts/{layout_id}/hotel_layout_floor{i}.png"
                    for i in range(-2, 6)
                    if (layout_dir / f"hotel_layout_floor{i}.png").exists()
                ],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting layout: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting layout: {str(e)}")


@app.get("/layouts")
async def list_layouts():
    """List all available layouts."""
    try:
        layouts = []

        # Get all layout directories
        for layout_dir in LAYOUTS_DIR.iterdir():
            if layout_dir.is_dir():
                layout_file = layout_dir / "hotel_layout.json"

                if layout_file.exists():
                    # Load basic layout info
                    with open(layout_file, "r") as f:
                        layout_data = json.load(f)

                    # Add to list
                    layouts.append(
                        {
                            "id": layout_dir.name,
                            "room_count": len(layout_data["rooms"]),
                            "metrics": layout_data.get("metrics", {}),
                            "created_at": (
                                layout_dir.name.split("_")[0]
                                if "_" in layout_dir.name
                                else ""
                            ),
                            "thumbnail": f"/layouts/{layout_dir.name}/hotel_layout_3d.png",
                        }
                    )

        return {"success": True, "layouts": layouts}

    except Exception as e:
        logger.error(f"Error listing layouts: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error listing layouts: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
