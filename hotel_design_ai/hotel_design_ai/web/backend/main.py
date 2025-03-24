import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Environment variables for LLM API (replace with your preferred LLM service)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.anthropic.com/v1/messages")

# Define paths to save generated configurations
DATA_DIR = Path("./data")
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"

# Ensure directories exist
BUILDING_DIR.mkdir(parents=True, exist_ok=True)
PROGRAM_DIR.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
