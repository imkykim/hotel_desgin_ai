from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import subprocess
import json
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/engine", tags=["Engine Integration"])

# Models
class StandardFloorZone(BaseModel):
    x: float
    y: float
    width: float
    height: float

class StandardFloorConfig(BaseModel):
    building_id: str
    floor_zones: List[StandardFloorZone]
    start_floor: int = Field(1, description="First standard floor")
    end_floor: int = Field(3, description="Last standard floor")

class LayoutFeedback(BaseModel):
    layout_id: str
    modified_layout: Dict[str, Any]
    user_rating: float = Field(..., ge=0, le=10, description="User rating from 0-10")
    comments: Optional[str] = None

# Define paths
DATA_DIR = Path("../data")
RL_MODELS_DIR = Path("../data/rl/models")
LAYOUTS_DIR = Path("../data/layouts")

# Ensure directories exist
RL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/standard-floor-zones")
async def set_standard_floor_zones(config: StandardFloorConfig):
    """
    Define which zones of the building should contain standard floors.
    This affects how the tower portion of the hotel is generated.
    """
    try:
        # Save the standard floor configuration
        filepath = DATA_DIR / "building" / f"{config.building_id}_std_floors.json"
        
        with open(filepath, "w") as f:
            json.dump(config.dict(), f, indent=2)
        
        return {
            "success": True,
            "message": "Standard floor zones saved successfully",
            "file": str(filepath)
        }
    except Exception as e:
        logger.error(f"Error saving standard floor zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-with-zones")
async def generate_with_zones(building_id: str = Body(...), program_id: str = Body(...)):
    """
    Generate a hotel layout using previously defined standard floor zones.
    """
    try:
        # Check if standard floor config exists
        std_floor_config = DATA_DIR / "building" / f"{building_id}_std_floors.json"
        if not std_floor_config.exists():
            raise HTTPException(status_code=404, detail="Standard floor configuration not found")
        
        # Run the main.py script with appropriate arguments
        result = subprocess.run(
            [
                "python", "../main.py",
                "--building-config", building_id,
                "--program-config", program_id,
                "--standard-floor-zones", str(std_floor_config),
                "--output", "layouts"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Generation failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {result.stderr}")
        
        # Parse the output to find the generated layout path
        layout_path = None
        for line in result.stdout.split("\n"):
            if "Outputs saved to:" in line:
                layout_path = line.split("Outputs saved to:")[1].strip()
                break
        
        if not layout_path:
            raise HTTPException(status_code=500, detail="Could not find layout path in output")
        
        # Load the layout JSON
        layout_file = Path(layout_path) / "hotel_layout.json"
        if not layout_file.exists():
            raise HTTPException(status_code=404, detail="Generated layout file not found")
        
        with open(layout_file, "r") as f:
            layout_data = json.load(f)
        
        return {
            "success": True,
            "layout_id": layout_path.split("/")[-1],
            "layout_data": layout_data,
            "layout_path": str(layout_file)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save-modified-layout")
async def save_modified_layout(layout_id: str = Body(...), layout_data: Dict[str, Any] = Body(...)):
    """
    Save a user-modified layout.
    """
    try:
        # Save the modified layout
        layout_dir = LAYOUTS_DIR / layout_id
        layout_dir.mkdir(exist_ok=True)
        
        filepath = layout_dir / "modified_layout.json"
        
        with open(filepath, "w") as f:
            json.dump(layout_data, f, indent=2)
        
        return {
            "success": True,
            "message": "Modified layout saved successfully",
            "layout_path": str(filepath)
        }
    except Exception as e:
        logger.error(f"Error saving modified layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-rl")
async def train_rl_with_feedback(feedback: LayoutFeedback):
    """
    Train the RL engine with user feedback on a layout.
    """
    try:
        # Save the feedback
        feedback_dir = LAYOUTS_DIR / feedback.layout_id / "feedback"
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / "user_feedback.json"
        
        with open(feedback_file, "w") as f:
            json.dump(feedback.dict(), f, indent=2)
        
        # Run RL training with feedback
        result = subprocess.run(
            [
                "python", "../main.py",
                "--mode", "rl",
                "--modified-layout", str(feedback_dir.parent / "modified_layout.json"),
                "--user-rating", str(feedback.user_rating),
                "--rl-model", str(RL_MODELS_DIR / "hotel_design_model.pt"),
                "--train-iterations", "10"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"RL training failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"RL training failed: {result.stderr}")
        
        return {
            "success": True,
            "message": "RL model trained successfully with user feedback",
            "model_path": str(RL_MODELS_DIR / "hotel_design_model.pt")
        }
    except Exception as e:
        logger.error(f"Error training RL model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-improved")
async def generate_improved_layout(
    building_id: str = Body(...), 
    program_id: str = Body(...),
    reference_layout_id: Optional[str] = Body(None)
):
    """
    Generate an improved layout using the trained RL model.
    Optionally use a reference layout as a starting point.
    """
    try:
        cmd = [
            "python", "../main.py",
            "--mode", "rl",
            "--building-config", building_id,
            "--program-config", program_id,
            "--rl-model", str(RL_MODELS_DIR / "hotel_design_model.pt")
        ]
        
        if reference_layout_id:
            reference_layout = LAYOUTS_DIR / reference_layout_id / "modified_layout.json"
            if reference_layout.exists():
                cmd.extend(["--reference-layout", str(reference_layout)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"RL generation failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"RL generation failed: {result.stderr}")
        
        # Parse the output to find the generated layout path
        layout_path = None
        for line in result.stdout.split("\n"):
            if "Outputs saved to:" in line:
                layout_path = line.split("Outputs saved to:")[1].strip()
                break
        
        if not layout_path:
            raise HTTPException(status_code=500, detail="Could not find layout path in output")
        
        # Load the layout JSON
        layout_file = Path(layout_path) / "hotel_layout.json"
        if not layout_file.exists():
            raise HTTPException(status_code=404, detail="Generated layout file not found")
        
        with open(layout_file, "r") as f:
            layout_data = json.load(f)
        
        return {
            "success": True,
            "layout_id": layout_path.split("/")[-1],
            "layout_data": layout_data,
            "layout_path": str(layout_file),
            "message": "Improved layout generated using RL model"
        }
    except Exception as e:
        logger.error(f"Error generating improved layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))
