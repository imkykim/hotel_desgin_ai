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


# Define paths - going up 4 levels from here to get to project root
PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data"
USER_DATA_DIR = PROJECT_ROOT / "user_data"
RL_MODELS_DIR = USER_DATA_DIR / "models"
LAYOUTS_DIR = USER_DATA_DIR / "layouts"
FEEDBACK_DIR = USER_DATA_DIR / "feedback"

# Ensure directories exist
for dir_path in [DATA_DIR, USER_DATA_DIR, RL_MODELS_DIR, LAYOUTS_DIR, FEEDBACK_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Helper functions
def success_response(message: str, **kwargs) -> Dict[str, Any]:
    """Create a standardized success response."""
    return {"success": True, "message": message, **kwargs}


def run_command(command: List[str], error_message: str = "Command execution failed"):
    """Run a subprocess command with standardized error handling."""
    try:
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"{error_message}: {result.stderr}")
            raise Exception(f"{error_message}: {result.stderr}")

        return result
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        raise e


def extract_layout_path(command_output: str) -> str:
    """Extract the layout path from command output."""
    for line in command_output.split("\n"):
        if "Outputs saved to:" in line:
            return line.split("Outputs saved to:")[1].strip()

    raise Exception("Could not find layout path in output")


def save_json_file(data: Dict, filepath: Path) -> str:
    """Save JSON data to file with proper directory creation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def load_json_file(filepath: Path) -> Dict:
    """Load JSON data from file with error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


@router.post("/standard-floor-zones")
async def set_standard_floor_zones(config: StandardFloorConfig):
    """
    Define which zones of the building should contain standard floors.
    This affects how the tower portion of the hotel is generated.
    """
    try:
        # Save the standard floor configuration
        filepath = DATA_DIR / "building" / f"{config.building_id}_std_floors.json"
        save_json_file(config.dict(), filepath)

        return success_response(
            "Standard floor zones saved successfully", file=str(filepath)
        )
    except Exception as e:
        logger.error(f"Error saving standard floor zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-with-zones")
async def generate_with_zones(
    building_id: str = Body(...), program_id: str = Body(...)
):
    """
    Generate a hotel layout using previously defined standard floor zones.
    """
    try:
        # Check if standard floor config exists
        std_floor_config = DATA_DIR / "building" / f"{building_id}_std_floors.json"
        if not std_floor_config.exists():
            # Fallback: try to extract from building config
            building_config_file = DATA_DIR / "building" / f"{building_id}.json"
            if building_config_file.exists():
                with open(building_config_file, "r") as f:
                    building_config = json.load(f)
                std = building_config.get("standard_floor")
                if std:
                    # Convert standard_floor to zones format and save
                    zone = {
                        "x": std.get("position_x", 0),
                        "y": std.get("position_y", 0),
                        "width": std.get("width", 0),
                        "height": std.get("length", 0),
                    }
                    zones_data = {
                        "building_id": building_id,
                        "floor_zones": [zone],
                        "start_floor": std.get("start_floor", 2),
                        "end_floor": std.get("end_floor", 20),
                    }
                    with open(std_floor_config, "w") as f:
                        json.dump(zones_data, f, indent=2)
                    logger.info(f"Generated {std_floor_config} from building config")
                else:
                    raise HTTPException(
                        status_code=404, detail="Standard floor configuration not found"
                    )
            else:
                raise HTTPException(
                    status_code=404, detail="Standard floor configuration not found"
                )

        # Run the main.py script with appropriate arguments
        command = [
            "python",
            "../../../main.py",  # <-- updated relative path
            "--building-config",
            building_id,
            "--program-config",
            program_id,
            "--standard-floor-zones",
            str(std_floor_config),
            "--output",
            "layouts",
        ]

        result = run_command(command, "Generation failed")
        layout_path = extract_layout_path(result.stdout)

        # Load the layout JSON
        layout_file = Path(layout_path) / "hotel_layout.json"
        if not layout_file.exists():
            raise HTTPException(
                status_code=404, detail="Generated layout file not found"
            )

        layout_data = load_json_file(layout_file)

        return success_response(
            "Layout generated successfully",
            layout_id=layout_path.split("/")[-1],
            layout_data=layout_data,
            layout_path=str(layout_file),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-modified-layout")
async def save_modified_layout(
    layout_id: str = Body(...), layout_data: Dict[str, Any] = Body(...)
):
    """
    Save a user-modified layout.
    """
    try:
        # Save the modified layout
        layout_dir = LAYOUTS_DIR / layout_id
        filepath = layout_dir / "modified_layout.json"
        save_json_file(layout_data, filepath)

        return success_response(
            "Modified layout saved successfully", layout_path=str(filepath)
        )
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
        save_json_file(feedback.dict(), feedback_file)

        # Run RL training with feedback
        command = [
            "python",
            "../../../main.py",  # <-- fix path to main.py (was "../main.py")
            "--mode",
            "rl",
            "--modified-layout",
            str(feedback_dir.parent / "modified_layout.json"),
            "--user-rating",
            str(feedback.user_rating),
            "--rl-model",
            str(RL_MODELS_DIR / "hotel_design_model.pt"),
            "--train-iterations",
            "10",
        ]

        run_command(command, "RL training failed")

        return success_response(
            "RL model trained successfully with user feedback",
            model_path=str(RL_MODELS_DIR / "hotel_design_model.pt"),
        )
    except Exception as e:
        logger.error(f"Error training RL model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-improved")
async def generate_improved_layout(
    building_id: str = Body(...),
    program_id: str = Body(...),
    reference_layout_id: Optional[str] = Body(None),
):
    """
    Generate an improved layout using the trained RL model.
    Optionally use a reference layout as a starting point.
    """
    try:
        # Build command for generating an improved layout
        command = [
            "python",
            "../../../main.py",  # <-- fix path to main.py (was "../main.py")
            "--mode",
            "rl",
            "--building-config",
            building_id,
            "--program-config",
            program_id,
            "--rl-model",
            str(RL_MODELS_DIR / "hotel_design_model.pt"),
        ]

        # Add reference layout if provided
        if reference_layout_id:
            reference_layout = (
                LAYOUTS_DIR / reference_layout_id / "modified_layout.json"
            )
            if reference_layout.exists():
                command.extend(["--reference-layout", str(reference_layout)])

        # Run the command
        result = run_command(command, "RL generation failed")
        layout_path = extract_layout_path(result.stdout)

        # Load the layout JSON
        layout_file = Path(layout_path) / "hotel_layout.json"
        if not layout_file.exists():
            raise HTTPException(
                status_code=404, detail="Generated layout file not found"
            )

        layout_data = load_json_file(layout_file)

        return success_response(
            "Improved layout generated using RL model",
            layout_id=layout_path.split("/")[-1],
            layout_data=layout_data,
            layout_path=str(layout_file),
        )
    except Exception as e:
        logger.error(f"Error generating improved layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))
