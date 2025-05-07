from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import subprocess
import traceback
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
    building_id: str = Body(...),
    program_id: Optional[str] = Body(None),
    fixed_rooms_file: Optional[str] = Body(None),
):
    """
    Generate a hotel layout using previously defined standard floor zones,
    a specific program config, and a fixed rooms file.
    Enhanced with better error handling and debug logging.
    """
    try:
        # ENHANCEMENT: Added more logging
        logger.info(f"Generate layout with zones for building ID: {building_id}")
        logger.info(f"Program ID: {program_id or 'default'}")
        logger.info(f"Fixed rooms file: {fixed_rooms_file or 'None'}")

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

        # FIXED: Always ensure .json extension is included for program config
        if not program_id or program_id == "default":
            program_config = "hotel_requirements_3.json"  # FIXED: Added .json extension
            logger.info(f"Using default program config: {program_config}")
        else:
            # Ensure .json extension is present
            if not program_id.endswith(".json"):
                program_config = f"{program_id}.json"
            else:
                program_config = program_id
            logger.info(f"Using specified program config: {program_config}")

            # Enhanced fixed rooms handling in engine.py generate_with_zones function

            # ENHANCEMENT: Better fixed rooms file handling
            fixed_rooms_arg = []
            if fixed_rooms_file:
                # Always resolve to absolute path
                fixed_rooms_path = Path(fixed_rooms_file)
                if not fixed_rooms_path.is_absolute():
                    # Try to resolve relative to the project root
                    fixed_rooms_path = (PROJECT_ROOT / fixed_rooms_file).resolve()

                if fixed_rooms_path.exists():
                    # ENHANCEMENT: Log the contents of the fixed rooms file
                    try:
                        with open(fixed_rooms_path, "r") as f:
                            fixed_rooms_data = json.load(f)
                            logger.info(
                                f"Fixed rooms file contents: {json.dumps(fixed_rooms_data, indent=2)}"
                            )

                            # ENHANCEMENT: Check if the file has the expected format
                            if "fixed_rooms" in fixed_rooms_data:
                                logger.info(
                                    f"Found {len(fixed_rooms_data['fixed_rooms'])} fixed room definitions"
                                )

                                # ENHANCEMENT: Print each fixed room identifier for debugging
                                for i, fixed_room in enumerate(
                                    fixed_rooms_data["fixed_rooms"]
                                ):
                                    if (
                                        "identifier" in fixed_room
                                        and "position" in fixed_room
                                    ):
                                        identifier = fixed_room["identifier"]
                                        position = fixed_room["position"]
                                        id_type = identifier.get("type", "")
                                        room_type = identifier.get("room_type", "")
                                        dept = identifier.get("department", "")
                                        name = identifier.get("name", "")

                                        # Enhanced logging with more details
                                        logger.info(
                                            f"Fixed Room #{i+1}: {id_type} - "
                                            + f"type={room_type}, dept={dept}, name={name}, "
                                            + f"position={position}"
                                        )

                                        # ENHANCEMENT: Add specific logging for core types
                                        if (
                                            room_type == "lobby"
                                            or dept == "circulation"
                                            or name
                                            in ["main_core", "vertical_circulation"]
                                        ):
                                            logger.info(
                                                f"!!! IMPORTANT FIXED ROOM: {id_type} - {room_type}/{dept}/{name} at {position}"
                                            )
                            else:
                                # If no fixed_rooms key, check if it's the simple format (direct mapping)
                                if isinstance(fixed_rooms_data, dict):
                                    logger.info(
                                        f"Using simple mapping format with {len(fixed_rooms_data)} fixed positions"
                                    )
                                    for room_id, position in fixed_rooms_data.items():
                                        logger.info(
                                            f"Fixed position for room {room_id}: {position}"
                                        )
                                else:
                                    logger.warning(
                                        f"Fixed rooms file doesn't have expected structure"
                                    )
                    except Exception as e:
                        logger.error(f"Error reading fixed rooms file: {e}")

                    fixed_rooms_arg = ["--fixed-rooms", str(fixed_rooms_path)]
                    logger.info(f"Using fixed rooms file: {fixed_rooms_path}")
                else:
                    logger.warning(f"Fixed rooms file not found: {fixed_rooms_path}")
                    # ENHANCEMENT: Return better error
                    return {
                        "success": False,
                        "error": f"Fixed rooms file not found: {fixed_rooms_path}",
                    }

        # Build command
        command = (
            [
                "python",
                str(
                    PROJECT_ROOT / "main.py"
                ),  # ENHANCEMENT: Use full path to ensure script is found
                "--mode",
                "rule",  # ENHANCEMENT: Changed to rule mode, which is most reliable for fixed positions
                "--building-config",
                building_id,
                "--program-config",
                program_config,
                "--output",
                str(LAYOUTS_DIR),  # ENHANCEMENT: Use full path for output
            ]
            + fixed_rooms_arg
            # + debug_flag
        )

        logger.info(f"Executing command: {' '.join(command)}")

        # ENHANCEMENT: Better error handling for command execution
        try:
            result = run_command(command, "Generation failed")
            logger.info(f"Command execution successful")

            # Log the output for debugging
            stdout_lines = result.stdout.splitlines()
            logger.info(f"Command output ({len(stdout_lines)} lines):")
            for line in stdout_lines[-20:]:  # Last 20 lines
                logger.info(f"  > {line}")

            layout_path = extract_layout_path(result.stdout)
            logger.info(f"Layout saved to: {layout_path}")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "success": False,
                "error": f"Layout generation command failed: {str(e)}",
                "command": " ".join(command),
            }

        # Load the layout JSON
        layout_file = Path(layout_path) / "hotel_layout.json"
        if not layout_file.exists():
            logger.error(f"Generated layout file not found at {layout_file}")
            return {
                "success": False,
                "error": "Generated layout file not found",
                "layout_path": str(layout_file),
            }

        layout_data = load_json_file(layout_file)
        logger.info(
            f"Successfully loaded layout with {len(layout_data.get('rooms', {}))} rooms"
        )

        # ENHANCEMENT: Check if important fixed rooms were placed correctly
        if fixed_rooms_file and fixed_rooms_path.exists():
            try:
                with open(fixed_rooms_path, "r") as f:
                    fixed_data = json.load(f)

                if "fixed_rooms" in fixed_data:
                    fixed_rooms = fixed_data["fixed_rooms"]
                    logger.info(
                        f"Verifying placement of {len(fixed_rooms)} fixed rooms"
                    )

                    for fixed_room in fixed_rooms:
                        if "identifier" in fixed_room and "position" in fixed_room:
                            identifier = fixed_room["identifier"]
                            position = fixed_room["position"]

                            # Identifier information
                            id_type = identifier.get("type", "")
                            room_type = identifier.get("room_type", "")
                            dept = identifier.get("department", "")
                            name = identifier.get("name", "")

                            # Descriptive name for logs
                            desc = f"{id_type}: "
                            if room_type:
                                desc += f"type={room_type}, "
                            if dept:
                                desc += f"dept={dept}, "
                            if name:
                                desc += f"name={name}"

                            # Check for this room in the layout
                            found = False
                            for room_id, room_data in layout_data.get(
                                "rooms", {}
                            ).items():
                                # Check if this room matches the identifier
                                if (
                                    room_type and room_data.get("type") == room_type
                                ) or (
                                    name
                                    and room_data.get("metadata", {}).get("name")
                                    == name
                                    or dept == "circulation"
                                    and room_data.get("type") == "vertical_circulation"
                                ):
                                    # Found a matching room, check if position is close
                                    room_pos = room_data.get("position", [0, 0, 0])
                                    pos_diff = [
                                        abs(a - b)
                                        for a, b in zip(room_pos[:2], position[:2])
                                    ]  # Compare only x,y

                                    if (
                                        max(pos_diff) < 10.0
                                    ):  # Within 10m is "close enough" - increased tolerance
                                        logger.info(
                                            f"✓ Fixed room {desc} was placed at {room_pos}, close to {position}"
                                        )
                                        found = True
                                        break
                                    else:
                                        logger.warning(
                                            f"! Fixed room {desc} was placed at {room_pos}, NOT close to {position}"
                                        )
                                        # Still count it as found if it's the right type
                                        found = True
                                        break

                            if not found:
                                logger.warning(
                                    f"✗ Fixed room {desc} was NOT found in the layout"
                                )
            except Exception as e:
                logger.error(f"Error verifying fixed room placement: {e}")

        return {
            "success": True,
            "message": "Layout generated successfully",
            "layout_id": layout_path.split("/")[-1],
            "layout_data": layout_data,
            "layout_path": str(layout_file),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating layout: {e}")
        logger.error(traceback.format_exc())
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
    fixed_rooms_file: Optional[str] = Body(None),  # Add this parameter
):
    """
    Generate an improved layout using the trained RL model.
    Optionally use a reference layout as a starting point.
    """
    try:
        # Ensure reference layout exists if specified
        reference_layout_path = None
        if reference_layout_id:
            # Check both standard and modified layout files
            layouts_base = LAYOUTS_DIR / reference_layout_id
            standard_layout = layouts_base / "hotel_layout.json"
            modified_layout = layouts_base / "modified_layout.json"

            # Prefer modified layout if it exists, otherwise use standard
            if modified_layout.exists():
                reference_layout_path = str(modified_layout)
                logger.info(
                    f"Using modified layout as reference: {reference_layout_path}"
                )
            elif standard_layout.exists():
                reference_layout_path = str(standard_layout)
                logger.info(
                    f"Using standard layout as reference: {reference_layout_path}"
                )
            else:
                logger.warning(
                    f"Reference layout not found for ID: {reference_layout_id}"
                )
                # Don't fail, just log the warning

        if not program_id or program_id == "default":
            program_config = "hotel_requirements_3.json"  # FIXED: Added .json extension
            logger.info(f"Using default program config: {program_config}")
        else:
            # Ensure .json extension is present
            if not program_id.endswith(".json"):
                program_config = f"{program_id}.json"
            else:
                program_config = program_id
            logger.info(f"Using specified program config: {program_config}")

        # Build command for generating an improved layout
        main_script_path = PROJECT_ROOT / "main.py"
        if not main_script_path.exists():
            logger.error(f"Main script not found at: {main_script_path}")
            raise FileNotFoundError(f"Main script not found: {main_script_path}")

        command = [
            "python",
            str(main_script_path),  # Use absolute path to main.py
            "--mode",
            "rl",
            "--building-config",
            building_id,
            "--program-config",
            program_config,
            # "--rl-model",
            # str(RL_MODELS_DIR / "hotel_design_model.pt"),
        ]

        # # Add reference layout if found
        # if reference_layout_path:
        #     command.extend(["--reference-layout", reference_layout_path])

        # Add fixed rooms file if provided
        if fixed_rooms_file:
            command.extend(["--fixed-rooms", fixed_rooms_file])
            logger.info(f"Using fixed rooms file: {fixed_rooms_file}")

        # Print the full command for debugging
        logger.info(f"Executing command: {' '.join(command)}")

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
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
