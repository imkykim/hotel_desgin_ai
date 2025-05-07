from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/files", tags=["File Browsing"])

# Define paths
PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data"
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"
USER_DATA_DIR = PROJECT_ROOT / "user_data"
LAYOUTS_DIR = USER_DATA_DIR / "layouts"
FIX_DIR = DATA_DIR / "fix"

# Ensure directories exist
for dir_path in [BUILDING_DIR, PROGRAM_DIR, USER_DATA_DIR, LAYOUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@router.get("/layouts")
async def list_layouts():
    """List all available layout outputs from local directory."""
    try:
        layouts = []
        logger.info(f"Searching for layouts in: {LAYOUTS_DIR}")

        # Get all layout directories
        for layout_dir in LAYOUTS_DIR.iterdir():
            if layout_dir.is_dir():
                layout_file = layout_dir / "hotel_layout.json"
                logger.info(f"Checking layout file: {layout_file}")

                if layout_file.exists():
                    try:
                        # Load basic layout info
                        with open(layout_file, "r") as f:
                            layout_data = json.load(f)

                        # Extract creation date from directory name if possible
                        creation_date = ""
                        if "_" in layout_dir.name:
                            date_part = layout_dir.name.split("_")[0]
                            try:
                                # Parse YYYYMMDD format
                                if len(date_part) == 8:
                                    year = date_part[:4]
                                    month = date_part[4:6]
                                    day = date_part[6:8]
                                    creation_date = f"{year}-{month}-{day}"
                            except Exception as e:
                                logger.warning(f"Error parsing date: {e}")

                        # Check for preview image
                        preview_image = (
                            f"/layouts/{layout_dir.name}/hotel_layout_3d.png"
                        )
                        preview_image_path = layout_dir / "hotel_layout_3d.png"
                        has_preview = preview_image_path.exists()

                        logger.info(
                            f"Preview image path: {preview_image_path}, exists: {has_preview}"
                        )

                        # Add to list
                        layouts.append(
                            {
                                "id": layout_dir.name,
                                "room_count": len(layout_data.get("rooms", {})),
                                "metrics": layout_data.get("metrics", {}),
                                "created_at": creation_date,
                                "preview_image": preview_image if has_preview else None,
                                "has_preview": has_preview,
                                "width": layout_data.get("width", 0),
                                "length": layout_data.get("length", 0),
                                "height": layout_data.get("height", 0),
                            }
                        )
                        logger.info(f"Added layout: {layout_dir.name}")
                    except Exception as e:
                        logger.error(f"Error processing layout {layout_dir.name}: {e}")

        logger.info(f"Found {len(layouts)} layouts")
        return {"success": True, "layouts": layouts}

    except Exception as e:
        logger.error(f"Error listing layouts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing layouts: {str(e)}")


@router.get("/layouts/{layout_id}")
async def get_layout_detail(layout_id: str):
    """Get details for a specific layout by ID."""
    try:
        # Check if layout exists
        layout_dir = LAYOUTS_DIR / layout_id
        layout_file = layout_dir / "hotel_layout.json"

        logger.info(f"Looking for layout file: {layout_file}")

        if not layout_file.exists():
            logger.error(f"Layout file not found: {layout_file}")
            raise HTTPException(status_code=404, detail="Layout not found")

        # Load layout
        with open(layout_file, "r") as f:
            layout_data = json.load(f)

        metrics_file = layout_dir / "hotel_layout_metrics.json"
        has_separate_metrics = metrics_file.exists()

        # Load metrics from separate file if it exists
        detailed_metrics = None
        if has_separate_metrics:
            try:
                with open(metrics_file, "r") as f:
                    detailed_metrics = json.load(f)
                logger.info(f"Loaded detailed metrics from {metrics_file}")
            except Exception as e:
                logger.error(f"Error loading metrics file: {e}")

        # Find available preview images
        preview_3d = layout_dir / "hotel_layout_3d.png"
        has_3d_preview = preview_3d.exists()

        # Find floor plan images
        floor_images = {}
        for floor in range(-2, 6):  # Check floors -2 to 5
            floor_image = layout_dir / f"hotel_layout_floor{floor}.png"
            if floor_image.exists():
                floor_images[floor] = (
                    f"/layouts/{layout_id}/hotel_layout_floor{floor}.png"
                )

        # Return layout data
        return {
            "success": True,
            "layout_id": layout_id,
            "layout_data": layout_data,
            "detailed_metrics": detailed_metrics,
            "has_separate_metrics": has_separate_metrics,
            "image_urls": {
                "3d": (
                    f"/layouts/{layout_id}/hotel_layout_3d.png"
                    if has_3d_preview
                    else None
                ),
                "has_3d_preview": has_3d_preview,
                "floor_plans": floor_images,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting layout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting layout: {str(e)}")


@router.get("/configurations")
async def list_configurations():
    """List all available building and program configurations."""
    try:
        # Gather building configurations
        building_configs = []
        for config_file in BUILDING_DIR.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                # Get file creation time
                file_stat = os.stat(config_file)
                creation_date = datetime.fromtimestamp(file_stat.st_ctime).strftime(
                    "%Y%m%d"
                )

                building_configs.append(
                    {
                        "id": config_file.stem,
                        "filename": config_file.name,
                        "type": "building",
                        "dimensions": {
                            "width": config_data.get("width", 0),
                            "length": config_data.get("length", 0),
                            "height": config_data.get("height", 0),
                        },
                        "created_at": creation_date,
                    }
                )
            except Exception as e:
                logger.error(f"Error processing file {config_file}: {e}")

        # Gather program configurations
        program_configs = []
        for config_file in PROGRAM_DIR.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                # Try to extract hotel type and room count from filename or content
                hotel_type = "Standard"
                room_count = 0

                # Extract from filename
                name_parts = config_file.stem.split("_")
                if len(name_parts) > 1:
                    hotel_type = name_parts[0].capitalize()

                # Get file creation time
                file_stat = os.stat(config_file)
                creation_date = datetime.fromtimestamp(file_stat.st_ctime).strftime(
                    "%Y%m%d"
                )

                program_configs.append(
                    {
                        "id": config_file.stem,
                        "filename": config_file.name,
                        "type": "program",
                        "hotel_type": hotel_type,
                        "room_count": room_count,
                        "created_at": creation_date,
                    }
                )
            except Exception as e:
                logger.error(f"Error processing file {config_file}: {e}")

        return {
            "success": True,
            "building_configs": building_configs,
            "program_configs": program_configs,
        }

    except Exception as e:
        logger.error(f"Error listing configurations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing configurations: {str(e)}"
        )


@router.get("/configuration/{config_type}/{config_id}")
async def get_configuration(config_type: str, config_id: str):
    """Get a specific configuration by type and ID."""
    try:
        config_dir = BUILDING_DIR if config_type == "building" else PROGRAM_DIR
        config_file = config_dir / f"{config_id}.json"

        if not config_file.exists():
            raise HTTPException(
                status_code=404, detail=f"Configuration not found: {config_id}"
            )

        with open(config_file, "r") as f:
            config_data = json.load(f)

        return {
            "success": True,
            "config_id": config_id,
            "config_type": config_type,
            "config_data": config_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting configuration: {str(e)}"
        )


@router.post("/save-fixed-elements")
async def save_fixed_elements(data: Dict[str, Any] = Body(...)):
    """
    Save fixed elements to a JSON file, ensuring no secondary cores are included.
    """
    try:
        building_id = data.get("building_id")
        fixed_elements = data.get("fixed_elements")

        if not building_id or not fixed_elements:
            raise HTTPException(
                status_code=400,
                detail="Building ID and fixed elements data are required",
            )

        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{building_id}_{timestamp}_fixed_rooms.json"
        filepath = FIX_DIR / filename

        # IMPORTANT: Filter out any secondary cores
        if "fixed_rooms" in fixed_elements:
            # Create a new list with only non-secondary core elements
            filtered_rooms = []
            for room in fixed_elements["fixed_rooms"]:
                # Check if this is a secondary core by name
                is_secondary = False
                if "identifier" in room and "name" in room["identifier"]:
                    if "secondary_core" in room["identifier"]["name"]:
                        logger.warning(
                            f"Filtering out secondary core: {room['identifier']['name']}"
                        )
                        is_secondary = True

                # Only keep non-secondary cores
                if not is_secondary:
                    filtered_rooms.append(room)

            # Replace the fixed_rooms list with our filtered version
            fixed_elements["fixed_rooms"] = filtered_rooms

            # Log what we're keeping
            logger.info(
                f"Keeping {len(filtered_rooms)} fixed elements after filtering out secondary cores"
            )

            # Verify all cores are main_core
            core_count = 0
            for room in filtered_rooms:
                if "identifier" in room and "value" in room["identifier"]:
                    if room["identifier"]["value"] == "vertical_circulation":
                        core_count += 1
                        # Make absolutely sure it's named main_core
                        if "name" in room["identifier"]:
                            room["identifier"]["name"] = "main_core"
                        if "metadata" in room and "original_name" in room["metadata"]:
                            room["metadata"]["original_name"] = "Main Circulation Core"

            logger.info(f"Final count of vertical circulation cores: {core_count}")

        # Add metadata to track this change
        if "metadata" not in fixed_elements:
            fixed_elements["metadata"] = {}

        fixed_elements["metadata"]["contains_secondary_cores"] = False
        fixed_elements["metadata"]["filtered_with_version"] = "1.3"
        fixed_elements["metadata"]["timestamp"] = datetime.now().isoformat()

        # Save to file
        with open(filepath, "w") as f:
            json.dump(fixed_elements, f, indent=2)

        logger.info(f"Fixed elements saved to: {filepath}")

        # Return the filepath so frontend can use it for layout generation
        return {
            "success": True,
            "message": "Fixed elements saved successfully (secondary cores removed)",
            "filepath": str(filepath),
            "filename": filename,
        }
    except Exception as e:
        logger.error(f"Error saving fixed elements: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error saving fixed elements: {str(e)}"
        )
