# Add to hotel_design_ai/hotel_design_ai/web/backend/routes/files.py

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
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

# Ensure directories exist
for dir_path in [BUILDING_DIR, PROGRAM_DIR, USER_DATA_DIR, LAYOUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@router.get("/layouts")
async def list_layouts():
    """List all available layout outputs from local directory."""
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
                        except Exception:
                            pass

                    # Add to list
                    layouts.append(
                        {
                            "id": layout_dir.name,
                            "room_count": len(layout_data.get("rooms", {})),
                            "metrics": layout_data.get("metrics", {}),
                            "created_at": creation_date,
                            "preview_image": f"/layouts/{layout_dir.name}/hotel_layout_3d.png",
                            "width": layout_data.get("width", 0),
                            "length": layout_data.get("length", 0),
                            "height": layout_data.get("height", 0),
                        }
                    )

        return {"success": True, "layouts": layouts}

    except Exception as e:
        logger.error(f"Error listing layouts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing layouts: {str(e)}")


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
