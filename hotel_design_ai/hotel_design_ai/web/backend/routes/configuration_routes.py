"""
Routes for configuration management for Hotel Design AI.
"""

import os
import logging
import json
from typing import Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Configuration Management"])

# Define paths
PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data"
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"

# Ensure directories exist
for dir_path in [BUILDING_DIR, PROGRAM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@router.get("/configuration/{config_type}/{config_id}")
async def get_configuration(config_type: str, config_id: str):
    """Get a specific configuration by type and ID."""
    try:
        logger.info(f"Fetching {config_type} configuration: {config_id}")

        # First try to find in the expected directory based on type
        if config_type.lower() == "building":
            primary_dir = BUILDING_DIR
            secondary_dir = PROGRAM_DIR  # Fallback directory
        elif config_type.lower() == "program":
            primary_dir = PROGRAM_DIR
            secondary_dir = BUILDING_DIR  # Fallback directory
        else:
            logger.error(f"Invalid configuration type: {config_type}")
            return {
                "success": False,
                "error": f"Invalid configuration type: {config_type}",
            }

        # Check primary directory first
        config_file = primary_dir / f"{config_id}.json"
        logger.info(f"Looking for configuration file at: {config_file}")

        # If not found in primary directory, try secondary directory
        if not config_file.exists():
            logger.warning(f"File not found in primary directory: {config_file}")
            config_file = secondary_dir / f"{config_id}.json"
            logger.info(f"Trying secondary directory: {config_file}")

        # Check if the file exists in either directory
        if not config_file.exists():
            logger.error(f"Configuration file not found in any directory: {config_id}")
            return {"success": False, "error": f"Configuration '{config_id}' not found"}

        # Load and parse the JSON file
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Return the configuration data
            return {
                "success": True,
                "config_id": config_id,
                "config_type": config_type,
                "config_data": config_data,
                "file_path": str(config_file),
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {config_file}: {e}")
            return {
                "success": False,
                "error": f"Invalid JSON in configuration file: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Error reading configuration file {config_file}: {e}")
            return {
                "success": False,
                "error": f"Error reading configuration file: {str(e)}",
            }

    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        return {"success": False, "error": f"Error getting configuration: {str(e)}"}
