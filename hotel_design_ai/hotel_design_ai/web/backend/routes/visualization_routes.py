"""
Routes for visualization generation for Hotel Design AI configurations.
"""

import os
import sys
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/visualize", tags=["Visualization"])

# Define paths
PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data"
BUILDING_DIR = DATA_DIR / "building"
PROGRAM_DIR = DATA_DIR / "program"
CONSTRAINTS_DIR = DATA_DIR / "constraints"
USER_DATA_DIR = PROJECT_ROOT / "user_data"
VISUALIZATION_DIR = USER_DATA_DIR / "visualizations"

# Ensure directories exist
for dir_path in [BUILDING_DIR, PROGRAM_DIR, USER_DATA_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Add project root to system path to import visualizers
sys.path.append(str(PROJECT_ROOT))

# Import visualizers
try:
    # Import the visualization modules
    from hotel_design_ai.visualization.constraints_visualizer import (
        HotelConstraintsVisualizer,
    )
    from hotel_design_ai.visualization.requirements_visualizer import (
        HotelRequirementsVisualizer,
    )

    VISUALIZERS_IMPORTED = True
    logger.info("Successfully imported visualization modules")
except ImportError as e:
    VISUALIZERS_IMPORTED = False
    logger.error(f"Error importing visualizers: {e}")


@router.post("/building/{config_id}")
async def visualize_building_config(config_id: str):
    """Generate visualizations for a building configuration."""
    try:
        # Check if visualizers were imported successfully
        if not VISUALIZERS_IMPORTED:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Visualization modules could not be imported",
                },
            )

        # Check if building configuration exists
        config_path = BUILDING_DIR / f"{config_id}.json"

        # If not found in building dir, try program dir as fallback
        if not config_path.exists():
            logger.warning(
                f"Building config not found at {config_path}, trying program directory"
            )
            config_path = PROGRAM_DIR / f"{config_id}.json"

        if not config_path.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Building configuration '{config_id}' not found",
                },
            )

        # Create a unique ID for this visualization set
        viz_id = f"{config_id}_{uuid.uuid4().hex[:8]}"

        # Create output directory
        output_dir = VISUALIZATION_DIR / viz_id
        output_dir.mkdir(exist_ok=True)

        # Visualizations list to return
        visualizations = []

        # For building configuration, we'll use the constraints visualizer
        # since it contains the information about building structure
        try:
            # Check if we have constraints to visualize
            if CONSTRAINTS_DIR.exists():
                # Create constraints visualizer
                visualizer = HotelConstraintsVisualizer(CONSTRAINTS_DIR)

                # Generate all visualizations
                visualizer.visualize_all()

                # Get output directory from visualizer
                constraints_output_dir = Path(visualizer.output_dir)

                # Copy the visualization files to our output directory
                for image_file in constraints_output_dir.glob("*.png"):
                    dest_file = output_dir / image_file.name
                    shutil.copy(image_file, dest_file)

                    # Add to visualizations list
                    visualizations.append(
                        {
                            "title": image_file.stem.replace("_", " ").title(),
                            "url": f"/visualizations/{viz_id}/{image_file.name}",
                            "description": f"Visualization of {image_file.stem.replace('_', ' ')}",
                            "type": "image/png",
                        }
                    )

                # Copy the HTML files as well
                for html_file in constraints_output_dir.glob("*.html"):
                    dest_file = output_dir / html_file.name
                    shutil.copy(html_file, dest_file)

                    # Add to visualizations list
                    visualizations.append(
                        {
                            "title": html_file.stem.replace("_", " ").title(),
                            "url": f"/visualizations/{viz_id}/{html_file.name}",
                            "description": f"Detailed table of {html_file.stem.replace('_', ' ')}",
                            "type": "text/html",
                        }
                    )

                # Clean up temporary directory
                if constraints_output_dir.exists() and constraints_output_dir.is_dir():
                    shutil.rmtree(constraints_output_dir)

            else:
                # If no constraints directory, return a helpful message
                logger.warning(f"Constraints directory not found: {CONSTRAINTS_DIR}")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "error": "Constraints directory not found",
                    },
                )

        except Exception as e:
            logger.error(f"Error running constraints visualizer: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Error generating constraints visualizations: {str(e)}",
                },
            )

        # Return success with visualizations
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "visualization_id": viz_id,
                "visualizations": visualizations,
            },
        )

    except Exception as e:
        logger.error(f"Error in visualize_building_config: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Error generating visualizations: {str(e)}",
            },
        )


@router.post("/program/{config_id}")
async def visualize_program_config(config_id: str):
    """Generate visualizations for a program configuration."""
    try:
        # Check if visualizers were imported successfully
        if not VISUALIZERS_IMPORTED:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Visualization modules could not be imported",
                },
            )

        # Check if program configuration exists
        config_path = PROGRAM_DIR / f"{config_id}.json"

        # If not found in program dir, try building dir as fallback
        if not config_path.exists():
            logger.warning(
                f"Program config not found at {config_path}, trying building directory"
            )
            config_path = BUILDING_DIR / f"{config_id}.json"

        if not config_path.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Program configuration '{config_id}' not found",
                },
            )

        # Create a unique ID for this visualization set
        viz_id = f"{config_id}_{uuid.uuid4().hex[:8]}"

        # Create output directory
        output_dir = VISUALIZATION_DIR / viz_id
        output_dir.mkdir(exist_ok=True)

        # Visualizations list to return
        visualizations = []

        # For program configuration, we'll use the requirements visualizer
        try:
            # Create requirements visualizer
            visualizer = HotelRequirementsVisualizer(str(config_path))

            # Generate all visualizations
            visualizer.visualize_all()

            # Get output directory from visualizer
            requirements_output_dir = Path(visualizer.output_dir)

            # Copy the visualization files to our output directory
            for image_file in requirements_output_dir.glob("*.png"):
                dest_file = output_dir / image_file.name
                shutil.copy(image_file, dest_file)

                # Add to visualizations list
                visualizations.append(
                    {
                        "title": image_file.stem.replace("_", " ").title(),
                        "url": f"/visualizations/{viz_id}/{image_file.name}",
                        "description": f"Visualization of {image_file.stem.replace('_', ' ')}",
                        "type": "image/png",
                    }
                )

            # Copy the HTML files as well
            for html_file in requirements_output_dir.glob("*.html"):
                dest_file = output_dir / html_file.name
                shutil.copy(html_file, dest_file)

                # Add to visualizations list
                visualizations.append(
                    {
                        "title": html_file.stem.replace("_", " ").title(),
                        "url": f"/visualizations/{viz_id}/{html_file.name}",
                        "description": f"Detailed table of {html_file.stem.replace('_', ' ')}",
                        "type": "text/html",
                    }
                )

            # Clean up temporary directory
            if requirements_output_dir.exists() and requirements_output_dir.is_dir():
                shutil.rmtree(requirements_output_dir)

        except Exception as e:
            logger.error(f"Error running requirements visualizer: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Error generating requirements visualizations: {str(e)}",
                },
            )

        # Return success with visualizations
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "visualization_id": viz_id,
                "visualizations": visualizations,
            },
        )

    except Exception as e:
        logger.error(f"Error in visualize_program_config: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Error generating visualizations: {str(e)}",
            },
        )


@router.get("/visualizations/{viz_id}/{filename}")
async def get_visualization_file(viz_id: str, filename: str):
    """Serve a visualization file."""
    file_path = VISUALIZATION_DIR / viz_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization file not found")

    return FileResponse(file_path)
