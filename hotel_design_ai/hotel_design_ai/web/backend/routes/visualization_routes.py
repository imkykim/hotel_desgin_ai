"""
Routes for configuration visualization generation for Hotel Design AI.
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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router - keeping the original route prefix for configurations
router = APIRouter(prefix="/visualize", tags=["Configuration Visualization"])

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
    logger.info("Successfully imported configuration visualization modules")
except ImportError as e:
    VISUALIZERS_IMPORTED = False
    logger.error(f"Error importing configuration visualizers: {e}")


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
                # If no constraints directory, create some basic visualizations
                # Load the building configuration
                with open(config_path, "r") as f:
                    building_config = json.load(f)

                # Create a building footprint visualization
                plt.figure(figsize=(10, 8))
                width = building_config.get("width", 80)
                length = building_config.get("length", 120)

                # Draw footprint
                plt.plot(
                    [0, width, width, 0, 0],
                    [0, 0, length, length, 0],
                    "b-",
                    linewidth=3,
                )

                # Add structural grid if available
                if (
                    "structural_grid_x" in building_config
                    and "structural_grid_y" in building_config
                ):
                    grid_x = building_config["structural_grid_x"]
                    grid_y = building_config["structural_grid_y"]

                    # Draw vertical grid lines
                    for x in np.arange(0, width + 0.1, grid_x):
                        plt.plot([x, x], [0, length], "k--", alpha=0.3)

                    # Draw horizontal grid lines
                    for y in np.arange(0, length + 0.1, grid_y):
                        plt.plot([0, width], [y, y], "k--", alpha=0.3)

                plt.axis("equal")
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.xlim(-10, width + 10)
                plt.ylim(-10, length + 10)
                plt.title(f"Building Footprint ({width}m × {length}m)")
                plt.xlabel("Width (m)")
                plt.ylabel("Length (m)")

                # Save the plot
                footprint_file = output_dir / "building_footprint.png"
                plt.savefig(footprint_file, dpi=200)
                plt.close()

                # Add to visualizations
                visualizations.append(
                    {
                        "title": "Building Footprint",
                        "url": f"/visualizations/{viz_id}/building_footprint.png",
                        "description": f"Building footprint with dimensions {width}m × {length}m",
                        "type": "image/png",
                    }
                )

                # Create HTML table with building parameters
                building_params = {
                    "Parameter": [
                        "Width",
                        "Length",
                        "Height",
                        "Min Floor",
                        "Max Floor",
                        "Floor Height",
                        "Structural Grid X",
                        "Structural Grid Y",
                        "Grid Size",
                    ],
                    "Value": [
                        f"{building_config.get('width', 'N/A')}m",
                        f"{building_config.get('length', 'N/A')}m",
                        f"{building_config.get('height', 'N/A')}m",
                        str(building_config.get("min_floor", "N/A")),
                        str(building_config.get("max_floor", "N/A")),
                        f"{building_config.get('floor_height', 'N/A')}m",
                        f"{building_config.get('structural_grid_x', 'N/A')}m",
                        f"{building_config.get('structural_grid_y', 'N/A')}m",
                        f"{building_config.get('grid_size', 'N/A')}m",
                    ],
                }

                df = pd.DataFrame(building_params)

                # Create HTML table with styling
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                        th {{ background-color: #3B71CA; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        tr:hover {{ background-color: #ddd; }}
                        .title {{ font-size: 24px; margin-bottom: 20px; color: #333; }}
                    </style>
                </head>
                <body>
                    <div class="title">Building Parameters</div>
                    {df.to_html(index=False)}
                </body>
                </html>
                """

                # Save HTML file
                html_file = output_dir / "building_parameters.html"
                with open(html_file, "w") as f:
                    f.write(html)

                # Add to visualizations
                visualizations.append(
                    {
                        "title": "Building Parameters",
                        "url": f"/visualizations/{viz_id}/building_parameters.html",
                        "description": "Detailed table of building parameters",
                        "type": "text/html",
                    }
                )

        except Exception as e:
            logger.error(f"Error creating building visualizations: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Error generating building visualizations: {str(e)}",
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
        import traceback

        logger.error(traceback.format_exc())
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
            import traceback

            logger.error(traceback.format_exc())
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
        import traceback

        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Error generating visualizations: {str(e)}",
            },
        )
