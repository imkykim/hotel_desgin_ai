"""
Routes for generating visualizations of layout outputs.
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
import time

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/visualize/layout", tags=["Layout Visualization"])

# Define paths
PROJECT_ROOT = Path(__file__).parents[4]
DATA_DIR = PROJECT_ROOT / "data"
USER_DATA_DIR = PROJECT_ROOT / "user_data"
LAYOUTS_DIR = USER_DATA_DIR / "layouts"
VISUALIZATION_DIR = USER_DATA_DIR / "visualizations"

# Ensure directories exist
for dir_path in [DATA_DIR, USER_DATA_DIR, LAYOUTS_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Add project root to system path to import visualizers
sys.path.append(str(PROJECT_ROOT))

# Import visualization modules
try:
    # Import the required modules
    from hotel_design_ai.visualization.renderer import LayoutRenderer
    from hotel_design_ai.models.layout import Layout
    from hotel_design_ai.utils.diagram_metrics import LayoutMetrics

    VISUALIZERS_IMPORTED = True
    logger.info("Successfully imported layout visualization modules")
except ImportError as e:
    VISUALIZERS_IMPORTED = False
    logger.error(f"Error importing layout visualization modules: {e}")


@router.post("/{layout_id}")
async def visualize_layout(layout_id: str):
    """Generate visualizations for a layout."""
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

        # Check if layout exists
        layout_dir = LAYOUTS_DIR / layout_id
        layout_file = layout_dir / "hotel_layout.json"

        if not layout_file.exists():
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Layout '{layout_id}' not found"},
            )

        # Create a unique ID for this visualization set
        viz_id = f"layout_{layout_id}_{uuid.uuid4().hex[:8]}"

        # Create output directory
        output_dir = VISUALIZATION_DIR / viz_id
        output_dir.mkdir(exist_ok=True)

        # Load the layout
        with open(layout_file, "r") as f:
            layout_data = json.load(f)

        # Visualizations list to return
        visualizations = []

        try:
            # The layout file already contains visualization images - let's use them
            # First, let's check if visualization files exist in the layout directory
            found_vis_files = False

            # Copy existing visualization files if they exist
            for image_file in layout_dir.glob("*.png"):
                dest_file = output_dir / image_file.name
                shutil.copy(image_file, dest_file)

                # Create a title based on filename
                title = image_file.stem
                title = title.replace("hotel_layout_", "").replace("_", " ").title()

                if "3d" in title.lower():
                    title = "3D Layout View"
                    description = "Three-dimensional view of the complete hotel layout"
                elif "floor" in title.lower():
                    floor_num = title.lower().replace("floor", "").strip()
                    try:
                        if floor_num.startswith("-"):
                            title = f"Basement {abs(int(floor_num))} Plan"
                            description = (
                                f"Floor plan for basement level {abs(int(floor_num))}"
                            )
                        else:
                            floor_num = int(floor_num)
                            title = f"Floor {floor_num} Plan"
                            description = f"Floor plan for level {floor_num}"
                    except ValueError:
                        # Just use the original title if parsing fails
                        description = f"Floor plan visualization"
                elif "standard" in title.lower():
                    title = "Standard Floor Plan"
                    description = "Floor plan for typical standard floors (guest rooms)"
                else:
                    description = f"Layout visualization for {title}"

                # Add to visualizations list
                visualizations.append(
                    {
                        "title": title,
                        "url": f"/visualizations/{viz_id}/{image_file.name}",
                        "description": description,
                        "type": "image/png",
                    }
                )
                found_vis_files = True

            # If we didn't find any visualizations, generate metrics visualizations
            if not found_vis_files:
                logger.info(
                    f"No existing visualization files found, generating metrics visualizations"
                )

                # Create temporary metrics visualizations using the metrics data
                # Create a simple bar chart of metrics using matplotlib
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np

                    # Extract metrics from layout data
                    metrics = layout_data.get("metrics", {})

                    if metrics and len(metrics) > 0:
                        # Create metrics visualization
                        plt.figure(figsize=(10, 6))

                        # Filter out the overall score and any non-numeric values
                        filtered_metrics = {
                            k: v
                            for k, v in metrics.items()
                            if k != "overall_score" and isinstance(v, (int, float))
                        }

                        # Format labels and values
                        labels = [
                            k.replace("_", " ").title() for k in filtered_metrics.keys()
                        ]
                        values = [
                            v * 100 if v <= 1 else v for v in filtered_metrics.values()
                        ]

                        # Create bar chart
                        y_pos = np.arange(len(labels))
                        plt.barh(
                            y_pos, values, align="center", alpha=0.7, color="#3b71ca"
                        )
                        plt.yticks(y_pos, labels)
                        plt.xlabel("Score (%)")
                        plt.title("Layout Performance Metrics")
                        plt.tight_layout()

                        # Save chart
                        metrics_file = output_dir / "layout_metrics.png"
                        plt.savefig(metrics_file)
                        plt.close()

                        # Add to visualizations
                        visualizations.append(
                            {
                                "title": "Performance Metrics",
                                "url": f"/visualizations/{viz_id}/layout_metrics.png",
                                "description": "Visualization of layout performance metrics",
                                "type": "image/png",
                            }
                        )

                        # Create room distribution visualization
                        rooms = layout_data.get("rooms", {})
                        if rooms:
                            # Count rooms by type
                            room_types = {}
                            for room_id, room_data in rooms.items():
                                room_type = room_data.get("type", "unknown")
                                if room_type not in room_types:
                                    room_types[room_type] = 0
                                room_types[room_type] += 1

                            if room_types:
                                plt.figure(figsize=(10, 6))

                                # Sort room types by count (descending)
                                sorted_types = sorted(
                                    room_types.items(), key=lambda x: x[1], reverse=True
                                )
                                labels = [
                                    t[0].replace("_", " ").title() for t in sorted_types
                                ]
                                values = [t[1] for t in sorted_types]

                                # Create bar chart
                                plt.bar(labels, values, color="#3b71ca")
                                plt.xticks(rotation=45, ha="right")
                                plt.ylabel("Count")
                                plt.title("Room Distribution by Type")
                                plt.tight_layout()

                                # Save chart
                                rooms_file = output_dir / "room_distribution.png"
                                plt.savefig(rooms_file)
                                plt.close()

                                # Add to visualizations
                                visualizations.append(
                                    {
                                        "title": "Room Distribution",
                                        "url": f"/visualizations/{viz_id}/room_distribution.png",
                                        "description": "Distribution of rooms by type",
                                        "type": "image/png",
                                    }
                                )

                except Exception as e:
                    logger.error(f"Error generating metrics visualizations: {e}")

            # If there are still no visualizations, create a placeholder
            if len(visualizations) == 0:
                logger.warning("No visualizations could be generated for this layout")

                # Create a simple placeholder image
                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(8, 6))
                    plt.text(
                        0.5,
                        0.5,
                        f"Layout {layout_id}\nNo visualizations available",
                        ha="center",
                        va="center",
                        fontsize=14,
                    )
                    plt.axis("off")

                    placeholder_file = output_dir / "placeholder.png"
                    plt.savefig(placeholder_file)
                    plt.close()

                    visualizations.append(
                        {
                            "title": "Layout Information",
                            "url": f"/visualizations/{viz_id}/placeholder.png",
                            "description": "Layout information visualization",
                            "type": "image/png",
                        }
                    )
                except Exception as e:
                    logger.error(f"Error creating placeholder visualization: {e}")

        except Exception as e:
            logger.error(f"Error processing layout visualizations: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Error processing layout visualizations: {str(e)}",
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
        logger.error(f"Error in visualize_layout: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Error generating layout visualizations: {str(e)}",
            },
        )
