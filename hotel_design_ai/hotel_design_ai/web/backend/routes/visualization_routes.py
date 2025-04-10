"""
Routes for layout visualization generation for Hotel Design AI.
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

# Create router
router = APIRouter(prefix="/visualize-layout", tags=["Layout Visualization"])

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

# Import visualizers
try:
    # Import the visualization modules
    from hotel_design_ai.utils.diagram_metrics import LayoutMetrics
    from hotel_design_ai.visualization.renderer import LayoutRenderer
    from hotel_design_ai.visualization.requirements_visualizer import (
        HotelRequirementsVisualizer,
    )
    from hotel_design_ai.models.layout import Layout
    from hotel_design_ai.core.spatial_grid import SpatialGrid

    VISUALIZERS_IMPORTED = True
    logger.info("Successfully imported visualization modules")
except ImportError as e:
    VISUALIZERS_IMPORTED = False
    logger.error(f"Error importing visualizers: {e}")


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

        # Load layout from JSON file
        with open(layout_file, "r") as f:
            layout_data = json.load(f)

        # Create the metrics visualizations
        visualizations = []

        try:
            # Create a temporary directory for the visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert layout_data to SpatialGrid or Layout object
                spatial_grid = SpatialGrid(
                    width=layout_data.get("width", 80),
                    length=layout_data.get("length", 120),
                    height=layout_data.get("height", 30),
                    grid_size=layout_data.get("grid_size", 1.0),
                )

                # Add rooms from the layout data
                for room_id, room_data in layout_data.get("rooms", {}).items():
                    room_id_int = int(room_id)
                    position = room_data.get("position", [0, 0, 0])
                    dimensions = room_data.get("dimensions", [0, 0, 0])
                    room_type = room_data.get("type", "default")
                    metadata = room_data.get("metadata", {})

                    # Add the room to the spatial grid
                    spatial_grid.place_room(
                        room_id=room_id_int,
                        x=position[0],
                        y=position[1],
                        z=position[2],
                        width=dimensions[0],
                        length=dimensions[1],
                        height=dimensions[2],
                        room_type=room_type,
                        metadata=metadata,
                    )

                # Create metrics calculator
                metrics = LayoutMetrics(spatial_grid)

                # Get the metrics data
                metrics_data = layout_data.get("metrics", {})

                # Generate relationship diagram
                plt.figure(figsize=(16, 12))
                G = nx.DiGraph()

                # Create nodes for rooms
                for room_id, room_data in spatial_grid.rooms.items():
                    G.add_node(
                        room_id,
                        name=room_data.get("type", "unknown"),
                        color=room_data.get("type", "default"),
                    )

                # Create edges based on adjacency
                adjacency_graph = (
                    spatial_grid.get_adjacency_graph()
                    if hasattr(spatial_grid, "get_adjacency_graph")
                    else {}
                )
                for room_id, neighbors in adjacency_graph.items():
                    for neighbor in neighbors:
                        G.add_edge(
                            room_id, neighbor, relationship="adjacency", weight=2
                        )

                # Draw the graph
                pos = nx.spring_layout(G, k=0.15, iterations=50)

                # Draw nodes with colors based on room type
                node_colors = [G.nodes[n].get("color", "default") for n in G.nodes()]
                nx.draw_networkx_nodes(
                    G, pos, node_size=700, node_color=node_colors, alpha=0.8
                )

                # Draw node labels
                labels = {n: G.nodes[n].get("name", str(n)) for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

                # Draw edges
                nx.draw_networkx_edges(G, pos, edge_color="green", arrows=True)

                plt.title("Room Relationship Diagram", fontsize=18)
                plt.axis("off")
                plt.tight_layout()

                # Save the plot
                relationship_file = Path(temp_dir) / "room_relationships.png"
                plt.savefig(relationship_file, dpi=200)
                plt.close()

                # Copy the file to our output directory
                relationship_dest = output_dir / "room_relationships.png"
                shutil.copy(relationship_file, relationship_dest)

                # Add to visualizations
                visualizations.append(
                    {
                        "title": "Room Relationships",
                        "url": f"/visualizations/{viz_id}/room_relationships.png",
                        "description": "Visualization of room relationships and adjacencies",
                        "type": "image/png",
                    }
                )

                # Generate metrics visualization
                for metric_name, metric_value in metrics_data.items():
                    if metric_name == "overall_score" or isinstance(metric_value, dict):
                        continue

                    # Create bar chart for the metric
                    plt.figure(figsize=(10, 6))
                    plt.bar(["Value"], [metric_value], color="blue", alpha=0.7)
                    plt.title(f"{metric_name.replace('_', ' ').title()}", fontsize=16)
                    plt.ylabel("Score (0-1)", fontsize=14)
                    plt.ylim(0, 1)
                    plt.text(
                        0,
                        metric_value + 0.05,
                        f"{metric_value:.2f}",
                        ha="center",
                        fontsize=12,
                    )
                    plt.tight_layout()

                    # Save the plot
                    metric_file = Path(temp_dir) / f"{metric_name}.png"
                    plt.savefig(metric_file, dpi=200)
                    plt.close()

                    # Copy the file to our output directory
                    metric_dest = output_dir / f"{metric_name}.png"
                    shutil.copy(metric_file, metric_dest)

                    # Add to visualizations
                    visualizations.append(
                        {
                            "title": f"{metric_name.replace('_', ' ').title()}",
                            "url": f"/visualizations/{viz_id}/{metric_name}.png",
                            "description": f"Visualization of {metric_name.replace('_', ' ')} metric",
                            "type": "image/png",
                        }
                    )

                # Create a summary table of rooms by type
                room_types = {}
                for room_data in spatial_grid.rooms.values():
                    room_type = room_data.get("type", "unknown")
                    if room_type not in room_types:
                        room_types[room_type] = 0
                    room_types[room_type] += 1

                # Create DataFrame and save as HTML
                df = pd.DataFrame(
                    list(room_types.items()), columns=["Room Type", "Count"]
                )
                df = df.sort_values("Count", ascending=False)

                html = df.to_html(index=False)
                styled_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                        th {{ background-color: #4CAF50; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        tr:hover {{ background-color: #ddd; }}
                        .caption {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }}
                    </style>
                </head>
                <body>
                    <div class="caption">Room Type Distribution</div>
                    {html}
                </body>
                </html>
                """

                # Save the HTML
                html_file = Path(temp_dir) / "room_types.html"
                with open(html_file, "w") as f:
                    f.write(styled_html)

                # Copy the file to our output directory
                html_dest = output_dir / "room_types.html"
                shutil.copy(html_file, html_dest)

                # Add to visualizations
                visualizations.append(
                    {
                        "title": "Room Type Distribution",
                        "url": f"/visualizations/{viz_id}/room_types.html",
                        "description": "Distribution of rooms by type",
                        "type": "text/html",
                    }
                )

            # Generate bar chart for all metrics
            plt.figure(figsize=(12, 8))
            metrics_to_plot = {}
            for k, v in metrics_data.items():
                if (
                    k != "overall_score"
                    and not isinstance(v, dict)
                    and not isinstance(v, list)
                ):
                    metrics_to_plot[k.replace("_", " ").title()] = v

            if metrics_to_plot:
                labels = list(metrics_to_plot.keys())
                values = list(metrics_to_plot.values())

                plt.bar(labels, values, color="skyblue", alpha=0.7)
                plt.title("Layout Performance Metrics", fontsize=16)
                plt.ylabel("Score (0-1)", fontsize=14)
                plt.ylim(0, 1)
                plt.xticks(rotation=45, ha="right")

                # Add value labels above bars
                for i, v in enumerate(values):
                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

                plt.tight_layout()

                # Save the plot
                all_metrics_file = output_dir / "all_metrics.png"
                plt.savefig(all_metrics_file, dpi=200)
                plt.close()

                # Add to visualizations
                visualizations.append(
                    {
                        "title": "All Metrics Summary",
                        "url": f"/visualizations/{viz_id}/all_metrics.png",
                        "description": "Summary of all performance metrics",
                        "type": "image/png",
                    }
                )

        except Exception as e:
            logger.error(f"Error creating layout visualizations: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Error creating layout visualizations: {str(e)}",
                },
            )

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
                "error": f"Error generating visualizations: {str(e)}",
            },
        )
