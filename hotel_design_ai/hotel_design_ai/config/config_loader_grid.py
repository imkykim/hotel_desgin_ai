"""
Grid-aligned dimension calculations for hotel room layout.
This module provides utilities for calculating room dimensions
that align to structural grids.
"""

import math
from typing import Dict, List, Tuple, Any, Optional, Union


def calculate_grid_aligned_dimensions(
    area: float,
    grid_x: float,
    grid_y: float,
    min_width: Optional[float] = None,
    grid_fraction: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate room dimensions aligned to structural grid.

    Args:
        area: Required area in square meters
        grid_x: X structural grid spacing in meters
        grid_y: Y structural grid spacing in meters
        min_width: Minimum width in meters (optional)
        grid_fraction: Fraction of grid to align to (0.5 = half grid)

    Returns:
        Tuple[float, float]: (width, length) in meters
    """
    # Ensure area is positive
    area = max(1.0, area)

    # Calculate grid snap sizes
    snap_size_x = grid_x * grid_fraction
    snap_size_y = grid_y * grid_fraction

    # Calculate grid cell area
    grid_cell_area = grid_x * grid_y

    # Calculate number of grid cells needed
    grid_cells_needed = area / grid_cell_area

    # Round up to the nearest fraction of grid
    grid_cells_rounded = math.ceil(grid_cells_needed / grid_fraction) * grid_fraction

    # If the room needs less than one grid cell
    if grid_cells_rounded < 1.0:
        grid_cells_rounded = max(0.5, grid_cells_rounded)  # Minimum half a grid cell

    # Calculate optimal grid-aligned dimensions
    # Try to make the shape as square as possible while respecting grid alignment
    width_cells = math.sqrt(grid_cells_rounded)
    length_cells = grid_cells_rounded / width_cells

    # Round width and length to nearest grid fraction
    width_cells = round(width_cells / grid_fraction) * grid_fraction

    # Ensure at least half a grid for width
    if width_cells < 0.5:
        width_cells = 0.5

    # Recalculate length based on width to maintain area
    length_cells = grid_cells_rounded / width_cells

    # Round length to nearest grid fraction
    length_cells = round(length_cells / grid_fraction) * grid_fraction

    # Ensure at least half a grid for length
    if length_cells < 0.5:
        length_cells = 0.5
        width_cells = grid_cells_rounded / length_cells
        width_cells = round(width_cells / grid_fraction) * grid_fraction

    # Convert from grid cells to meters
    width = width_cells * grid_x
    length = length_cells * grid_y

    # Make sure we meet minimum width if specified
    if min_width is not None and width < min_width:
        # Round up to nearest snap size
        width = math.ceil(min_width / snap_size_x) * snap_size_x

        # Recalculate length to maintain area
        area = grid_cells_rounded * grid_cell_area
        length = area / width
        length = round(length / snap_size_y) * snap_size_y

    # Add a bit of buffer to ensure area requirements
    actual_area = width * length
    if actual_area < area * 0.95:  # Allow 5% tolerance
        # Increase dimensions slightly
        if width <= length:
            width = math.ceil(width / snap_size_x) * snap_size_x
        else:
            length = math.ceil(length / snap_size_y) * snap_size_y

    return width, length


def adjust_room_list_to_grid(
    rooms: List[Dict[str, Any]],
    grid_x: float,
    grid_y: float,
    grid_fraction: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Adjust dimensions of all rooms in a list to align with structural grid.

    Args:
        rooms: List of room dictionaries
        grid_x: X structural grid spacing in meters
        grid_y: Y structural grid spacing in meters
        grid_fraction: Fraction of grid to align to (0.5 = half grid)

    Returns:
        List[Dict[str, Any]]: Adjusted room list
    """
    adjusted_rooms = []

    for room in rooms:
        # Skip rooms that don't have dimensions or area
        if "width" not in room or "length" not in room:
            adjusted_rooms.append(room.copy())
            continue

        # Calculate current area
        current_area = room["width"] * room["length"]

        # Get minimum width if specified
        min_width = room.get("min_width", None)

        # Calculate grid-aligned dimensions
        width, length = calculate_grid_aligned_dimensions(
            current_area, grid_x, grid_y, min_width, grid_fraction
        )

        # Create adjusted room
        adjusted_room = room.copy()
        adjusted_room["width"] = width
        adjusted_room["length"] = length
        adjusted_room["grid_aligned"] = True

        # Add metadata about original dimensions
        if "metadata" not in adjusted_room:
            adjusted_room["metadata"] = {}

        adjusted_room["metadata"]["original_width"] = room["width"]
        adjusted_room["metadata"]["original_length"] = room["length"]
        adjusted_room["metadata"]["original_area"] = current_area
        adjusted_room["metadata"]["adjusted_area"] = width * length

        adjusted_rooms.append(adjusted_room)

    return adjusted_rooms


def calculate_room_dimensions_grid_aligned(
    area: float,
    min_width: float,
    structural_grid_x: float,
    structural_grid_y: float,
    grid_fraction: float = 0.5,
) -> Tuple[float, float]:
    """
    Enhanced version of _calculate_room_dimensions that aligns with structural grid.
    This is a drop-in replacement for the original function in config_loader.py.

    Args:
        area: Required area in square meters
        min_width: Minimum width in meters
        structural_grid_x: X structural grid spacing
        structural_grid_y: Y structural grid spacing
        grid_fraction: Fraction of grid to align to (0.5 = half grid)

    Returns:
        Tuple of (width, length) in meters
    """
    return calculate_grid_aligned_dimensions(
        area, structural_grid_x, structural_grid_y, min_width, grid_fraction
    )
