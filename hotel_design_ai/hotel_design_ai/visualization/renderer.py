import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Set, Any, Union

from hotel_design_ai.core.spatial_grid import SpatialGrid


class StyleConfig:
    """Configuration for visualization styles."""

    # Default room type colors
    DEFAULT_ROOM_COLORS = {
        "entrance": "#2c7fb8",
        "lobby": "#7fcdbb",
        "vertical_circulation": "#FF0000",
        "restaurant": "#f0f9e8",
        "meeting_room": "#edf8b1",
        "guest_room": "#f7fcb9",
        "service_area": "#d9f0a3",
        "back_of_house": "#addd8e",
        # Default color for unknown types
        "default": "#efefef",
    }

    # Default room type transparencies
    DEFAULT_ROOM_ALPHAS = {
        "entrance": 0.9,
        "lobby": 0.7,
        "vertical_circulation": 0.8,
        "restaurant": 0.7,
        "meeting_room": 0.7,
        "guest_room": 0.6,
        "service_area": 0.5,
        "back_of_house": 0.5,
        "default": 0.5,
    }

    @classmethod
    def brighten_color(cls, color_str: str) -> str:
        """
        Brighten a color for highlighting.

        Args:
            color_str: Color to brighten

        Returns:
            str: Brightened color
        """
        rgb = mcolors.to_rgb(color_str)
        # Make color brighter (closer to white)
        brightened = [min(1.0, c * 1.5) for c in rgb]
        return mcolors.rgb2hex(brightened)


class LayoutRenderer:
    """
    3D visualization renderer for hotel layouts.
    """

    def __init__(self, layout: SpatialGrid, building_config: Dict[str, Any] = None):
        """
        Initialize the renderer.

        Args:
            layout: The spatial grid layout to render
            building_config: Building configuration parameters
        """
        self.layout = layout
        self.building_config = building_config or {"floor_height": 4.0}

        # Set default colors and alphas from StyleConfig
        self.room_colors = StyleConfig.DEFAULT_ROOM_COLORS.copy()
        self.room_alphas = StyleConfig.DEFAULT_ROOM_ALPHAS.copy()

    def render_3d(
        self,
        ax=None,
        fig=None,
        show_labels: bool = True,
        highlight_rooms: List[int] = None,
    ):
        """
        Render the layout in 3D.

        Args:
            ax: Optional matplotlib 3D axis
            fig: Optional matplotlib figure
            show_labels: Whether to show room labels
            highlight_rooms: Optional list of room IDs to highlight

        Returns:
            fig, ax: The matplotlib figure and axis
        """
        if fig is None or ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        # Set up the axes
        self._setup_3d_axes(ax)

        # Draw bounding box
        self._draw_bounding_box(ax)

        # Draw rooms
        for room_id, room_data in self.layout.rooms.items():
            is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
            self._draw_room(ax, room_id, room_data, show_labels, is_highlighted)

        # Adjust view for better visibility
        ax.view_init(elev=30, azim=-45)

        # Set equal aspect ratio
        self._set_aspect_equal_3d(ax)

        return fig, ax

    def _setup_3d_axes(self, ax):
        """Set up the 3D axes with proper labels and limits."""
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Length (m)")
        ax.set_zlabel("Height (m)")

        # Set axis limits
        ax.set_xlim(0, self.layout.width)
        ax.set_ylim(0, self.layout.length)
        ax.set_zlim(0, self.layout.height)

    def render_floor_plan(
        self,
        floor: int = 0,
        ax=None,
        fig=None,
        show_labels: bool = True,
        highlight_rooms: List[int] = None,
    ):
        """
        Render a 2D floor plan.

        Args:
            floor: Floor number to render (0 = ground floor)
            ax: Optional matplotlib axis
            fig: Optional matplotlib figure
            show_labels: Whether to show room labels
            highlight_rooms: Optional list of room IDs to highlight

        Returns:
            fig, ax: The matplotlib figure and axis
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        # Set up the axes
        self._setup_2d_axes(ax, floor)

        # Draw rooms on this floor
        self._draw_rooms_on_floor(ax, floor, show_labels, highlight_rooms)

        # Draw structural grid
        self._draw_structural_grid(ax)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        return fig, ax

    def _setup_2d_axes(self, ax, floor: int):
        """Set up the 2D axes with proper labels and limits."""
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Length (m)")
        ax.set_title(f"Floor Plan - Level {floor}")

        # Set axis limits
        ax.set_xlim(0, self.layout.width)
        ax.set_ylim(0, self.layout.length)

    def _draw_rooms_on_floor(
        self, ax, floor: int, show_labels: bool, highlight_rooms: List[int] = None
    ):
        """
        Draw all rooms on a specific floor.

        Args:
            ax: Matplotlib axis
            floor: Floor number to render
            show_labels: Whether to show room labels
            highlight_rooms: Optional list of room IDs to highlight
        """
        floor_height = self.building_config.get("floor_height", 5.0)
        z_min = floor * floor_height
        z_max = (floor + 1) * floor_height

        # Find rooms on this floor
        rooms_on_floor = []
        for room_id, room_data in self.layout.rooms.items():
            x, y, z = room_data["position"]
            h = room_data["dimensions"][2]

            # Debug: Print room data for basement floor
            if floor == -1:
                print(
                    f"Room {room_id} ({room_data.get('type')}): z={z}, h={h}, z_min={z_min}, z_max={z_max}"
                )

            # Check if room is on this floor - improved condition for better basement handling
            room_bottom = z
            room_top = z + h
            floor_bottom = z_min
            floor_top = z_max

            # Room is on this floor if:
            # 1. Room bottom is within the floor range, OR
            # 2. Room top is within the floor range, OR
            # 3. Room completely contains the floor (spans multiple floors)
            if (
                (room_bottom >= floor_bottom and room_bottom < floor_top)
                or (room_top > floor_bottom and room_top <= floor_top)
                or (room_bottom <= floor_bottom and room_top >= floor_top)
            ):
                rooms_on_floor.append(room_id)

        # Debug: Show count of rooms found on this floor
        print(f"Found {len(rooms_on_floor)} rooms on floor {floor}")

        # Draw each room
        for room_id in rooms_on_floor:
            room_data = self.layout.rooms[room_id]
            is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
            self._draw_room_2d(ax, room_id, room_data, show_labels, is_highlighted)

    def save_renders(
        self,
        output_dir: str = "renders",
        prefix: str = "layout",
        include_3d: bool = True,
        include_floor_plans: bool = True,
        num_floors: int = None,
        min_floor: int = None,
    ):
        """
        Save renders to disk.

        Args:
            output_dir: Directory to save renders in
            prefix: Filename prefix
            include_3d: Whether to include 3D render
            include_floor_plans: Whether to include floor plans
            num_floors: Number of floors to render (if None, detect automatically)
            min_floor: Minimum floor to render (if None, use building_config or default to 0)
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save 3D render
        if include_3d:
            self._save_3d_render(output_dir, prefix)

        # Save floor plans
        if include_floor_plans:
            self._save_floor_plans(output_dir, prefix, num_floors, min_floor)

    def _save_3d_render(self, output_dir: str, prefix: str):
        """
        Save 3D render to disk.

        Args:
            output_dir: Directory to save renders in
            prefix: Filename prefix
        """
        fig, ax = self.render_3d()
        filename = os.path.join(output_dir, f"{prefix}_3d.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 3D render to {filename}")

    def _save_floor_plans(
        self,
        output_dir: str,
        prefix: str,
        num_floors: Optional[int] = None,
        min_floor: Optional[int] = None,
    ):
        """
        Save floor plans to disk.

        Args:
            output_dir: Directory to save renders in
            prefix: Filename prefix
            num_floors: Number of floors to render
            min_floor: Minimum floor to render
        """
        import os

        # Get min_floor from building_config if not specified
        if min_floor is None:
            min_floor = self.building_config.get("min_floor", 0)

        # Determine max_floor if num_floors is not specified
        if num_floors is None:
            # Try to get max_floor from building_config
            max_floor = self.building_config.get("max_floor")

            # If max_floor is not in building_config, calculate it
            if max_floor is None:
                max_floor = self._calculate_max_floor()
        else:
            # If num_floors is specified, calculate max_floor
            max_floor = min_floor + num_floors - 1

        # Render each floor
        for floor in range(min_floor, max_floor + 1):
            fig, ax = self.render_floor_plan(floor=floor)
            floor_name = f"basement" if floor < 0 else f"floor{floor}"
            filename = os.path.join(output_dir, f"{prefix}_{floor_name}.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved floor plan for level {floor} to {filename}")

    def _calculate_max_floor(self) -> int:
        """
        Calculate the maximum floor occupied by rooms.

        Returns:
            int: Maximum floor number
        """
        max_z = 0
        for _, room_data in self.layout.rooms.items():
            _, _, z = room_data["position"]
            _, _, h = room_data["dimensions"]
            max_z = max(max_z, z + h)

        floor_height = self.building_config["floor_height"]
        return int(np.ceil(max_z / floor_height)) - 1

    def _draw_bounding_box(self, ax):
        """Draw the building bounding box."""
        # Get vertices and faces of the bounding box
        vertices, faces = self._get_bounding_box_geometry()

        # Draw faces with very light gray, slightly transparent
        box = Poly3DCollection(
            faces, alpha=0.1, facecolor="lightgray", edgecolor="gray", linewidth=0.5
        )
        ax.add_collection3d(box)

    def _get_bounding_box_geometry(self):
        """
        Get vertices and faces for the bounding box.

        Returns:
            tuple: (vertices, faces) for the bounding box
        """
        # Vertices of the bounding box
        vertices = [
            [0, 0, 0],
            [self.layout.width, 0, 0],
            [self.layout.width, self.layout.length, 0],
            [0, self.layout.length, 0],
            [0, 0, self.layout.height],
            [self.layout.width, 0, self.layout.height],
            [self.layout.width, self.layout.length, self.layout.height],
            [0, self.layout.length, self.layout.height],
        ]

        # Faces of the bounding box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        ]

        return vertices, faces

    def _draw_room(
        self,
        ax,
        room_id: int,
        room_data: Dict[str, Any],
        show_labels: bool,
        highlighted: bool,
    ):
        """Draw a 3D room."""
        # Get room geometry
        vertices, faces = self._get_room_geometry(room_data)

        # Get styling
        face_color, edge_color, alpha, linewidth = self._get_room_style(
            room_data, highlighted
        )

        # Draw room
        room_poly = Poly3DCollection(
            faces,
            alpha=alpha,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=linewidth,
        )
        ax.add_collection3d(room_poly)

        # Add label to center of room
        if show_labels:
            self._add_room_label_3d(ax, room_id, room_data)

    def _get_room_geometry(self, room_data: Dict[str, Any]):
        """
        Get vertices and faces for a room.

        Args:
            room_data: Room data

        Returns:
            tuple: (vertices, faces) for the room
        """
        x, y, z = room_data["position"]
        w, l, h = room_data["dimensions"]

        # Vertices of the room
        vertices = [
            [x, y, z],
            [x + w, y, z],
            [x + w, y + l, z],
            [x, y + l, z],
            [x, y, z + h],
            [x + w, y, z + h],
            [x + w, y + l, z + h],
            [x, y + l, z + h],
        ]

        # Faces of the room
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        ]

        return vertices, faces

    def _get_room_style(self, room_data: Dict[str, Any], highlighted: bool):
        """
        Get styling information for a room.

        Args:
            room_data: Room data
            highlighted: Whether the room is highlighted

        Returns:
            tuple: (face_color, edge_color, alpha, linewidth)
        """
        room_type = room_data["type"]

        # Get room color and alpha
        face_color = self.room_colors.get(room_type, self.room_colors["default"])
        alpha = self.room_alphas.get(room_type, self.room_alphas["default"])

        # Adjust for highlighting
        if highlighted:
            face_color = StyleConfig.brighten_color(face_color)
            alpha = 0.9
            edge_color = "red"
            linewidth = 2
        else:
            edge_color = "black"
            linewidth = 0.5

        return face_color, edge_color, alpha, linewidth

    def _add_room_label_3d(self, ax, room_id: int, room_data: Dict[str, Any]):
        """
        Add a label to a 3D room.

        Args:
            ax: Matplotlib axis
            room_id: Room ID
            room_data: Room data
        """
        x, y, z = room_data["position"]
        w, l, h = room_data["dimensions"]

        # Calculate center position
        center_x = x + w / 2
        center_y = y + l / 2
        center_z = z + h / 2

        # Get appropriate room name
        room_name = self._get_room_display_name(room_id, room_data)

        # Add the label
        ax.text(
            center_x,
            center_y,
            center_z,
            room_name,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )

    def _get_room_display_name(self, room_id: int, room_data: Dict[str, Any]) -> str:
        """
        Get the display name for a room.

        Args:
            room_id: Room ID
            room_data: Room data

        Returns:
            str: Display name for the room
        """
        # First try to get name from metadata (highest priority)
        metadata = room_data.get("metadata", {})
        if "original_name" in metadata:
            return metadata["original_name"]

        # Try room type as it shows the program purpose (preferred for visualization)
        room_type = room_data["type"].replace("_", " ").title()

        # Then try to use the room's name field
        name = room_data.get("name")
        if name and not name.startswith("Room "):  # Avoid generic Room names
            return name

        # Return room type as default - shows the program purpose
        return room_type

    def _draw_room_2d(
        self,
        ax,
        room_id: int,
        room_data: Dict[str, Any],
        show_labels: bool,
        highlighted: bool,
    ):
        """Draw a 2D room on the floor plan."""
        x, y, _ = room_data["position"]
        w, l, _ = room_data["dimensions"]

        # Get styling
        face_color, edge_color, alpha, linewidth = self._get_room_style(
            room_data, highlighted
        )

        # Draw room rectangle
        rect = plt.Rectangle(
            (x, y),
            w,
            l,
            facecolor=face_color,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)

        # Add label to center of room
        if show_labels:
            self._add_room_label_2d(ax, room_id, room_data)

    def _add_room_label_2d(self, ax, room_id: int, room_data: Dict[str, Any]):
        """
        Add a label to a 2D room.

        Args:
            ax: Matplotlib axis
            room_id: Room ID
            room_data: Room data
        """
        x, y, _ = room_data["position"]
        w, l, _ = room_data["dimensions"]

        # Calculate center position
        center_x = x + w / 2
        center_y = y + l / 2

        # Get appropriate room name
        room_name = self._get_room_display_name(room_id, room_data)

        # Add the label
        ax.text(
            center_x,
            center_y,
            room_name,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )

    def _draw_structural_grid(self, ax):
        """Draw the structural grid on the floor plan."""
        # Get structural grid spacing from building config
        grid_x = self.building_config.get("structural_grid_x", 8.0)
        grid_y = self.building_config.get("structural_grid_y", 8.0)

        # Draw vertical grid lines
        for x in np.arange(0, self.layout.width + 0.1, grid_x):
            ax.axvline(x=x, color="gray", linestyle="--", alpha=0.3)

        # Draw horizontal grid lines
        for y in np.arange(0, self.layout.length + 0.1, grid_y):
            ax.axhline(y=y, color="gray", linestyle="--", alpha=0.3)

    def _set_aspect_equal_3d(self, ax):
        """Set equal aspect ratio for 3D plots."""
        # Get axis limits
        x_lim = ax.get_xlim3d()
        y_lim = ax.get_ylim3d()
        z_lim = ax.get_zlim3d()

        # Calculate range of each axis
        x_range = abs(x_lim[1] - x_lim[0])
        y_range = abs(y_lim[1] - y_lim[0])
        z_range = abs(z_lim[1] - z_lim[0])

        # Find the biggest range
        max_range = max(x_range, y_range, z_range)

        # Compute mid-point
        x_mid = np.mean(x_lim)
        y_mid = np.mean(y_lim)
        z_mid = np.mean(z_lim)

        # Set new limits based on mid-point and max_range
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    def create_room_legend(self, ax=None, fig=None):
        """Create a legend for room types."""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # Create legend handles
        handles = []
        labels = []

        for room_type, color in self.room_colors.items():
            if room_type == "default":
                continue

            patch = plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                alpha=self.room_alphas.get(room_type, 0.5),
                edgecolor="black",
                linewidth=0.5,
            )
            handles.append(patch)
            labels.append(room_type.replace("_", " ").title())

        # Add legend to the axes
        ax.legend(handles, labels, loc="center")

        # Hide axes
        ax.axis("off")

        return fig, ax


# Export functions separated from the renderer class
def export_to_obj(layout: SpatialGrid, filename: str):
    """
    Export the layout to OBJ format for use in 3D modeling software.

    Args:
        layout: The spatial grid layout to export
        filename: Output OBJ filename
    """
    with open(filename, "w") as f:
        # Write header
        f.write("# Hotel Layout OBJ file\n")
        f.write(f"# Generated from SpatialGrid\n")
        f.write(
            f"# Width: {layout.width}, Length: {layout.length}, Height: {layout.height}\n\n"
        )

        # Track vertex indices
        vertex_index = 1

        # Write rooms
        for room_id, room_data in layout.rooms.items():
            x, y, z = room_data["position"]
            w, l, h = room_data["dimensions"]
            room_type = room_data["type"]

            # Add comment for room
            f.write(f"# Room {room_id}: {room_type}\n")

            # Define vertices for this room
            vertices = [
                [x, y, z],  # 0
                [x + w, y, z],  # 1
                [x + w, y + l, z],  # 2
                [x, y + l, z],  # 3
                [x, y, z + h],  # 4
                [x + w, y, z + h],  # 5
                [x + w, y + l, z + h],  # 6
                [x, y + l, z + h],  # 7
            ]

            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces (with proper vertex indexing)
            f.write(
                f"f {vertex_index} {vertex_index+1} {vertex_index+2} {vertex_index+3}\n"
            )  # Bottom
            f.write(
                f"f {vertex_index+4} {vertex_index+5} {vertex_index+6} {vertex_index+7}\n"
            )  # Top
            f.write(
                f"f {vertex_index} {vertex_index+1} {vertex_index+5} {vertex_index+4}\n"
            )  # Front
            f.write(
                f"f {vertex_index+2} {vertex_index+3} {vertex_index+7} {vertex_index+6}\n"
            )  # Back
            f.write(
                f"f {vertex_index+1} {vertex_index+2} {vertex_index+6} {vertex_index+5}\n"
            )  # Right
            f.write(
                f"f {vertex_index} {vertex_index+3} {vertex_index+7} {vertex_index+4}\n"
            )  # Left

            # Update vertex index for next room
            vertex_index += 8

            f.write("\n")


def export_to_gltf(layout: SpatialGrid, filename: str):
    """
    Export the layout to glTF format for web visualization.
    This requires the pygltflib library.

    Args:
        layout: The spatial grid layout to export
        filename: Output glTF filename
    """
    try:
        import pygltflib
        from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor
        from pygltflib import AccessorType, ComponentType, BufferTarget, PrimitiveMode
    except ImportError:
        print(
            "pygltflib is required for glTF export. Install with 'pip install pygltflib'"
        )
        return

    # Note: Full implementation of glTF export would be complex
    # This placeholder would need to be expanded for proper implementation
    print("glTF export not yet implemented")
