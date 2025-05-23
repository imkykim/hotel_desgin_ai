import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import os


class RenderingTheme:
    """Configuration for visualization styles and colors."""

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
        "parking": "#D3D3D3",  # Light gray for parking
        "mechanical": "#A0A0A0",  # Darker gray for mechanical
        "maintenance": "#909090",  # Medium gray for maintenance
        # Default color for unknown types
        "default": "#efefef",
        "suite_room": "#FFD700",
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
        "parking": 0.5,
        "mechanical": 0.6,
        "maintenance": 0.6,
        "default": 0.5,
        "suite_room": 0.8,
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
    Enhanced to handle any floor range dynamically.
    """

    def __init__(
        self,
        layout,
        building_config: Dict[str, Any] = None,
        render_config: Dict[str, Any] = None,
    ):
        """
        Initialize the renderer.

        Args:
            layout: The spatial grid layout to render
            building_config: Building configuration parameters
            render_config: Rendering configuration parameters (optional)
        """
        self.layout = layout
        self.building_config = building_config or {"floor_height": 4.0}
        self.render_config = render_config or {}

        # Set default colors and alphas from RenderingTheme
        self.room_colors = self.render_config.get(
            "room_colors", RenderingTheme.DEFAULT_ROOM_COLORS.copy()
        )
        self.room_alphas = self.render_config.get(
            "room_alphas", RenderingTheme.DEFAULT_ROOM_ALPHAS.copy()
        )

        # Determine floor range from layout and building_config
        self._init_floor_range()

    def _init_floor_range(self):
        """
        Determine the floor range by analyzing the layout and building_config.
        Add this method to your renderer class.
        """
        # Start with defaults from building_config
        self.min_floor = self.building_config.get("min_floor", -1)
        self.max_floor = self.building_config.get("max_floor", 3)
        self.floor_height = self.building_config.get("floor_height", 4.0)

        # Extract standard floor range
        std_floor_config = self.building_config.get("standard_floor", {})
        self.std_floor_start = std_floor_config.get("start_floor", 5)
        self.std_floor_end = std_floor_config.get("end_floor", 20)

        # Extract podium floor range
        podium_config = self.building_config.get("podium", {})
        self.podium_min = podium_config.get("min_floor", self.min_floor)
        self.podium_max = podium_config.get("max_floor", 4)

        # Check for floor range consistency
        if self.std_floor_start <= self.podium_max:
            print(
                f"Warning: Standard floor start ({self.std_floor_start}) overlaps with podium max ({self.podium_max})"
            )
            self.podium_max = self.std_floor_start - 1

        # Validate floor range
        if self.min_floor > self.max_floor:
            print(
                f"Warning: min_floor ({self.min_floor}) > max_floor ({self.max_floor}). Setting min_floor = max_floor."
            )
            self.min_floor = self.max_floor

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
        self._draw_all_rooms_3d(ax, show_labels, highlight_rooms)

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

        # Calculate z limits based on floor range
        min_z = self.min_floor * self.floor_height
        max_z = (self.max_floor + 1) * self.floor_height

        # Set axis limits to include all floors
        ax.set_xlim(0, self.layout.width)
        ax.set_ylim(0, self.layout.length)
        ax.set_zlim(min_z, max_z)

        # Add a grid at z=0 to delineate ground level
        xx, yy = np.meshgrid(
            np.linspace(0, self.layout.width, 5), np.linspace(0, self.layout.length, 5)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color="gray")

    def _draw_all_rooms_3d(
        self, ax, show_labels: bool, highlight_rooms: List[int] = None
    ):
        """Draw all rooms in 3D."""
        for room_id, room_data in self.layout.rooms.items():
            is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
            self._draw_room(
                ax, room_id, room_data, show_labels, is_highlighted, is_3d=True
            )

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
        Simplified to just show rooms that exist - no template rendering.

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

        # Draw rooms on this floor (only those that exist)
        self._draw_rooms_on_floor(ax, floor, show_labels, highlight_rooms)

        # Draw structural grid
        self._draw_structural_grid(ax)

        self._draw_standard_floor_boundary(ax, self.building_config)

        # Add floor type annotation
        self._add_floor_type_annotation(ax, floor)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        return fig, ax

    def _setup_2d_axes(self, ax, floor: int):
        """Set up the 2D axes with proper labels and limits."""
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Length (m)")

        # Determine a more descriptive floor name
        if floor == 0:
            floor_name = "Ground Floor"
        elif floor < 0:
            floor_name = f"Basement {abs(floor)}"
        else:
            floor_name = f"Floor {floor}"

        ax.set_title(f"Floor Plan - {floor_name}")

        # Set axis limits
        ax.set_xlim(0, self.layout.width)
        ax.set_ylim(self.layout.length, 0)

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
        # Calculate minimum z based on min_floor
        min_z = self.min_floor * self.floor_height

        # Vertices of the bounding box, including basement if needed
        vertices = [
            [0, 0, min_z],  # Bottom left at lowest level
            [self.layout.width, 0, min_z],  # Bottom right at lowest level
            [self.layout.width, self.layout.length, min_z],  # Top right at lowest level
            [0, self.layout.length, min_z],  # Top left at lowest level
            [0, 0, self.layout.height],  # Top points remain the same
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
        is_3d: bool = True,
        is_circulation: bool = False,
    ):
        """
        Draw a room in either 2D or 3D.

        Args:
            ax: Matplotlib axis
            room_id: Room ID
            room_data: Room data
            show_labels: Whether to show labels
            highlighted: Whether the room is highlighted
            is_3d: Whether rendering in 3D
            is_circulation: Whether this is a circulation element
        """
        # Get styling
        style = self._get_room_style(room_data, highlighted, is_3d, is_circulation)

        # Draw room
        if is_3d:
            self._draw_room_3d(ax, room_data, style)
        else:
            self._draw_room_2d(ax, room_data, style)

        # Add label
        if show_labels:
            self._add_room_label(ax, room_id, room_data, is_3d)

    def _draw_room_3d(self, ax, room_data: Dict[str, Any], style: Dict[str, Any]):
        """Draw a 3D room."""
        # Get room geometry
        vertices, faces = self._get_room_geometry(room_data)

        # Draw room
        room_poly = Poly3DCollection(
            faces,
            alpha=style["alpha"],
            facecolor=style["face_color"],
            edgecolor=style["edge_color"],
            linewidth=style["linewidth"],
        )
        ax.add_collection3d(room_poly)

    def _draw_room_2d(self, ax, room_data: Dict[str, Any], style: Dict[str, Any]):
        """Draw a 2D room on the floor plan."""
        x, y, _ = room_data["position"]
        w, l, _ = room_data["dimensions"]

        # Check if this room overlaps with others
        has_overlaps = "overlaps_with" in room_data and room_data["overlaps_with"]

        # If room has overlaps, use a special hatch pattern to indicate it
        if has_overlaps:
            # Highlighted edge to indicate overlap
            rect = plt.Rectangle(
                (x, y),
                w,
                l,
                facecolor=style["face_color"],
                alpha=style["alpha"],
                edgecolor=style["edge_color"],
                # edgecolor="red",
                linewidth=style["linewidth"],
                # hatch="//",  # Add hatch pattern to indicate overlap
            )
        else:
            # Regular room rectangle
            rect = plt.Rectangle(
                (x, y),
                w,
                l,
                facecolor=style["face_color"],
                alpha=style["alpha"],
                edgecolor=style["edge_color"],
                linewidth=style["linewidth"],
            )

        ax.add_patch(rect)
        # if room_data["type"] == "lobby":
        #     # Determine which side of the building this entrance is on
        #     is_front = abs(y) < 0.1
        #     is_back = abs(y + l - self.layout.length) < 0.1
        #     is_left = abs(x) < 0.1
        #     is_right = abs(x + w - self.layout.width) < 0.1

        #     # Calculate arrow parameters
        #     center_x = x + w / 2
        #     center_y = y + l / 2
        #     arrow_length = max(w, l) * 0.3  # Arrow length depends on lobby size

        #     if is_front:
        #         # Front entrance (bottom of building)
        #         ax.annotate(
        #             "MAIN ENTRANCE",
        #             xy=(center_x, y),
        #             xytext=(center_x, y - arrow_length),
        #             arrowprops=dict(arrowstyle="->", color="red", lw=2),
        #             ha="center",
        #             va="top",
        #             fontsize=10,
        #             fontweight="bold",
        #             color="red",
        #         )

        #     elif is_back:
        #         # Back entrance (top of building)
        #         ax.annotate(
        #             "MAIN ENTRANCE",
        #             xy=(center_x, y + l),
        #             xytext=(center_x, y + l + arrow_length),
        #             arrowprops=dict(arrowstyle="->", color="red", lw=2),
        #             ha="center",
        #             va="bottom",
        #             fontsize=10,
        #             fontweight="bold",
        #             color="red",
        #         )

        #     elif is_left:
        #         # Left entrance
        #         ax.annotate(
        #             "MAIN ENTRANCE",
        #             xy=(x, center_y),
        #             xytext=(x - arrow_length, center_y),
        #             arrowprops=dict(arrowstyle="->", color="red", lw=2),
        #             ha="right",
        #             va="center",
        #             fontsize=10,
        #             fontweight="bold",
        #             color="red",
        #         )

        #     elif is_right:
        #         # Right entrance
        #         ax.annotate(
        #             "MAIN ENTRANCE",
        #             xy=(x + w, center_y),
        #             xytext=(x + w + arrow_length, center_y),
        #             arrowprops=dict(arrowstyle="->", color="red", lw=2),
        #             ha="left",
        #             va="center",
        #             fontsize=10,
        #             fontweight="bold",
        #             color="red",
        #         )

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

    def _get_room_style(
        self,
        room_data: Dict[str, Any],
        highlighted: bool,
        is_3d: bool = True,
        is_circulation: bool = False,
    ):
        """
        Get styling information for a room.

        Args:
            room_data: Room data
            highlighted: Whether the room is highlighted
            is_3d: Whether rendering in 3D
            is_circulation: Whether this is a circulation element

        Returns:
            Dict: Styling information
        """
        room_type = room_data["type"]

        # Get room color and alpha
        face_color = self.room_colors.get(room_type, self.room_colors["default"])
        alpha = self.room_alphas.get(room_type, self.room_alphas["default"])

        # Adjust for highlighting
        if highlighted:
            face_color = RenderingTheme.brighten_color(face_color)
            alpha = 0.9
            edge_color = "red"
            linewidth = 2
        else:
            edge_color = "black"
            linewidth = 0.5

        # Special adjustment for vertical circulation
        if room_type == "vertical_circulation" or is_circulation:
            # Make vertical circulation elements more visible
            alpha = min(1.0, alpha + 0.3)
            linewidth = 2.0
            edge_color = "red"

            # If it's specifically a circulation element on a floor plan
            if is_circulation:
                # Use an even more pronounced style
                alpha = 0.9
                linewidth = 2.5

        return {
            "face_color": face_color,
            "edge_color": edge_color,
            "alpha": alpha,
            "linewidth": linewidth,
        }

    def _add_room_label(
        self, ax, room_id: int, room_data: Dict[str, Any], is_3d: bool = True
    ):
        """
        Add a label to a room.

        Args:
            ax: Matplotlib axis
            room_id: Room ID
            room_data: Room data
            is_3d: Whether rendering in 3D
        """
        x, y, z = room_data["position"]
        w, l, h = room_data["dimensions"]

        # Calculate center position
        center_x = x + w / 2
        center_y = y + l / 2
        center_z = z + h / 2 if is_3d else 0

        # Get room name
        room_name = self._get_room_display_name(room_id, room_data)

        # Add the label
        if is_3d:
            ax.text(
                center_x,
                center_y,
                center_z,
                room_name,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
            )
        else:
            ax.text(
                center_x,
                center_y,
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
        if isinstance(metadata, dict) and "original_name" in metadata:
            return metadata["original_name"]

        # Try room type as it shows the program purpose (preferred for visualization)
        room_type = room_data["type"].replace("_", " ").title()

        # Then try to use the room's name field
        name = room_data.get("name")
        if name and not name.startswith("Room "):  # Avoid generic Room names
            return name

        # Return room type as default - shows the program purpose
        return room_type

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
        z_min = floor * self.floor_height
        z_max = (floor + 1) * self.floor_height

        # Get rooms for this floor
        rooms_on_floor = self._get_rooms_on_floor(floor, z_min, z_max)

        # Draw regular rooms first
        for room_id in rooms_on_floor["regular"]:
            room_data = self.layout.rooms[room_id]
            is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
            self._draw_room(
                ax, room_id, room_data, show_labels, is_highlighted, is_3d=False
            )

        # Draw vertical circulation on top with enhanced visibility
        for room_id in rooms_on_floor["circulation"]:
            room_data = self.layout.rooms[room_id]
            is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
            self._draw_room(
                ax,
                room_id,
                room_data,
                show_labels,
                is_highlighted,
                is_3d=False,
                is_circulation=True,
            )

    def _get_rooms_on_floor(
        self, floor: int, z_min: float, z_max: float
    ) -> Dict[str, List[int]]:
        """
        Get all room IDs for rooms on a specific floor, categorized by type.

        Args:
            floor: Floor number
            z_min: Minimum z coordinate for the floor
            z_max: Maximum z coordinate for the floor

        Returns:
            Dict with 'regular' and 'circulation' lists of room IDs
        """
        regular_rooms = []
        circulation_rooms = []

        for room_id, room_data in self.layout.rooms.items():
            # Special handling for vertical circulation - make sure it shows on all floors it spans
            if room_data["type"] == "vertical_circulation":
                # Check if this vertical circulation spans this floor
                x, y, z = room_data["position"]
                w, l, h = room_data["dimensions"]

                # Vertical circulation appears on any floor it passes through
                if (z <= z_min and z + h >= z_max) or (z >= z_min and z < z_max):
                    circulation_rooms.append(room_id)
                    continue

            # Check if regular room is on this floor
            x, y, z = room_data["position"]
            w, l, h = room_data["dimensions"]

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
                regular_rooms.append(room_id)

        return {"regular": regular_rooms, "circulation": circulation_rooms}

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

    def save_renders(
        self,
        output_dir: str = "renders",
        prefix: str = "layout",
        include_3d: bool = True,
        include_floor_plans: bool = True,
        num_floors: int = None,
        min_floor: int = None,
        sample_standard: bool = True,
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
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save 3D render (two angles)
        if include_3d:
            # Default view (elev=30, azim=-45)
            self._save_3d_render(output_dir, prefix, suffix="_3d", view=(30, 135))
            # Opposite view (elev=30, azim=135)
            self._save_3d_render(
                output_dir, prefix, suffix="_3d_opposite", view=(30, -45)
            )

        # Save floor plans
        if include_floor_plans:
            self._save_floor_plans(
                output_dir, prefix, num_floors, min_floor, sample_standard
            )

    def _save_3d_render(
        self, output_dir: str, prefix: str, suffix: str = "_3d", view: tuple = (30, -45)
    ):
        """
        Save 3D render to disk.

        Args:
            output_dir: Directory to save renders in
            prefix: Filename prefix
            suffix: Suffix for the filename (e.g., "_3d" or "_3d_opposite")
            view: Tuple of (elev, azim) for the 3D view angle
        """
        fig, ax = self.render_3d()
        # Set the requested view
        ax.view_init(elev=view[0], azim=view[1])
        filename = os.path.join(output_dir, f"{prefix}{suffix}.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_floor_plans(
        self,
        output_dir: str,
        prefix: str,
        num_floors: Optional[int] = None,
        min_floor: Optional[int] = None,
        sample_standard: bool = True,
    ):
        """
        Save floor plans to disk.
        Enhanced to handle standard floors and only render occupied floors.
        """
        # Initialize floor ranges if needed
        if not hasattr(self, "std_floor_start"):
            self._init_floor_range()

        # Always start by getting occupied floors using the intelligent selection
        # This will include at least one standard floor even if it's not occupied
        floors_to_render = self.get_floors_to_render(sample_standard)

        # If no floors are found, don't try to render anything
        if not floors_to_render:
            print("No floors found to render")
            return

        # If num_floors and min_floor are specified, filter the floors to that range
        # but make sure to keep at least one standard floor
        if num_floors is not None and min_floor is not None:
            max_floor = min_floor + num_floors - 1
            range_floors = set(range(min_floor, max_floor + 1))

            # First get podium floors in range
            podium_floors = [
                f
                for f in floors_to_render
                if f in range_floors and not self.is_standard_floor(f)
            ]

            # Then add one standard floor if in range
            std_floors = [f for f in floors_to_render if self.is_standard_floor(f)]
            if std_floors and any(f in range_floors for f in std_floors):
                # Add the first standard floor in range
                for f in std_floors:
                    if f in range_floors:
                        podium_floors.append(f)
                        break
            elif sample_standard and self.std_floor_start in range_floors:
                # Add the start standard floor if in range
                podium_floors.append(self.std_floor_start)

            floors_to_render = sorted(podium_floors)

        print(f"Rendering floor plans for floors: {floors_to_render}")

        # Render each floor
        for floor in floors_to_render:
            fig, ax = self.render_floor_plan(floor=floor)

            # Generate appropriate name
            if floor < 0:
                floor_name = f"basement{abs(floor)}"
            elif self.is_standard_floor(floor):
                floor_name = f"standard_floor"
                if len([f for f in floors_to_render if self.is_standard_floor(f)]) > 1:
                    # If rendering multiple standard floors, add floor number
                    floor_name += f"_{floor}"
            else:
                floor_name = f"floor{floor}"

            filename = os.path.join(output_dir, f"{prefix}_{floor_name}.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

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

    def is_standard_floor(self, floor):
        """
        Check if a floor is a standard floor.
        Add this method to your renderer class.

        Args:
            floor: Floor number

        Returns:
            bool: True if it's a standard floor
        """
        if not hasattr(self, "std_floor_start"):
            # Initialize floor ranges if not done yet
            self._init_floor_range()

        return self.std_floor_start <= floor <= self.std_floor_end

    def is_podium_floor(self, floor):
        """
        Check if a floor is a podium floor.
        Add this method to your renderer class.

        Args:
            floor: Floor number

        Returns:
            bool: True if it's a podium floor
        """
        if not hasattr(self, "podium_min"):
            # Initialize floor ranges if not done yet
            self._init_floor_range()

        return self.podium_min <= floor <= self.podium_max

    def get_floors_to_render(self, sample_standard: bool = True):
        """
        Get the list of floors that should be rendered.
        Enhanced to always include at least one standard floor for rendering.

        Args:
            sample_standard: If True, only include one sample standard floor

        Returns:
            List of floor numbers to render
        """
        if not hasattr(self, "std_floor_start"):
            # Initialize floor ranges if not done yet
            self._init_floor_range()

        # Find floors that actually have rooms in the layout
        occupied_floors = set()
        for room_id, room_data in self.layout.rooms.items():
            z = room_data["position"][2]
            floor = int(z / self.floor_height)
            occupied_floors.add(floor)

        # Always include all podium floors
        podium_floors = [
            f for f in occupied_floors if self.podium_min <= f <= self.podium_max
        ]

        # For standard floors, either include all or just a sample
        standard_floors = [
            f
            for f in occupied_floors
            if self.std_floor_start <= f <= self.std_floor_end
        ]

        # If no standard floors are detected but we want to show a sample,
        # add the first standard floor even if it's not occupied
        if sample_standard and not standard_floors:
            standard_floors = [self.std_floor_start]

        # Combine and sort
        floors_to_render = sorted(podium_floors + standard_floors)

        return floors_to_render

    # This function is an extension of the existing _save_floor_plans method
    # You may need to adapt it to match your exact implementation

    # This function adds a floor type annotation to the rendered floor plan
    # It can be called from your render_floor_plan method
    def _add_floor_type_annotation(self, ax, floor):
        """
        Add floor type annotation to the floor plan.
        Call this from your render_floor_plan method to show floor type.

        Args:
            ax: Matplotlib axis
            floor: Floor number
        """
        if not hasattr(self, "std_floor_start"):
            self._init_floor_range()

        floor_type = (
            "Standard Floor" if self.is_standard_floor(floor) else "Podium Floor"
        )
        floor_position = (
            "Tower Section" if self.is_standard_floor(floor) else "裙房 Section"
        )
        ax.annotate(
            f"{floor_type} (Floor {floor})\n{floor_position}",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8
            ),
        )

    def _draw_standard_floor_boundary(self, ax, building_config):
        """
        Draw the outline of the standard floor boundary on the floor plan.

        Args:
            ax: Matplotlib axis
            building_config: Building configuration dictionary
        """
        std_floor_config = building_config.get("standard_floor", {})
        if not std_floor_config:
            print("No standard floor configuration found.")
            return

        # Extract boundary dimensions and position
        width = std_floor_config.get("width", 0.0)
        length = std_floor_config.get("length", 0.0)
        position_x = std_floor_config.get("position_x", 0.0)
        position_y = std_floor_config.get("position_y", 0.0)

        # Draw rectangle representing the boundary
        rect = plt.Rectangle(
            (position_x, position_y),  # Bottom-left corner
            width,  # Width of the rectangle
            length,  # Height of the rectangle
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            linestyle="--",
            label="Standard Floor Boundary",
        )
        ax.add_patch(rect)
        ax.legend()
