import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Set, Any

from hotel_design_ai.core.spatial_grid import SpatialGrid


class LayoutRenderer:
    """
    3D visualization renderer for hotel layouts.
    """
    
    def __init__(self, layout: SpatialGrid):
        """
        Initialize the renderer.
        
        Args:
            layout: The spatial grid layout to render
        """
        self.layout = layout
        
        # Room type to color mapping
        self.room_colors = {
            'entrance': '#2c7fb8',
            'lobby': '#7fcdbb',
            'vertical_circulation': '#c7e9b4',
            'restaurant': '#f0f9e8',
            'meeting_room': '#edf8b1',
            'guest_room': '#f7fcb9',
            'service_area': '#d9f0a3',
            'back_of_house': '#addd8e',
            # Default color for unknown types
            'default': '#efefef'
        }
        
        # Room type to alpha (transparency) mapping
        self.room_alphas = {
            'entrance': 0.9,
            'lobby': 0.7,
            'vertical_circulation': 0.8,
            'restaurant': 0.7,
            'meeting_room': 0.7,
            'guest_room': 0.6,
            'service_area': 0.5,
            'back_of_house': 0.5,
            'default': 0.5
        }
    
    def render_3d(
        self, 
        ax=None, 
        fig=None,
        show_labels: bool = True,
        highlight_rooms: List[int] = None
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
            ax = fig.add_subplot(111, projection='3d')
        
        # Set up the axes
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Length (m)')
        ax.set_zlabel('Height (m)')
        
        # Set axis limits
        ax.set_xlim(0, self.layout.width)
        ax.set_ylim(0, self.layout.length)
        ax.set_zlim(0, self.layout.height)
        
        # Draw bounding box
        self._draw_bounding_box(ax)
        
        # Draw rooms
        for room_id, room_data in self.layout.rooms.items():
            is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
            self._draw_room(ax, room_id, room_data, show_labels, is_highlighted)
        
        # Adjust view for better visibility
        ax.view_init(elev=30, azim=-45)
        
        # Set equal aspect ratio (making a cube look like a cube)
        self._set_aspect_equal_3d(ax)
        
        return fig, ax
    
    def render_floor_plan(
        self,
        floor: int = 0,
        ax=None,
        fig=None,
        show_labels: bool = True,
        highlight_rooms: List[int] = None
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
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Length (m)')
        ax.set_title(f'Floor Plan - Level {floor}')
        
        # Set axis limits
        ax.set_xlim(0, self.layout.width)
        ax.set_ylim(0, self.layout.length)
        
        # Calculate floor height (assuming typical floor height)
        floor_height = 4.0
        z_min = floor * floor_height
        z_max = (floor + 1) * floor_height
        
        # Draw rooms on this floor
        for room_id, room_data in self.layout.rooms.items():
            x, y, z = room_data['position']
            w, l, h = room_data['dimensions']
            
            # Check if room is on this floor
            if z < z_max and z + h > z_min:
                is_highlighted = highlight_rooms is not None and room_id in highlight_rooms
                self._draw_room_2d(ax, room_id, room_data, show_labels, is_highlighted)
        
        # Draw structural grid
        self._draw_structural_grid(ax)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        return fig, ax
    
    def save_renders(
        self, 
        output_dir: str = "renders", 
        prefix: str = "layout",
        include_3d: bool = True,
        include_floor_plans: bool = True,
        num_floors: int = None
    ):
        """
        Save renders to disk.
        
        Args:
            output_dir: Directory to save renders in
            prefix: Filename prefix
            include_3d: Whether to include 3D render
            include_floor_plans: Whether to include floor plans
            num_floors: Number of floors to render (if None, detect automatically)
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save 3D render
        if include_3d:
            fig, ax = self.render_3d()
            filename = os.path.join(output_dir, f"{prefix}_3d.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved 3D render to {filename}")
        
        # Save floor plans
        if include_floor_plans:
            # Determine number of floors if not specified
            if num_floors is None:
                max_z = 0
                for _, room_data in self.layout.rooms.items():
                    _, _, z = room_data['position']
                    _, _, h = room_data['dimensions']
                    max_z = max(max_z, z + h)
                
                num_floors = int(np.ceil(max_z / 4.0))  # Assuming 4m floor height
            
            # Render each floor
            for floor in range(num_floors):
                fig, ax = self.render_floor_plan(floor=floor)
                filename = os.path.join(output_dir, f"{prefix}_floor{floor}.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved floor plan for level {floor} to {filename}")
    
    def _draw_bounding_box(self, ax):
        """Draw the building bounding box"""
        # Vertices of the bounding box
        vertices = [
            [0, 0, 0],
            [self.layout.width, 0, 0],
            [self.layout.width, self.layout.length, 0],
            [0, self.layout.length, 0],
            [0, 0, self.layout.height],
            [self.layout.width, 0, self.layout.height],
            [self.layout.width, self.layout.length, self.layout.height],
            [0, self.layout.length, self.layout.height]
        ]
        
        # Faces of the bounding box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left
        ]
        
        # Draw faces with very light gray, slightly transparent
        box = Poly3DCollection(faces, alpha=0.1, facecolor='lightgray', edgecolor='gray', linewidth=0.5)
        ax.add_collection3d(box)
    
    def _draw_room(
        self, 
        ax, 
        room_id: int, 
        room_data: Dict[str, Any],
        show_labels: bool,
        highlighted: bool
    ):
        """Draw a 3D room"""
        x, y, z = room_data['position']
        w, l, h = room_data['dimensions']
        room_type = room_data['type']
        
        # Vertices of the room
        vertices = [
            [x, y, z],
            [x + w, y, z],
            [x + w, y + l, z],
            [x, y + l, z],
            [x, y, z + h],
            [x + w, y, z + h],
            [x + w, y + l, z + h],
            [x, y + l, z + h]
        ]
        
        # Faces of the room
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left
        ]
        
        # Get room color
        color = self.room_colors.get(room_type, self.room_colors['default'])
        
        # Adjust color and transparency for highlighting
        if highlighted:
            color = self._brighten_color(color)
            alpha = 0.9
            edgecolor = 'red'
            linewidth = 2
        else:
            alpha = self.room_alphas.get(room_type, self.room_alphas['default'])
            edgecolor = 'black'
            linewidth = 0.5
        
        # Draw room
        room_poly = Poly3DCollection(
            faces, 
            alpha=alpha, 
            facecolor=color, 
            edgecolor=edgecolor, 
            linewidth=linewidth
        )
        ax.add_collection3d(room_poly)
        
        # Add label to center of room
        if show_labels:
            center_x = x + w / 2
            center_y = y + l / 2
            center_z = z + h / 2
            
            room_name = room_data.get('name', f"Room {room_id}")
            
            # If room has a long name, use room_type instead
            if len(room_name) > 15:
                room_name = room_type.capitalize()
            
            ax.text(
                center_x, center_y, center_z, 
                room_name,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8
            )
    
    def _draw_room_2d(
        self, 
        ax, 
        room_id: int, 
        room_data: Dict[str, Any],
        show_labels: bool,
        highlighted: bool
    ):
        """Draw a 2D room on the floor plan"""
        x, y, _ = room_data['position']
        w, l, _ = room_data['dimensions']
        room_type = room_data['type']
        
        # Get room color
        color = self.room_colors.get(room_type, self.room_colors['default'])
        
        # Adjust color and transparency for highlighting
        if highlighted:
            color = self._brighten_color(color)
            alpha = 0.9
            edgecolor = 'red'
            linewidth = 2
        else:
            alpha = self.room_alphas.get(room_type, self.room_alphas['default'])
            edgecolor = 'black'
            linewidth = 0.5
        
        # Draw room rectangle
        rect = plt.Rectangle(
            (x, y), w, l, 
            facecolor=color, 
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth
        )
        ax.add_patch(rect)
        
        # Add label to center of room
        if show_labels:
            center_x = x + w / 2
            center_y = y + l / 2
            
            room_name = room_data.get('name', f"Room {room_id}")
            
            # If room has a long name, use room_type instead
            if len(room_name) > 15:
                room_name = room_type.capitalize()
            
            ax.text(
                center_x, center_y, 
                room_name,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8
            )
    
    def _draw_structural_grid(self, ax):
        """Draw the structural grid on the floor plan"""
        # Get structural grid spacing (default to 8m if not available)
        grid_x = 8.0
        grid_y = 8.0
        
        # Draw vertical grid lines
        for x in np.arange(0, self.layout.width + 0.1, grid_x):
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
        
        # Draw horizontal grid lines
        for y in np.arange(0, self.layout.length + 0.1, grid_y):
            ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    
    def _set_aspect_equal_3d(self, ax):
        """Set equal aspect ratio for 3D plots"""
        # Get axis limits
        x_lim = ax.get_xlim3d()
        y_lim = ax.get_ylim3d()
        z_lim = ax.get_zlim3d()
        
        # Calculate range of each axis
        x_range = abs(x_lim[1] - x_lim[0])
        y_range = abs(y_lim[1] - y_lim[0])
        z_range = abs(z_lim[1] - z_lim[0])
        
        # Find the biggest range to create a cube
        max_range = max(x_range, y_range, z_range)
        
        # Compute mid-point
        x_mid = np.mean(x_lim)
        y_mid = np.mean(y_lim)
        z_mid = np.mean(z_lim)
        
        # Set new limits based on mid-point and max_range
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
    
    def _brighten_color(self, color_str: str) -> str:
        """Brighten a color for highlighting"""
        rgb = mcolors.to_rgb(color_str)
        # Make color brighter (closer to white)
        brightened = [min(1.0, c * 1.5) for c in rgb]
        return mcolors.rgb2hex(brightened)
    
    def create_room_legend(self, ax=None, fig=None):
        """Create a legend for room types"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create legend handles
        handles = []
        labels = []
        
        for room_type, color in self.room_colors.items():
            if room_type == 'default':
                continue
                
            patch = plt.Rectangle(
                (0, 0), 1, 1, 
                facecolor=color, 
                alpha=self.room_alphas.get(room_type, 0.5),
                edgecolor='black',
                linewidth=0.5
            )
            handles.append(patch)
            labels.append(room_type.replace('_', ' ').title())
        
        # Add legend to the axes
        ax.legend(handles, labels, loc='center')
        
        # Hide axes
        ax.axis('off')
        
        return fig, ax


def export_to_obj(layout: SpatialGrid, filename: str):
    """
    Export the layout to OBJ format for use in 3D modeling software.
    
    Args:
        layout: The spatial grid layout to export
        filename: Output OBJ filename
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# Hotel Layout OBJ file\n")
        f.write(f"# Generated from SpatialGrid\n")
        f.write(f"# Width: {layout.width}, Length: {layout.length}, Height: {layout.height}\n\n")
        
        # Track vertex indices
        vertex_index = 1
        
        # Write rooms
        for room_id, room_data in layout.rooms.items():
            x, y, z = room_data['position']
            w, l, h = room_data['dimensions']
            room_type = room_data['type']
            
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
                [x, y + l, z + h]  # 7
            ]
            
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # Write faces (with proper vertex indexing)
            f.write(f"f {vertex_index} {vertex_index+1} {vertex_index+2} {vertex_index+3}\n")  # Bottom
            f.write(f"f {vertex_index+4} {vertex_index+5} {vertex_index+6} {vertex_index+7}\n")  # Top
            f.write(f"f {vertex_index} {vertex_index+1} {vertex_index+5} {vertex_index+4}\n")  # Front
            f.write(f"f {vertex_index+2} {vertex_index+3} {vertex_index+7} {vertex_index+6}\n")  # Back
            f.write(f"f {vertex_index+1} {vertex_index+2} {vertex_index+6} {vertex_index+5}\n")  # Right
            f.write(f"f {vertex_index} {vertex_index+3} {vertex_index+7} {vertex_index+4}\n")  # Left
            
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
        print("pygltflib is required for glTF export. Install with 'pip install pygltflib'")
        return
    
    # TODO: Implement glTF export
    # This is more involved and requires setting up proper buffers, accessors, etc.
    # Consider implementing this in a future version
    
    print("glTF export not yet implemented")
