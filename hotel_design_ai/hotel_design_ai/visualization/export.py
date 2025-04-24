import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional

from hotel_design_ai.core.spatial_grid import SpatialGrid


def export_to_json(layout: SpatialGrid, filename: str):
    """
    Export the layout to JSON format.

    Args:
        layout: The spatial grid layout to export
        filename: Output JSON filename
    """
    # Convert the layout to a dictionary
    layout_dict = layout.to_dict()

    # Write to file
    with open(filename, "w") as f:
        json.dump(layout_dict, f, indent=2)


def export_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Export layout metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the JSON file
    """
    # Ensure directory exists
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to file
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def export_to_csv(layout: SpatialGrid, filename: str):
    """
    Export the room data to CSV format.

    Args:
        layout: The spatial grid layout to export
        filename: Output CSV filename
    """
    room_data = []

    # Extract room data
    for room_id, room in layout.rooms.items():
        x, y, z = room["position"]
        w, l, h = room["dimensions"]

        data = {
            "room_id": room_id,
            "room_type": room["type"],
            "name": room.get("name", f"Room {room_id}"),
            "x": x,
            "y": y,
            "z": z,
            "width": w,
            "length": l,
            "height": h,
            "volume": w * l * h,
            "area": w * l,
        }
        room_data.append(data)

    # Write header
    with open(filename, "w") as f:
        # Define columns
        columns = [
            "room_id",
            "room_type",
            "name",
            "x",
            "y",
            "z",
            "width",
            "length",
            "height",
            "volume",
            "area",
        ]

        # Write header
        f.write(",".join(columns) + "\n")

        # Write data rows
        for room in room_data:
            row = [str(room[col]) for col in columns]
            f.write(",".join(row) + "\n")


def export_for_three_js(layout: SpatialGrid, filename: str):
    """
    Export the layout for use with Three.js visualizations.

    Args:
        layout: The spatial grid layout to export
        filename: Output JSON filename for Three.js
    """
    # Create a Three.js friendly data structure
    three_js_data = {
        "metadata": {
            "version": 1.0,
            "type": "hotel_layout",
            "dimensions": {
                "width": layout.width,
                "length": layout.length,
                "height": layout.height,
            },
        },
        "rooms": [],
    }

    # Room type to color mapping (hex colors)
    room_colors = {
        "entrance": "#2c7fb8",
        "lobby": "#7fcdbb",
        "vertical_circulation": "#c7e9b4",
        "restaurant": "#f0f9e8",
        "meeting_room": "#edf8b1",
        "guest_room": "#f7fcb9",
        "service_area": "#d9f0a3",
        "back_of_house": "#addd8e",
        "default": "#efefef",
    }

    # Add rooms
    for room_id, room in layout.rooms.items():
        x, y, z = room["position"]
        w, l, h = room["dimensions"]
        room_type = room["type"]

        three_js_room = {
            "id": room_id,
            "type": room_type,
            "name": room.get("name", f"Room {room_id}"),
            "position": [x, z, y],  # Three.js uses Y-up, change coordinates
            "dimensions": [w, h, l],  # Three.js uses Y-up, change dimensions
            "color": room_colors.get(room_type, room_colors["default"]),
        }

        three_js_data["rooms"].append(three_js_room)

    # Write to JSON file
    with open(filename, "w") as f:
        json.dump(three_js_data, f, indent=2)


def export_to_blender(layout: SpatialGrid, filename: str):
    """
    Export the layout to a Python script that can be run in Blender to create the 3D model.

    Args:
        layout: The spatial grid layout to export
        filename: Output Python filename for Blender
    """
    # Room type to material/color mapping
    room_colors = {
        "entrance": (0.173, 0.5, 0.72, 1.0),  # RGB values (0-1)
        "lobby": (0.498, 0.804, 0.733, 1.0),
        "vertical_circulation": (0.78, 0.914, 0.706, 1.0),
        "restaurant": (0.941, 0.976, 0.91, 1.0),
        "meeting_room": (0.929, 0.973, 0.694, 1.0),
        "guest_room": (0.969, 0.988, 0.725, 1.0),
        "service_area": (0.851, 0.941, 0.639, 1.0),
        "back_of_house": (0.678, 0.867, 0.557, 1.0),
        "parking": (0.827, 0.827, 0.827, 1.0),
        "mechanical": (0.627, 0.627, 0.627, 1.0),
        "maintenance": (0.565, 0.565, 0.565, 1.0),
        "default": (0.937, 0.937, 0.937, 1.0),
    }

    # Create the Blender Python script
    with open(filename, "w") as f:
        # Write header and imports
        f.write(
            """import bpy
import bmesh
import math
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create a new collection for the hotel
hotel_collection = bpy.data.collections.new("Hotel Layout")
bpy.context.scene.collection.children.link(hotel_collection)

# Create materials for different room types
materials = {}
"""
        )

        # Create materials for each room type
        for room_type, color in room_colors.items():
            f.write(
                f"""
# Create material for {room_type}
mat_{room_type} = bpy.data.materials.new(name="{room_type.replace('_', ' ').title()}")
mat_{room_type}.use_nodes = True
nodes = mat_{room_type}.node_tree.nodes
nodes["Principled BSDF"].inputs[0].default_value = {color}  # Base color
materials["{room_type}"] = mat_{room_type}
"""
            )

        # Default material
        f.write(
            """
# Create default material
default_mat = bpy.data.materials.new(name="Default")
default_mat.use_nodes = True
nodes = default_mat.node_tree.nodes
nodes["Principled BSDF"].inputs[0].default_value = (0.937, 0.937, 0.937, 1.0)  # Gray
materials["default"] = default_mat

# Function to create a room as a box
def create_room(name, room_type, x, y, z, width, length, height):
    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    # Link object to collection
    hotel_collection.objects.link(obj)
    
    # Create mesh geometry
    bm = bmesh.new()
    
    # Create vertices
    v1 = bm.verts.new((0, 0, 0))
    v2 = bm.verts.new((width, 0, 0))
    v3 = bm.verts.new((width, length, 0))
    v4 = bm.verts.new((0, length, 0))
    v5 = bm.verts.new((0, 0, height))
    v6 = bm.verts.new((width, 0, height))
    v7 = bm.verts.new((width, length, height))
    v8 = bm.verts.new((0, length, height))
    
    # Create faces
    bm.faces.new((v1, v2, v3, v4))  # Bottom
    bm.faces.new((v5, v6, v7, v8))  # Top
    bm.faces.new((v1, v2, v6, v5))  # Front
    bm.faces.new((v2, v3, v7, v6))  # Right
    bm.faces.new((v3, v4, v8, v7))  # Back
    bm.faces.new((v4, v1, v5, v8))  # Left
    
    # Finish up mesh
    bm.to_mesh(mesh)
    bm.free()
    
    # Position the object
    obj.location = (x, y, z)
    
    # Assign material based on room type
    if room_type in materials:
        obj.data.materials.append(materials[room_type])
    else:
        obj.data.materials.append(materials["default"])
    
    # Add custom properties
    obj["room_type"] = room_type
    obj["room_id"] = name.split('_')[-1] if '_' in name else name
    
    return obj

# Create each room
"""
        )

        # Write code to create each room
        for room_id, room in layout.rooms.items():
            x, y, z = room["position"]
            w, l, h = room["dimensions"]
            room_type = room["type"]
            name = room.get("name", f"Room_{room_id}")

            # Sanitize name for Blender (no spaces, special chars)
            safe_name = name.replace(" ", "_").replace("-", "_")

            f.write(
                f'create_room("{safe_name}", "{room_type}", {x}, {y}, {z}, {w}, {l}, {h})\n'
            )

        # Final camera setup
        f.write(
            f"""
# Create a camera with a good view of the model
bpy.ops.object.camera_add(location=({layout.width/2}, -{layout.length}, {layout.height * 1.5}))
camera = bpy.context.active_object
camera.rotation_euler = (math.pi/4, 0, 0)
bpy.context.scene.camera = camera

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(0, 0, {layout.height * 2}))
sun = bpy.context.active_object
sun.rotation_euler = (math.radians(45), 0, math.radians(45))
sun.data.energy = 2.0

# Set viewport display
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

# Zoom to fit all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.view3d.camera_to_view_selected()

print("Hotel layout imported successfully!")
"""
        )

    print(f"Exported Blender Python script to {filename}")
    print(
        f"Open Blender and execute this script (File > Open > Text Editor > Open) to import the layout"
    )


def export_to_rhino(layout, output_filepath):
    """
    Export a layout to a Python script that can be run in Rhino.

    Args:
        layout: The SpatialGrid object or dictionary containing room data
        output_filepath: The path where the Rhino Python script will be saved
    """
    # Create a header for the script
    script_content = """# Hotel Layout for Rhino
# This script creates a 3D model of a hotel layout in Rhino
# To run: In Rhino, type RunPythonScript and select this file

import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
import System.Guid

# explicitly target the already-open document
sc.doc = Rhino.RhinoDoc.ActiveDoc

# (Optional) clear existing geometry in this file
# rs.DeleteObjects( rs.AllObjects() )

# Create layers for different room types
def create_layer(name, color):
    layer_name = "Hotel_" + name.replace(" ", "_")
    if not rs.IsLayer(layer_name):
        rs.AddLayer(layer_name, color)
    return layer_name

# Create layers for room types
layer_entrance = create_layer("entrance", [44, 127, 184])
layer_lobby = create_layer("lobby", [127, 205, 187])
layer_vertical_circulation = create_layer("vertical_circulation", [199, 233, 180])
layer_restaurant = create_layer("restaurant", [240, 249, 232])
layer_meeting_room = create_layer("meeting_room", [237, 248, 177])
layer_guest_room = create_layer("guest_room", [247, 252, 185])
layer_service_area = create_layer("service_area", [217, 240, 163])
layer_back_of_house = create_layer("back_of_house", [173, 221, 142])
layer_parking = create_layer("parking", [211, 211, 211])
layer_mechanical = create_layer("mechanical", [160, 160, 160])
layer_maintenance = create_layer("maintenance", [144, 144, 144])
layer_default = create_layer("default", [239, 239, 239])

# Function to create a room box
def create_room(name, room_type, x, y, z, width, length, height):
    # Get the layer for this room type
    layer_name = "Hotel_" + room_type.replace(" ", "_")
    if not rs.IsLayer(layer_name):
        layer_name = "Hotel_default"
    
    # Set current layer
    rs.CurrentLayer(layer_name)
    
    # Create the box
    corner_point = Rhino.Geometry.Point3d(x, y, z)
    box = rs.AddBox([corner_point, 
                     Rhino.Geometry.Point3d(x+width, y, z), 
                     Rhino.Geometry.Point3d(x+width, y+length, z), 
                     Rhino.Geometry.Point3d(x, y+length, z),
                     Rhino.Geometry.Point3d(x, y, z+height), 
                     Rhino.Geometry.Point3d(x+width, y, z+height), 
                     Rhino.Geometry.Point3d(x+width, y+length, z+height), 
                     Rhino.Geometry.Point3d(x, y+length, z+height)])
    
    # Name the object
    rs.ObjectName(box, name)
    
    # Add attributes
    rs.SetUserText(box, "RoomType", room_type)
    rs.SetUserText(box, "RoomID", str(name))
    
    return box

"""

    # Get rooms from the layout
    rooms = {}
    if hasattr(layout, "rooms"):
        # SpatialGrid object
        rooms = layout.rooms
    elif isinstance(layout, dict) and "rooms" in layout:
        # Dictionary representation of layout
        rooms = layout["rooms"]
    else:
        rooms = layout  # Assume it's already a rooms dictionary

    # Add create_room calls for each room
    room_creation_calls = []
    for room_id, room_data in rooms.items():
        # Make sure room_id is a string
        room_id_str = str(room_id)

        # Extract position, dimensions, and type
        if isinstance(room_data, dict):
            # Dictionary format
            position = room_data.get("position", [0, 0, 0])
            dimensions = room_data.get("dimensions", [0, 0, 0])
            room_type = room_data.get("type", "default")

            # Get name from metadata if available
            name = f"Room_{room_id_str}"
            if "metadata" in room_data and room_data["metadata"]:
                if "name" in room_data["metadata"]:
                    name = room_data["metadata"]["name"]
                elif "original_name" in room_data["metadata"]:
                    name = room_data["metadata"]["original_name"]
        else:
            # Object with attributes
            position = getattr(room_data, "position", [0, 0, 0])
            dimensions = getattr(room_data, "dimensions", [0, 0, 0])
            room_type = getattr(room_data, "room_type", "default")
            name = getattr(room_data, "name", f"Room_{room_id_str}")

        # Make sure position and dimensions are lists/tuples
        if not isinstance(position, (list, tuple)):
            position = [0, 0, 0]
        if not isinstance(dimensions, (list, tuple)):
            dimensions = [0, 0, 0]

        # Ensure we have at least 3 elements for position and dimensions
        while len(position) < 3:
            position.append(0)
        while len(dimensions) < 3:
            dimensions.append(0)

        # Format the create_room call
        room_call = f'create_room("{name}", "{room_type}", {position[0]}, {position[1]}, {position[2]}, {dimensions[0]}, {dimensions[1]}, {dimensions[2]})'
        room_creation_calls.append(room_call)

    # Add room creation calls to the script
    script_content += "\n# Create each room\n"
    script_content += "\n".join(room_creation_calls)

    # Add final setup commands
    script_content += """

# Set up a good view
rs.Command("_-NamedView _Save HotelOverview _Enter", False)
rs.Command("_-View _Top", False)
rs.Command("_-Zoom _All _Extents", False)

# Enter perspective view
rs.Command("_-Perspective", False)
rs.Command("_-Zoom _All _Extents", False)

# Set the display mode
rs.Command("_-SetDisplayMode Shaded", False)

# Report success
print("Hotel layout imported successfully!")
print("Created {0} rooms")
""".format(
        len(rooms)
    )

    # Write the script to the output file
    with open(output_filepath, "w") as f:
        f.write(script_content)

    return output_filepath


def export_to_ifc(layout: SpatialGrid, filename: str):
    """
    Export the layout to IFC format for BIM applications.
    Requires ifcopenshell library.

    Args:
        layout: The spatial grid layout to export
        filename: Output IFC filename
    """
    try:
        import ifcopenshell
        import ifcopenshell.api
    except ImportError:
        print(
            "ifcopenshell is required for IFC export. Install with 'pip install ifcopenshell'"
        )
        print("Note: Installation may be complex depending on your platform")
        return

    # Create new IFC file
    ifc_file = ifcopenshell.file()

    # TODO: Implement full IFC export
    # This requires significant code to set up the IFC schema correctly
    # Consider implementing in future versions

    print("Full IFC export not yet implemented")

    # For now, provide a simpler export
    export_to_csv(layout, filename.replace(".ifc", ".csv"))
    print(f"Exported simplified CSV instead to {filename.replace('.ifc', '.csv')}")
