import numpy as np
import json
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
    with open(filename, 'w') as f:
        json.dump(layout_dict, f, indent=2)


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
        x, y, z = room['position']
        w, l, h = room['dimensions']
        
        data = {
            'room_id': room_id,
            'room_type': room['type'],
            'name': room.get('name', f"Room {room_id}"),
            'x': x,
            'y': y,
            'z': z,
            'width': w,
            'length': l,
            'height': h,
            'volume': w * l * h,
            'area': w * l,
        }
        room_data.append(data)
    
    # Write header
    with open(filename, 'w') as f:
        # Define columns
        columns = [
            'room_id', 'room_type', 'name', 
            'x', 'y', 'z', 
            'width', 'length', 'height',
            'volume', 'area'
        ]
        
        # Write header
        f.write(','.join(columns) + '\n')
        
        # Write data rows
        for room in room_data:
            row = [str(room[col]) for col in columns]
            f.write(','.join(row) + '\n')


def export_to_revit(layout: SpatialGrid, filename: str):
    """
    Export layout data in a format that can be imported into Revit.
    This exports as a simplified CSV that can be used with Dynamo.
    
    Args:
        layout: The spatial grid layout to export
        filename: Output filename for Revit import
    """
    # Extract room data in Revit-friendly format
    room_data = []
    
    for room_id, room in layout.rooms.items():
        x, y, z = room['position']
        w, l, h = room['dimensions']
        
        # Revit typically uses feet as units, convert if needed
        # Assuming layout is in meters, convert to feet
        meters_to_feet = 3.28084
        
        data = {
            'Name': room.get('name', f"Room {room_id}"),
            'Type': room['type'].replace('_', ' ').title(),
            'Level': int(z / 4.0) + 1,  # Assuming 4m floor height
            'X_Meters': x,
            'Y_Meters': y, 
            'Z_Meters': z,
            'Width_Meters': w,
            'Length_Meters': l,
            'Height_Meters': h,
            'X_Feet': x * meters_to_feet,
            'Y_Feet': y * meters_to_feet,
            'Z_Feet': z * meters_to_feet,
            'Width_Feet': w * meters_to_feet,
            'Length_Feet': l * meters_to_feet,
            'Height_Feet': h * meters_to_feet,
        }
        room_data.append(data)
    
    # Write to CSV
    with open(filename, 'w') as f:
        # Define columns - include both metric and imperial units for Revit
        columns = [
            'Name', 'Type', 'Level',
            'X_Meters', 'Y_Meters', 'Z_Meters',
            'Width_Meters', 'Length_Meters', 'Height_Meters',
            'X_Feet', 'Y_Feet', 'Z_Feet',
            'Width_Feet', 'Length_Feet', 'Height_Feet'
        ]
        
        # Write header
        f.write(','.join(columns) + '\n')
        
        # Write data rows
        for room in room_data:
            row = [str(room[col]) for col in columns]
            f.write(','.join(row) + '\n')


def export_for_three_js(layout: SpatialGrid, filename: str):
    """
    Export the layout for use with Three.js visualizations.
    
    Args:
        layout: The spatial grid layout to export
        filename: Output JSON filename for Three.js
    """
    # Create a Three.js friendly data structure
    three_js_data = {
        'metadata': {
            'version': 1.0,
            'type': 'hotel_layout',
            'dimensions': {
                'width': layout.width,
                'length': layout.length,
                'height': layout.height
            }
        },
        'rooms': []
    }
    
    # Room type to color mapping (hex colors)
    room_colors = {
        'entrance': '#2c7fb8',
        'lobby': '#7fcdbb',
        'vertical_circulation': '#c7e9b4',
        'restaurant': '#f0f9e8',
        'meeting_room': '#edf8b1',
        'guest_room': '#f7fcb9',
        'service_area': '#d9f0a3',
        'back_of_house': '#addd8e',
        'default': '#efefef'
    }
    
    # Add rooms
    for room_id, room in layout.rooms.items():
        x, y, z = room['position']
        w, l, h = room['dimensions']
        room_type = room['type']
        
        three_js_room = {
            'id': room_id,
            'type': room_type,
            'name': room.get('name', f"Room {room_id}"),
            'position': [x, z, y],  # Three.js uses Y-up, change coordinates
            'dimensions': [w, h, l],  # Three.js uses Y-up, change dimensions
            'color': room_colors.get(room_type, room_colors['default'])
        }
        
        three_js_data['rooms'].append(three_js_room)
    
    # Write to JSON file
    with open(filename, 'w') as f:
        json.dump(three_js_data, f, indent=2)


def export_to_sketchup(layout: SpatialGrid, filename: str):
    """
    Export the layout to SketchUp-compatible format.
    Uses a simplified CSV format that can be imported with Ruby scripts.
    
    Args:
        layout: The spatial grid layout to export
        filename: Output CSV filename for SketchUp
    """
    # Extract room data
    room_data = []
    
    for room_id, room in layout.rooms.items():
        x, y, z = room['position']
        w, l, h = room['dimensions']
        room_type = room['type']
        name = room.get('name', f"Room {room_id}")
        
        data = {
            'ID': room_id,
            'Name': name,
            'Type': room_type,
            'X': x,
            'Y': y,
            'Z': z,
            'Width': w,
            'Length': l,
            'Height': h
        }
        room_data.append(data)
    
    # Write to CSV
    with open(filename, 'w') as f:
        # Define columns
        columns = ['ID', 'Name', 'Type', 'X', 'Y', 'Z', 'Width', 'Length', 'Height']
        
        # Write header
        f.write(','.join(columns) + '\n')
        
        # Write data rows
        for room in room_data:
            row = [str(room[col]) for col in columns]
            f.write(','.join(row) + '\n')
    
    # Create a simple SketchUp Ruby script to import the data
    script_filename = filename.replace('.csv', '.rb')
    with open(script_filename, 'w') as f:
        f.write("""
# SketchUp Ruby script to import hotel layout
require 'csv'

# Function to create a room
def create_room(model, name, type, x, y, z, width, length, height)
  group = model.active_entities.add_group
  entities = group.entities
  
  # Create faces for the room
  face = entities.add_face([0,0,0], [width,0,0], [width,length,0], [0,length,0])
  face.pushpull(height)
  
  # Set attributes
  group.name = name
  
  # Move to position
  transformation = Geom::Transformation.new([x, y, z])
  group.transform!(transformation)
  
  # Add attributes
  if model.attribute_dictionaries == nil
    model.attribute_dictionaries = AttributeDictionaries.new(model)
  end
  
  dict_name = "RoomInfo"
  if !model.attribute_dictionary(dict_name)
    model.attribute_dictionaries.add(dict_name)
  end
  
  attrdicts = group.attribute_dictionaries
  if attrdicts == nil
    attrdicts = AttributeDictionaries.new(group)
  end
  
  attrdict = attrdicts.add(dict_name)
  attrdict["Type"] = type
  
  return group
end

# Main import function
def import_hotel_layout(csv_file)
  model = Sketchup.active_model
  
  # Start an operation to undo all at once
  model.start_operation("Import Hotel Layout", true)
  
  # Read CSV file
  CSV.foreach(csv_file, headers: true) do |row|
    # Extract values
    name = row['Name'].to_s
    type = row['Type'].to_s
    x = row['X'].to_f
    y = row['Y'].to_f
    z = row['Z'].to_f
    width = row['Width'].to_f
    length = row['Length'].to_f
    height = row['Height'].to_f
    
    # Create the room
    create_room(model, name, type, x, y, z, width, length, height)
  end
  
  # Commit the operation
  model.commit_operation
end

# Call import with the CSV file path - update this path
import_hotel_layout("REPLACE_WITH_ABSOLUTE_PATH_TO_CSV")

""".replace("REPLACE_WITH_ABSOLUTE_PATH_TO_CSV", filename))
    
    print(f"Exported SketchUp CSV to {filename}")
    print(f"Created SketchUp Ruby import script: {script_filename}")
    print("In SketchUp, run this script via Ruby Console to import the layout.")


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
        print("ifcopenshell is required for IFC export. Install with 'pip install ifcopenshell'")
        print("Note: Installation may be complex depending on your platform")
        return
    
    # Create new IFC file
    ifc_file = ifcopenshell.file()
    
    # TODO: Implement full IFC export
    # This requires significant code to set up the IFC schema correctly
    # Consider implementing in future versions
    
    print("Full IFC export not yet implemented")
    
    # For now, provide a simpler export
    export_to_csv(layout, filename.replace('.ifc', '.csv'))
    print(f"Exported simplified CSV instead to {filename.replace('.ifc', '.csv')}")
