import numpy as np
from typing import Tuple, List, Dict, Set, Optional

class SpatialGrid:
    """
    3D spatial representation system for hotel layouts.
    Uses a voxel-based grid to represent spaces.
    """
    
    def __init__(
        self, 
        width: float, 
        length: float, 
        height: float, 
        grid_size: float = 1.0
    ):
        """
        Initialize a 3D spatial grid.
        
        Args:
            width: Total width of the bounding box in meters
            length: Total length of the bounding box in meters
            height: Total height of the bounding box in meters
            grid_size: Size of each grid cell in meters
        """
        self.width = width
        self.length = length
        self.height = height
        self.grid_size = grid_size
        
        # Calculate grid dimensions
        self.width_cells = int(width / grid_size)
        self.length_cells = int(length / grid_size)
        self.height_cells = int(height / grid_size)
        
        # Initialize 3D grid (0 = empty space, positive int = room ID)
        self.grid = np.zeros((
            self.width_cells,
            self.length_cells,
            self.height_cells
        ), dtype=int)
        
        # Room metadata
        self.rooms = {}  # room_id -> metadata
        
    def place_room(
        self, 
        room_id: int, 
        x: float, 
        y: float, 
        z: float, 
        width: float, 
        length: float, 
        height: float,
        room_type: str = "generic",
        metadata: Dict = None
    ) -> bool:
        """
        Place a room in the grid at the specified position.
        
        Args:
            room_id: Unique identifier for the room
            x, y, z: Coordinates of the room's origin (bottom-left corner)
            width, length, height: Dimensions of the room
            room_type: Type of room (e.g., "guest_room", "lobby", etc.)
            metadata: Additional room information
            
        Returns:
            bool: True if placement was successful, False otherwise
        """
        # Convert to grid coordinates
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)
        
        # Check if placement is valid
        if not self._is_valid_placement(grid_x, grid_y, grid_z, grid_width, grid_length, grid_height):
            return False
        
        # Place room in grid
        self.grid[
            grid_x:grid_x + grid_width,
            grid_y:grid_y + grid_length,
            grid_z:grid_z + grid_height
        ] = room_id
        
        # Store room metadata
        self.rooms[room_id] = {
            'id': room_id,
            'type': room_type,
            'position': (x, y, z),
            'dimensions': (width, length, height),
            'grid_position': (grid_x, grid_y, grid_z),
            'grid_dimensions': (grid_width, grid_length, grid_height),
            **(metadata or {})
        }
        
        return True
    
    def remove_room(self, room_id: int) -> bool:
        """
        Remove a room from the grid.
        
        Args:
            room_id: ID of the room to remove
            
        Returns:
            bool: True if room was found and removed, False otherwise
        """
        if room_id not in self.rooms:
            return False
        
        # Get room grid coordinates
        grid_x, grid_y, grid_z = self.rooms[room_id]['grid_position']
        grid_width, grid_length, grid_height = self.rooms[room_id]['grid_dimensions']
        
        # Clear room from grid
        self.grid[
            grid_x:grid_x + grid_width,
            grid_y:grid_y + grid_length,
            grid_z:grid_z + grid_height
        ] = 0
        
        # Remove from room metadata
        del self.rooms[room_id]
        
        return True
    
    def _is_valid_placement(
        self, 
        grid_x: int, 
        grid_y: int, 
        grid_z: int, 
        grid_width: int, 
        grid_length: int, 
        grid_height: int
    ) -> bool:
        """Check if a room can be placed at the specified location"""
        # Check bounds
        if (grid_x < 0 or grid_y < 0 or grid_z < 0 or
            grid_x + grid_width > self.width_cells or
            grid_y + grid_length > self.length_cells or
            grid_z + grid_height > self.height_cells):
            return False
        
        # Check for collisions with existing rooms
        target_region = self.grid[
            grid_x:grid_x + grid_width,
            grid_y:grid_y + grid_length,
            grid_z:grid_z + grid_height
        ]
        
        return np.all(target_region == 0)
    
    def are_adjacent(self, room_id1: int, room_id2: int) -> bool:
        """Check if two rooms are adjacent to each other"""
        if room_id1 not in self.rooms or room_id2 not in self.rooms:
            return False
        
        # Create a temporary grid with only these two rooms
        temp_grid = np.zeros_like(self.grid)
        
        # Fill room 1
        room1 = self.rooms[room_id1]
        x1, y1, z1 = room1['grid_position']
        w1, l1, h1 = room1['grid_dimensions']
        temp_grid[x1:x1+w1, y1:y1+l1, z1:z1+h1] = 1
        
        # Fill room 2
        room2 = self.rooms[room_id2]
        x2, y2, z2 = room2['grid_position']
        w2, l2, h2 = room2['grid_dimensions']
        temp_grid[x2:x2+w2, y2:y2+l2, z2:z2+h2] = 2
        
        # Expand room 1 by 1 cell in all directions
        expanded = np.zeros_like(temp_grid)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    # Skip diagonals, only consider face adjacency
                    if abs(dx) + abs(dy) + abs(dz) != 1:
                        continue
                    
                    # Compute shifted indices
                    x_min = max(0, x1 + dx)
                    x_max = min(self.width_cells, x1 + w1 + dx)
                    y_min = max(0, y1 + dy)
                    y_max = min(self.length_cells, y1 + l1 + dy)
                    z_min = max(0, z1 + dz)
                    z_max = min(self.height_cells, z1 + h1 + dz)
                    
                    # Set expanded cells
                    expanded[x_min:x_max, y_min:y_max, z_min:z_max] = 1
        
        # Check if expanded room 1 overlaps with room 2
        return np.any((expanded == 1) & (temp_grid == 2))
    
    def get_room_neighbors(self, room_id: int) -> List[int]:
        """Get all rooms adjacent to the specified room"""
        if room_id not in self.rooms:
            return []
        
        neighbors = []
        for other_id in self.rooms:
            if other_id != room_id and self.are_adjacent(room_id, other_id):
                neighbors.append(other_id)
        
        return neighbors
    
    def get_exterior_rooms(self) -> List[int]:
        """Get rooms that are on the exterior of the building"""
        exterior_rooms = []
        
        # Check each face of the bounding box
        for room_id in self.rooms:
            room = self.rooms[room_id]
            x, y, z = room['grid_position']
            w, l, h = room['grid_dimensions']
            
            # Check if room touches any exterior face
            if (x == 0 or y == 0 or z == 0 or
                x + w == self.width_cells or
                y + l == self.length_cells or
                z + h == self.height_cells):
                exterior_rooms.append(room_id)
        
        return exterior_rooms
    
    def calculate_space_utilization(self) -> float:
        """Calculate what percentage of the volume is utilized"""
        occupied_cells = np.count_nonzero(self.grid)
        total_cells = self.width_cells * self.length_cells * self.height_cells
        return occupied_cells / total_cells
    
    def to_dict(self) -> Dict:
        """Convert the spatial grid to a serializable dictionary"""
        return {
            'dimensions': {
                'width': self.width,
                'length': self.length,
                'height': self.height,
                'grid_size': self.grid_size
            },
            'rooms': self.rooms,
            'grid': self.grid.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpatialGrid':
        """Create a SpatialGrid from a dictionary"""
        grid = cls(
            width=data['dimensions']['width'],
            length=data['dimensions']['length'],
            height=data['dimensions']['height'],
            grid_size=data['dimensions']['grid_size']
        )
        
        grid.grid = np.array(data['grid'])
        grid.rooms = data['rooms']
        
        return grid
