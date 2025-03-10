from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
import random
from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room

class RuleEngine:
    """
    Rule-based layout generation engine that uses architectural principles
    to create hotel layouts.
    """
    
    def __init__(
        self,
        bounding_box: Tuple[float, float, float],
        grid_size: float = 1.0,
        structural_grid: Tuple[float, float] = (8.0, 8.0)
    ):
        """
        Initialize the rule engine.
        
        Args:
            bounding_box: (width, length, height) of buildable area in meters
            grid_size: Size of spatial grid cells in meters
            structural_grid: (x_spacing, y_spacing) of structural grid in meters
        """
        self.width, self.length, self.height = bounding_box
        self.grid_size = grid_size
        self.structural_grid = structural_grid
        
        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(
            width=self.width,
            length=self.length,
            height=self.height,
            grid_size=self.grid_size
        )
        
        # Room placement priorities (architectural knowledge)
        self.placement_priorities = {
            'entrance': 10,
            'lobby': 9,
            'vertical_circulation': 8,
            'restaurant': 7,
            'meeting_rooms': 6,
            'guest_rooms': 5,
            'service_areas': 4,
            'back_of_house': 3
        }
        
        # Adjacency preferences (architectural knowledge)
        self.adjacency_preferences = {
            'entrance': ['lobby', 'vertical_circulation'],
            'lobby': ['entrance', 'restaurant', 'vertical_circulation', 'meeting_rooms'],
            'vertical_circulation': ['lobby', 'guest_rooms', 'service_areas'],
            'restaurant': ['lobby', 'kitchen'],
            'meeting_rooms': ['lobby', 'vertical_circulation'],
            'guest_rooms': ['vertical_circulation'],
            'service_areas': ['vertical_circulation', 'back_of_house'],
            'back_of_house': ['service_areas', 'kitchen']
        }
        
        # Exterior access preferences (0 = no preference, 1 = preferred, 2 = required)
        self.exterior_preferences = {
            'entrance': 2,  # Required
            'lobby': 1,     # Preferred
            'restaurant': 1,  # Preferred
            'guest_rooms': 1,  # Preferred
            'vertical_circulation': 0,  # No preference
            'service_areas': 0,  # No preference
            'back_of_house': 0,  # No preference
            'meeting_rooms': 0   # No preference
        }
    
    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout based on architectural rules.
        
        Args:
            rooms: List of Room objects to place
            
        Returns:
            SpatialGrid: The generated layout
        """
        # Clear the spatial grid
        self.spatial_grid = SpatialGrid(
            width=self.width, 
            length=self.length, 
            height=self.height,
            grid_size=self.grid_size
        )
        
        # Sort rooms by architectural priority
        sorted_rooms = sorted(
            rooms,
            key=lambda r: self.placement_priorities.get(r.room_type, 0),
            reverse=True
        )
        
        # Track placed rooms by type for adjacency checks
        placed_rooms_by_type = {}
        
        # Place rooms according to priorities
        for room in sorted_rooms:
            best_position = self._find_best_position(room, placed_rooms_by_type)
            
            if best_position:
                x, y, z = best_position
                success = self.spatial_grid.place_room(
                    room_id=room.id,
                    x=x,
                    y=y,
                    z=z,
                    width=room.width,
                    length=room.length,
                    height=room.height,
                    room_type=room.room_type,
                    metadata=room.metadata
                )
                
                if success:
                    # Add to placed rooms tracking
                    if room.room_type not in placed_rooms_by_type:
                        placed_rooms_by_type[room.room_type] = []
                    placed_rooms_by_type[room.room_type].append(room.id)
        
        return self.spatial_grid
    
    def _find_best_position(
        self, 
        room: Room, 
        placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find the best position for a room based on architectural rules.
        
        Args:
            room: Room to place
            placed_rooms_by_type: Dictionary of already placed rooms by type
            
        Returns:
            Optional[Tuple[float, float, float]]: Best (x, y, z) position or None if no valid position
        """
        # Special handling based on room type
        if room.room_type == 'entrance':
            return self._place_entrance(room)
        elif room.room_type == 'lobby':
            return self._place_lobby(room, placed_rooms_by_type)
        elif room.room_type == 'vertical_circulation':
            return self._place_vertical_circulation(room, placed_rooms_by_type)
        elif room.room_type == 'guest_rooms':
            return self._place_guest_room(room, placed_rooms_by_type)
        else:
            # Default placement logic for other room types
            return self._place_general_room(room, placed_rooms_by_type)
    
    def _place_entrance(self, room: Room) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for entrance"""
        # By architectural convention, entrances are typically:
        # 1. On the ground floor (z=0)
        # 2. At the front of the building (commonly y=0 or y=max)
        # 3. Centered along the x-axis
        
        # Center along x-axis
        center_x = (self.width - room.width) / 2
        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        
        # Try front of building first (y=0)
        if self._is_valid_position(grid_x, 0, 0, room.width, room.length, room.height):
            return (grid_x, 0, 0)
        
        # Try back of building if front doesn't work
        back_y = self.length - room.length
        if self._is_valid_position(grid_x, back_y, 0, room.width, room.length, room.height):
            return (grid_x, back_y, 0)
        
        # Try left side
        if self._is_valid_position(0, center_x, 0, room.width, room.length, room.height):
            return (0, center_x, 0)
        
        # Try right side
        right_x = self.width - room.width
        if self._is_valid_position(right_x, center_x, 0, room.width, room.length, room.height):
            return (right_x, center_x, 0)
        
        # If all preferred positions fail, try a more exhaustive search
        return self._find_valid_position_on_grid(room, z=0)
    
    def _place_lobby(
        self, 
        room: Room, 
        placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for lobby"""
        # Lobbies should be:
        # 1. Adjacent to entrance
        # 2. On the ground floor
        # 3. Central to the building
        
        # Check if entrance has been placed
        if 'entrance' in placed_rooms_by_type and placed_rooms_by_type['entrance']:
            entrance_id = placed_rooms_by_type['entrance'][0]
            entrance = self.spatial_grid.rooms[entrance_id]
            
            # Try positions adjacent to the entrance
            entrance_x, entrance_y, _ = entrance['position']
            entrance_w, entrance_l, _ = entrance['dimensions']
            
            # Try positioning after the entrance (further into the building)
            lobby_y = entrance_y + entrance_l
            if self._is_valid_position(entrance_x, lobby_y, 0, room.width, room.length, room.height):
                return (entrance_x, lobby_y, 0)
            
            # Try positioning beside the entrance
            lobby_x = entrance_x + entrance_w
            if self._is_valid_position(lobby_x, entrance_y, 0, room.width, room.length, room.height):
                return (lobby_x, entrance_y, 0)
        
        # If no entrance or no valid position near entrance, place centrally
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2
        
        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        grid_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]
        
        if self._is_valid_position(grid_x, grid_y, 0, room.width, room.length, room.height):
            return (grid_x, grid_y, 0)
        
        # Fallback to grid search
        return self._find_valid_position_on_grid(room, z=0)
    
    def _place_vertical_circulation(
        self, 
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for vertical circulation (stairs, elevators)"""
        # Vertical circulation should be:
        # 1. Placed consistently on all floors
        # 2. Near the center or in corners
        # 3. Adjacent to lobby on ground floor
        
        # If lobby is placed, try to position adjacent to it
        if 'lobby' in placed_rooms_by_type and placed_rooms_by_type['lobby']:
            lobby_id = placed_rooms_by_type['lobby'][0]
            lobby = self.spatial_grid.rooms[lobby_id]
            
            lobby_x, lobby_y, _ = lobby['position']
            lobby_w, lobby_l, _ = lobby['dimensions']
            
            # Try positions around the lobby
            positions = [
                (lobby_x + lobby_w, lobby_y, 0),  # Right
                (lobby_x, lobby_y + lobby_l, 0),  # Behind
                (lobby_x - room.width, lobby_y, 0),  # Left
            ]
            
            for pos in positions:
                x, y, z = pos
                if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                    return pos
        
        # Central core positioning
        center_x = (self.width - room.width) / 2
        center_y = (self.length - room.length) / 2
        
        # Align with structural grid
        grid_x = round(center_x / self.structural_grid[0]) * self.structural_grid[0]
        grid_y = round(center_y / self.structural_grid[1]) * self.structural_grid[1]
        
        if self._is_valid_position(grid_x, grid_y, 0, room.width, room.length, room.height):
            return (grid_x, grid_y, 0)
        
        # Check corners
        corner_positions = [
            (0, 0, 0),  # Bottom-left
            (self.width - room.width, 0, 0),  # Bottom-right
            (0, self.length - room.length, 0),  # Top-left
            (self.width - room.width, self.length - room.length, 0)  # Top-right
        ]
        
        for pos in corner_positions:
            x, y, z = pos
            if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                return pos
        
        # Fallback to grid search
        return self._find_valid_position_on_grid(room, z=0)
    
    def _place_guest_room(
        self, 
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Find optimal position for guest rooms"""
        # Guest rooms are typically:
        # 1. On upper floors
        # 2. Arranged in efficient layouts along corridors
        # 3. Have exterior access for natural light
        
        # Determine which floor to place on (start from floor 1, not ground floor)
        floor_height = 4.0  # Typical floor height
        current_floor = 1  # Start from first floor
        
        # Calculate z coordinate for this floor
        z = current_floor * floor_height
        
        # Try to place along the perimeter for exterior access
        perimeter_positions = []
        
        # Front edge
        for x in range(0, int(self.width - room.width) + 1, int(self.structural_grid[0])):
            perimeter_positions.append((x, 0, z))
        
        # Back edge
        back_y = self.length - room.length
        for x in range(0, int(self.width - room.width) + 1, int(self.structural_grid[0])):
            perimeter_positions.append((x, back_y, z))
        
        # Left edge
        for y in range(0, int(self.length - room.length) + 1, int(self.structural_grid[1])):
            perimeter_positions.append((0, y, z))
        
        # Right edge
        right_x = self.width - room.width
        for y in range(0, int(self.length - room.length) + 1, int(self.structural_grid[1])):
            perimeter_positions.append((right_x, y, z))
        
        # Shuffle to avoid always filling in the same pattern
        random.shuffle(perimeter_positions)
        
        for pos in perimeter_positions:
            x, y, z = pos
            if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                return pos
        
        # If perimeter is full, try interior positions
        return self._find_valid_position_on_grid(room, z=z)
    
    def _place_general_room(
        self, 
        room: Room,
        placed_rooms_by_type: Dict[str, List[int]]
    ) -> Optional[Tuple[float, float, float]]:
        """Default placement logic for other room types"""
        # Check for adjacency preferences
        adjacent_to = self.adjacency_preferences.get(room.room_type, [])
        
        # Try to place adjacent to preferred room types
        for preferred_type in adjacent_to:
            if preferred_type in placed_rooms_by_type:
                # Get all rooms of this type
                for room_id in placed_rooms_by_type[preferred_type]:
                    existing_room = self.spatial_grid.rooms[room_id]
                    
                    x, y, z = existing_room['position']
                    w, l, h = existing_room['dimensions']
                    
                    # Try positions around this room
                    positions = [
                        (x + w, y, z),  # Right
                        (x, y + l, z),  # Behind
                        (x - room.width, y, z),  # Left
                        (x, y - room.length, z)  # In front
                    ]
                    
                    for pos in positions:
                        test_x, test_y, test_z = pos
                        if self._is_valid_position(test_x, test_y, test_z, 
                                                 room.width, room.length, room.height):
                            return pos
        
        # Check if exterior access is needed
        exterior_pref = self.exterior_preferences.get(room.room_type, 0)
        
        if exterior_pref > 0:
            # Similar perimeter logic as for guest rooms
            perimeter_positions = []
            
            # Front edge
            for x in range(0, int(self.width - room.width) + 1, int(self.structural_grid[0])):
                perimeter_positions.append((x, 0, 0))
            
            # Back edge
            back_y = self.length - room.length
            for x in range(0, int(self.width - room.width) + 1, int(self.structural_grid[0])):
                perimeter_positions.append((x, back_y, 0))
            
            # Left edge
            for y in range(0, int(self.length - room.length) + 1, int(self.structural_grid[1])):
                perimeter_positions.append((0, y, 0))
            
            # Right edge
            right_x = self.width - room.width
            for y in range(0, int(self.length - room.length) + 1, int(self.structural_grid[1])):
                perimeter_positions.append((right_x, y, 0))
            
            random.shuffle(perimeter_positions)
            
            for pos in perimeter_positions:
                x, y, z = pos
                if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                    return pos
            
            # If exterior is required but no position found, return None
            if exterior_pref == 2:
                return None
        
        # Default to grid-based placement
        return self._find_valid_position_on_grid(room)
    
    def _is_valid_position(
        self, 
        x: float, 
        y: float, 
        z: float, 
        width: float, 
        length: float, 
        height: float
    ) -> bool:
        """Check if a position is valid for room placement"""
        # Convert to grid coordinates
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_z = int(z / self.grid_size)
        grid_width = int(width / self.grid_size)
        grid_length = int(length / self.grid_size)
        grid_height = int(height / self.grid_size)
        
        # Check bounds
        if (grid_x < 0 or grid_y < 0 or grid_z < 0 or
            grid_x + grid_width > self.spatial_grid.width_cells or
            grid_y + grid_length > self.spatial_grid.length_cells or
            grid_z + grid_height > self.spatial_grid.height_cells):
            return False
        
        # Check for collisions with existing rooms
        target_region = self.spatial_grid.grid[
            grid_x:grid_x + grid_width,
            grid_y:grid_y + grid_length,
            grid_z:grid_z + grid_height
        ]
        
        return np.all(target_region == 0)
    
    def _find_valid_position_on_grid(
        self, 
        room: Room, 
        z: float = 0
    ) -> Optional[Tuple[float, float, float]]:
        """Find a valid position on the structural grid"""
        grid_step_x = self.structural_grid[0]
        grid_step_y = self.structural_grid[1]
        
        # Try all positions aligned with structural grid
        for x in range(0, int(self.width - room.width) + 1, int(grid_step_x)):
            for y in range(0, int(self.length - room.length) + 1, int(grid_step_y)):
                if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                    return (x, y, z)
        
        # If structural grid positions don't work, try arbitrary positions
        for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
            for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                    return (x, y, z)
        
        # If z=0 doesn't work and this is not a ground-floor-specific room type,
        # try other floors
        if z == 0 and room.room_type not in ['entrance', 'lobby', 'restaurant']:
            floor_height = 4.0
            for floor in range(1, int(self.height / floor_height)):
                new_z = floor * floor_height
                result = self._find_valid_position_on_grid(room, z=new_z)
                if result:
                    return result
        
        # No valid position found
        return None
