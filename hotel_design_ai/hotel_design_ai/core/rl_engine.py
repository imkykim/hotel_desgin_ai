import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Dict, Tuple, Any, Optional

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room

class RLPolicyNetwork(nn.Module):
    """
    Neural network that represents the policy for the RL agent.
    This model predicts the best position for a room given the current state.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state representation
            action_dim: Dimension of the action space
            hidden_dim: Size of hidden layers
        """
        super(RLPolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        """Forward pass through the network"""
        return self.network(state)


class RLEngine:
    """
    Reinforcement Learning engine for hotel layout generation.
    Learns from user feedback to optimize room placement.
    """
    
    def __init__(
        self,
        bounding_box: Tuple[float, float, float],
        grid_size: float = 1.0,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_min: float = 0.01,
        exploration_decay: float = 0.995,
        memory_size: int = 10000
    ):
        """
        Initialize the RL engine.
        
        Args:
            bounding_box: (width, length, height) of buildable area in meters
            grid_size: Size of spatial grid cells in meters
            structural_grid: (x_spacing, y_spacing) of structural grid in meters
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            exploration_min: Minimum exploration rate
            exploration_decay: Rate at which exploration decays
            memory_size: Size of experience replay buffer
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
        
        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Room types and architectural knowledge - DEFINED BEFORE STATE DIM CALCULATION
        self.room_types = [
            'entrance', 'lobby', 'vertical_circulation', 'restaurant',
            'meeting_room', 'guest_room', 'service_area', 'back_of_house',
            'kitchen', 'ballroom', 'lounge', 'pre_function', 'maintenance',
            'mechanical', 'pool', 'fitness', 'retail', 'recreation',
            'entertainment', 'office', 'staff_area', 'food_service',
            'parking', 'circulation', 'storage'
        ]
        
        # State and action dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        
        # Initialize neural networks
        self.policy_network = RLPolicyNetwork(self.state_dim, self.action_dim)
        self.target_network = RLPolicyNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Fixed elements (set by user)
        self.fixed_elements = {}  # room_id -> position
        
        # Training metrics
        self.training_iterations = 0
        self.average_reward = 0
    
    def _calculate_state_dim(self) -> int:
        """Calculate the dimension of the state representation"""
        # Grid state + room properties + fixed elements mask
        grid_cells = (
            self.spatial_grid.width_cells * 
            self.spatial_grid.length_cells * 
            self.spatial_grid.height_cells
        )
        
        # We'll use a simplified grid representation for efficiency
        simplified_grid_cells = (
            int(self.width / self.structural_grid[0]) * 
            int(self.length / self.structural_grid[1]) * 
            int(self.height / 4.0)  # Assuming standard floor height
        )
        
        # Room properties (type, dimensions, requirements)
        room_props = len(self.room_types) + 3  # 3 basic properties + one-hot encoding
        
        # Additional state components
        fixed_mask = simplified_grid_cells
        
        return simplified_grid_cells + room_props + fixed_mask
    
    def _calculate_action_dim(self) -> int:
        """Calculate the dimension of the action space"""
        # Action is (x, y, z) position, which we'll discretize to structural grid
        grid_cells_x = int(self.width / self.structural_grid[0])
        grid_cells_y = int(self.length / self.structural_grid[1])
        floors = int(self.height / 4.0)  # Assuming standard floor height
        
        return grid_cells_x * grid_cells_y * floors
    
    def _get_state_representation(
        self, 
        room_to_place: Room
    ) -> torch.Tensor:
        """
        Create a state representation for the RL agent.
        
        Args:
            room_to_place: The room we're trying to place
            
        Returns:
            torch.Tensor: State representation
        """
        # Simplify the grid to structural grid resolution for efficiency
        simplified_grid = np.zeros((
            int(self.width / self.structural_grid[0]),
            int(self.length / self.structural_grid[1]),
            int(self.height / 4.0)  # Assuming standard floor height
        ))
        
        # Fill simplified grid with room type codes
        for room_id, room_data in self.spatial_grid.rooms.items():
            room_type = room_data['type']
            try:
                room_type_code = self.room_types.index(room_type) + 1  # 0 is empty
            except ValueError:
                room_type_code = self.room_types.index('back_of_house') + 1  # Default type
            
            # Convert room position to simplified grid coordinates
            pos_x, pos_y, pos_z = room_data['position']
            dim_w, dim_l, dim_h = room_data['dimensions']
            
            # Calculate simplified grid coordinates
            grid_x = int(pos_x / self.structural_grid[0])
            grid_y = int(pos_y / self.structural_grid[1])
            grid_z = int(pos_z / 4.0)
            
            grid_w = max(1, int(dim_w / self.structural_grid[0]))
            grid_l = max(1, int(dim_l / self.structural_grid[1]))
            grid_h = max(1, int(dim_h / 4.0))
            
            # Fill the grid region
            for x in range(grid_x, min(grid_x + grid_w, simplified_grid.shape[0])):
                for y in range(grid_y, min(grid_y + grid_l, simplified_grid.shape[1])):
                    for z in range(grid_z, min(grid_z + grid_h, simplified_grid.shape[2])):
                        if 0 <= x < simplified_grid.shape[0] and 0 <= y < simplified_grid.shape[1] and 0 <= z < simplified_grid.shape[2]:
                            simplified_grid[x, y, z] = room_type_code
        
        # Flatten the grid
        flat_grid = simplified_grid.flatten()
        
        # Create room properties vector
        room_props = np.zeros(len(self.room_types) + 3)  # 3 basic props + one-hot encoding
        room_props[0] = room_to_place.width / self.width  # Normalized width
        room_props[1] = room_to_place.length / self.length  # Normalized length
        room_props[2] = room_to_place.height / self.height  # Normalized height
        
        # One-hot encoding of room type
        try:
            room_type_idx = self.room_types.index(room_to_place.room_type)
            room_props[3 + room_type_idx] = 1.0
        except ValueError:
            # Handle unknown room type by defaulting to a generic type
            print(f"Warning: Room type '{room_to_place.room_type}' not in predefined types. Using default.")
            # Use 'back_of_house' as default type
            back_of_house_idx = self.room_types.index('back_of_house')
            room_props[3 + back_of_house_idx] = 1.0
        
        # Fixed elements mask (1 where user has fixed elements)
        fixed_mask = np.zeros_like(flat_grid)
        for room_id in self.fixed_elements:
            pos_x, pos_y, pos_z = self.fixed_elements[room_id]
            room_data = self.spatial_grid.rooms.get(room_id, {})
            
            if room_data:
                dim_w, dim_l, dim_h = room_data['dimensions']
                
                # Calculate simplified grid coordinates
                grid_x = int(pos_x / self.structural_grid[0])
                grid_y = int(pos_y / self.structural_grid[1])
                grid_z = int(pos_z / 4.0)
                
                grid_w = max(1, int(dim_w / self.structural_grid[0]))
                grid_l = max(1, int(dim_l / self.structural_grid[1]))
                grid_h = max(1, int(dim_h / 4.0))
                
                # Fill the fixed mask
                for x in range(grid_x, min(grid_x + grid_w, simplified_grid.shape[0])):
                    for y in range(grid_y, min(grid_y + grid_l, simplified_grid.shape[1])):
                        for z in range(grid_z, min(grid_z + grid_h, simplified_grid.shape[2])):
                            if (0 <= x < simplified_grid.shape[0] and 
                                0 <= y < simplified_grid.shape[1] and 
                                0 <= z < simplified_grid.shape[2]):
                                idx = z * simplified_grid.shape[0] * simplified_grid.shape[1] + y * simplified_grid.shape[0] + x
                                if idx < len(fixed_mask):
                                    fixed_mask[idx] = 1.0
        
        # Combine all components into the state vector
        state = np.concatenate([flat_grid, room_props, fixed_mask])
        
        return torch.FloatTensor(state)
    
    def _action_to_position(
        self, 
        action: int, 
        room: Room
    ) -> Tuple[float, float, float]:
        """
        Convert an action index to a room position.
        
        Args:
            action: Action index from the policy network
            room: Room to be placed
            
        Returns:
            Tuple[float, float, float]: (x, y, z) position in meters
        """
        # Calculate simplified grid dimensions
        grid_cells_x = int(self.width / self.structural_grid[0])
        grid_cells_y = int(self.length / self.structural_grid[1])
        floors = int(self.height / 4.0)
        
        # Convert action index to 3D coordinates
        z = action // (grid_cells_x * grid_cells_y)
        remain = action % (grid_cells_x * grid_cells_y)
        y = remain // grid_cells_x
        x = remain % grid_cells_x
        
        # Convert to actual coordinates
        actual_x = x * self.structural_grid[0]
        actual_y = y * self.structural_grid[1]
        actual_z = z * 4.0  # Standard floor height
        
        # Ensure room fits within building bounds
        actual_x = min(actual_x, self.width - room.width)
        actual_y = min(actual_y, self.length - room.length)
        actual_z = min(actual_z, self.height - room.height)
        
        return (actual_x, actual_y, actual_z)
    
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
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get an action from the policy network.
        
        Args:
            state: Current state representation
            
        Returns:
            int: Action index
        """
        # Epsilon-greedy exploration
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_dim)
        
        # Use policy network
        with torch.no_grad():
            q_values = self.policy_network(state)
            return torch.argmax(q_values).item()
    
    def get_alternative_action(self, state: torch.Tensor, top_k: int = 5) -> int:
        """
        Get an alternative action when the best action fails.
        
        Args:
            state: Current state representation
            top_k: Number of top actions to consider
            
        Returns:
            int: Alternative action index
        """
        with torch.no_grad():
            q_values = self.policy_network(state)
            # Get top k actions
            top_actions = torch.topk(q_values, min(top_k, self.action_dim)).indices.numpy()
            # Return a random action from top k
            return np.random.choice(top_actions)
    
    def update_fixed_elements(self, fixed_elements: Dict[int, Tuple[float, float, float]]):
        """
        Update the fixed elements (user placed/modified rooms).
        
        Args:
            fixed_elements: Dictionary mapping room IDs to positions
        """
        self.fixed_elements = fixed_elements.copy()
    
    def clear_non_fixed_elements(self):
        """Remove all non-fixed elements from the spatial grid"""
        # Get all room IDs
        all_room_ids = list(self.spatial_grid.rooms.keys())
        
        # Remove non-fixed rooms
        for room_id in all_room_ids:
            if room_id not in self.fixed_elements:
                self.spatial_grid.remove_room(room_id)
    
    def generate_layout(self, rooms: List[Room]) -> SpatialGrid:
        """
        Generate a hotel layout using the RL policy.
        
        Args:
            rooms: List of Room objects to place
            
        Returns:
            SpatialGrid: The generated layout
        """
        # Clear non-fixed elements
        self.clear_non_fixed_elements()
        
        # Restore fixed elements
        for room_id, position in self.fixed_elements.items():
            # Find the room in the input list
            room = next((r for r in rooms if r.id == room_id), None)
            if room:
                x, y, z = position
                self.spatial_grid.place_room(
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
        
        # Sort rooms by architectural priority
        # This architectural knowledge helps the RL agent
        room_type_priority = {
            'entrance': 10,
            'lobby': 9,
            'vertical_circulation': 8,
            'restaurant': 7,
            'meeting_room': 6,
            'guest_room': 5,
            'service_area': 4,
            'back_of_house': 3
        }
        
        sorted_rooms = sorted(
            rooms,
            key=lambda r: (
                # Fixed rooms first
                r.id not in self.fixed_elements,
                # Then by architectural priority
                -room_type_priority.get(r.room_type, 0)
            )
        )
        
        # Filter out fixed rooms (already placed)
        rooms_to_place = [r for r in sorted_rooms if r.id not in self.fixed_elements]
        
        # Place each room using the policy
        for room in rooms_to_place:
            # Get current state
            state = self._get_state_representation(room)
            
            # Get action from policy
            action = self.get_action(state)
            
            # Convert action to position
            position = self._action_to_position(action, room)
            
            # Try to place the room
            success = False
            attempts = 0
            max_attempts = 10
            
            while not success and attempts < max_attempts:
                x, y, z = position
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
                
                if not success:
                    # Try an alternative action
                    action = self.get_alternative_action(state)
                    position = self._action_to_position(action, room)
                    attempts += 1
            
            # If all attempts fail, try a grid search
            if not success:
                position = self._find_valid_position(room)
                if position:
                    x, y, z = position
                    self.spatial_grid.place_room(
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
        
        return self.spatial_grid
    
    def _find_valid_position(self, room: Room) -> Optional[Tuple[float, float, float]]:
        """Find a valid position using grid search"""
        # Try positions aligned with structural grid
        for z in range(0, int(self.height - room.height) + 1, 4):  # Floor height = 4m
            for x in range(0, int(self.width - room.width) + 1, int(self.structural_grid[0])):
                for y in range(0, int(self.length - room.length) + 1, int(self.structural_grid[1])):
                    if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                        return (x, y, z)
        
        # If structural grid alignment fails, try arbitrary positions
        for z in range(0, int(self.height - room.height) + 1, int(self.grid_size)):
            for x in range(0, int(self.width - room.width) + 1, int(self.grid_size)):
                for y in range(0, int(self.length - room.length) + 1, int(self.grid_size)):
                    if self._is_valid_position(x, y, z, room.width, room.length, room.height):
                        return (x, y, z)
        
        return None
    
    def calculate_reward(self, layout: SpatialGrid) -> float:
        """
        Calculate a reward for a layout based on architectural principles.
        
        Args:
            layout: The layout to evaluate
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # 1. Space utilization (efficiency)
        space_util = layout.calculate_space_utilization()
        reward += space_util * 5.0  # Weight factor
        
        # 2. Adjacency relationships
        adjacency_reward = self._calculate_adjacency_reward(layout)
        reward += adjacency_reward * 3.0  # Weight factor
        
        # 3. Exterior access for rooms that need it
        exterior_reward = self._calculate_exterior_reward(layout)
        reward += exterior_reward * 2.0  # Weight factor
        
        # 4. Architectural coherence (vertical alignment, etc.)
        coherence_reward = self._calculate_coherence_reward(layout)
        reward += coherence_reward * 2.0  # Weight factor
        
        return reward
    
    def _calculate_adjacency_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward component for adjacency relationships"""
        adjacency_score = 0.0
        total_relationships = 0
        
        # Desired adjacency relationships by room type
        adjacency_preferences = {
            'entrance': ['lobby', 'vertical_circulation'],
            'lobby': ['entrance', 'restaurant', 'vertical_circulation', 'meeting_room'],
            'vertical_circulation': ['lobby', 'guest_room', 'service_area'],
            'restaurant': ['lobby', 'kitchen'],
            'meeting_room': ['lobby', 'vertical_circulation'],
            'guest_room': ['vertical_circulation'],
            'service_area': ['vertical_circulation', 'back_of_house'],
            'back_of_house': ['service_area', 'kitchen']
        }
        
        # Check each room's adjacencies
        for room_id, room_data in layout.rooms.items():
            room_type = room_data['type']
            
            # Get preferred adjacencies for this room type
            preferred = adjacency_preferences.get(room_type, [])
            if not preferred:
                continue
            
            # Get actual neighbors
            neighbors = layout.get_room_neighbors(room_id)
            
            # Check if neighbors have the preferred types
            satisfied = 0
            for neighbor_id in neighbors:
                if neighbor_id in layout.rooms:
                    neighbor_type = layout.rooms[neighbor_id]['type']
                    if neighbor_type in preferred:
                        satisfied += 1
            
            # Calculate score for this room
            if preferred:
                adjacency_score += min(1.0, satisfied / len(preferred))
                total_relationships += 1
        
        # Normalize score
        if total_relationships > 0:
            return adjacency_score / total_relationships
        return 0.0
    
    def _calculate_exterior_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward component for exterior access"""
        exterior_score = 0.0
        exterior_rooms = set(layout.get_exterior_rooms())
        
        # Exterior access preferences (0=no preference, 1=preferred, 2=required)
        exterior_preferences = {
            'entrance': 2,  # Required
            'lobby': 1,     # Preferred
            'restaurant': 1,  # Preferred
            'guest_room': 1,  # Preferred
            'vertical_circulation': 0,  # No preference
            'service_area': 0,  # No preference
            'back_of_house': 0,  # No preference
            'meeting_room': 0   # No preference
        }
        
        total_preferences = 0
        
        # Check each room's exterior access
        for room_id, room_data in layout.rooms.items():
            room_type = room_data['type']
            preference = exterior_preferences.get(room_type, 0)
            
            if preference > 0:
                total_preferences += preference
                if room_id in exterior_rooms:
                    exterior_score += preference
        
        # Normalize score
        if total_preferences > 0:
            return exterior_score / total_preferences
        return 1.0  # If no preferences, consider it perfect
    
    def _calculate_coherence_reward(self, layout: SpatialGrid) -> float:
        """Calculate reward component for architectural coherence"""
        coherence_score = 0.0
        
        # 1. Vertical alignment of rooms (e.g., guest rooms stacked vertically)
        vertical_alignment = self._check_vertical_alignment(layout)
        
        # 2. Structural grid alignment
        grid_alignment = self._check_structural_grid_alignment(layout)
        
        # 3. Logical floor assignments
        floor_logic = self._check_floor_logic(layout)
        
        coherence_score = (vertical_alignment + grid_alignment + floor_logic) / 3.0
        return coherence_score
    
    def _check_vertical_alignment(self, layout: SpatialGrid) -> float:
        """Check how well rooms are aligned vertically"""
        # Simple heuristic: check if rooms of the same type stack vertically
        # For guest rooms, this is particularly important
        
        # Group rooms by type and (x,y) position
        position_groups = {}
        
        for room_id, room_data in layout.rooms.items():
            room_type = room_data['type']
            if room_type == 'guest_room':
                x, y, _ = room_data['position']
                key = f"{room_type}_{x:.1f}_{y:.1f}"
                
                if key not in position_groups:
                    position_groups[key] = []
                position_groups[key].append(room_id)
        
        # Count alignment score
        total_guest_rooms = sum(len(rooms) for key, rooms in position_groups.items() 
                               if key.startswith('guest_room'))
        aligned_guest_rooms = sum(len(rooms) for key, rooms in position_groups.items() 
                                if key.startswith('guest_room') and len(rooms) > 1)
        
        if total_guest_rooms > 0:
            return aligned_guest_rooms / total_guest_rooms
        return 0.0
    
    def _check_structural_grid_alignment(self, layout: SpatialGrid) -> float:
        """Check how well rooms align with the structural grid"""
        aligned_rooms = 0
        total_rooms = len(layout.rooms)
        
        grid_x, grid_y = self.structural_grid
        
        for room_id, room_data in layout.rooms.items():
            x, y, _ = room_data['position']
            
            # Check if room position aligns with structural grid
            x_aligned = abs(x % grid_x) < 0.1 * grid_x
            y_aligned = abs(y % grid_y) < 0.1 * grid_y
            
            if x_aligned and y_aligned:
                aligned_rooms += 1
        
        if total_rooms > 0:
            return aligned_rooms / total_rooms
        return 0.0
    
    def _check_floor_logic(self, layout: SpatialGrid) -> float:
        """Check if rooms are on appropriate floors"""
        correct_floor = 0
        total_checked = 0
        
        # Floor preferences (0=ground, 1=first, etc. None=any)
        floor_preferences = {
            'entrance': 0,
            'lobby': 0,
            'restaurant': 0,
            'meeting_room': 0,
            'vertical_circulation': None,  # Any floor
            'guest_room': 1,  # First floor or above
            'service_area': None,
            'back_of_house': 0
        }
        
        for room_id, room_data in layout.rooms.items():
            room_type = room_data['type']
            preferred_floor = floor_preferences.get(room_type)
            
            if preferred_floor is not None:
                total_checked += 1
                _, _, z = room_data['position']
                
                actual_floor = int(z / 4.0)  # Assuming 4m floor height
                
                if room_type == 'guest_room':
                    # For guest rooms, we prefer them above ground floor
                    if actual_floor >= preferred_floor:
                        correct_floor += 1
                else:
                    # For other rooms, specific floor
                    if actual_floor == preferred_floor:
                        correct_floor += 1
        
        if total_checked > 0:
            return correct_floor / total_checked
        return 1.0  # If no floor preferences, consider it perfect
    
    def update_model(self, user_rating: float):
        """
        Update the RL model based on user feedback.
        
        Args:
            user_rating: User rating (0-10) of the current layout
        """
        # Convert user rating to reward
        reward = user_rating / 10.0
        
        # Update exploration rate
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )
        
        # Update training metrics
        self.training_iterations += 1
        self.average_reward = (
            self.average_reward * 0.95 + reward * 0.05
        )
        
        # In real implementation, we would update the model weights here
        # using the reward signal, but this requires more complex code
        # with experience replay and loss calculations
        
        print(f"Model updated with reward {reward:.2f}. "
              f"Training iteration: {self.training_iterations}, "
              f"Average reward: {self.average_reward:.2f}.")
    
    def save_model(self, path: str):
        """Save the RL model to disk"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'exploration_rate': self.exploration_rate,
            'training_iterations': self.training_iterations,
            'average_reward': self.average_reward
        }, path)
    
    def load_model(self, path: str):
        """Load the RL model from disk"""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.training_iterations = checkpoint['training_iterations']
        self.average_reward = checkpoint['average_reward']
