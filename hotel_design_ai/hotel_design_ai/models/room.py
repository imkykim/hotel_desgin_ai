from typing import Dict, Any, List, Optional, Set, Tuple
import uuid

class Room:
    """
    Represents a room or space in the hotel design.
    """
    
    def __init__(
        self,
        width: float,
        length: float,
        height: float,
        room_type: str,
        name: Optional[str] = None,
        floor: Optional[int] = None,
        min_area: Optional[float] = None,
        requires_natural_light: bool = False,
        requires_exterior_access: bool = False,
        preferred_adjacencies: Optional[List[str]] = None,
        avoid_adjacencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[int] = None
    ):
        """
        Initialize a room.
        
        Args:
            width: Width of the room in meters
            length: Length of the room in meters
            height: Height of the room in meters
            room_type: Type of room (e.g., "guest_room", "lobby", etc.)
            name: Optional name of the room
            floor: Preferred floor (0 = ground)
            min_area: Minimum area required (if flexible dimensions)
            requires_natural_light: Whether the room needs natural light
            requires_exterior_access: Whether the room needs exterior access
            preferred_adjacencies: List of room types this should be adjacent to
            avoid_adjacencies: List of room types this should not be adjacent to
            metadata: Additional room information
            id: Optional room ID (will be auto-generated if not provided)
        """
        self.width = width
        self.length = length
        self.height = height
        self.room_type = room_type
        self.name = name or f"{room_type}_{uuid.uuid4().hex[:8]}"
        self.floor = floor
        self.min_area = min_area or (width * length)
        self.requires_natural_light = requires_natural_light
        self.requires_exterior_access = requires_exterior_access
        self.preferred_adjacencies = preferred_adjacencies or []
        self.avoid_adjacencies = avoid_adjacencies or []
        self.metadata = metadata or {}
        self.id = id or int(uuid.uuid4().int % 100000)
        
        # Position will be set when placed in the layout
        self.position: Optional[Tuple[float, float, float]] = None
    
    @property
    def area(self) -> float:
        """Calculate the area of the room"""
        return self.width * self.length
    
    @property
    def volume(self) -> float:
        """Calculate the volume of the room"""
        return self.width * self.length * self.height
    
    def can_resize(self, new_width: float, new_length: float) -> bool:
        """Check if room can be resized while maintaining min area"""
        if self.min_area is None:
            return True
        return (new_width * new_length) >= self.min_area
    
    def resize(self, new_width: float, new_length: float) -> bool:
        """
        Attempt to resize the room.
        
        Returns:
            bool: True if resize was successful, False otherwise
        """
        if not self.can_resize(new_width, new_length):
            return False
        
        self.width = new_width
        self.length = new_length
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'room_type': self.room_type,
            'width': self.width,
            'length': self.length,
            'height': self.height,
            'floor': self.floor,
            'min_area': self.min_area,
            'requires_natural_light': self.requires_natural_light,
            'requires_exterior_access': self.requires_exterior_access,
            'preferred_adjacencies': self.preferred_adjacencies,
            'avoid_adjacencies': self.avoid_adjacencies,
            'position': self.position,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Room':
        """Create a Room from dictionary representation"""
        room = cls(
            width=data['width'],
            length=data['length'],
            height=data['height'],
            room_type=data['room_type'],
            name=data.get('name'),
            floor=data.get('floor'),
            min_area=data.get('min_area'),
            requires_natural_light=data.get('requires_natural_light', False),
            requires_exterior_access=data.get('requires_exterior_access', False),
            preferred_adjacencies=data.get('preferred_adjacencies', []),
            avoid_adjacencies=data.get('avoid_adjacencies', []),
            metadata=data.get('metadata', {}),
            id=data.get('id')
        )
        
        room.position = data.get('position')
        return room
    
    def __repr__(self) -> str:
        return f"Room(id={self.id}, type={self.room_type}, dim={self.width}x{self.length}x{self.height})"


class RoomFactory:
    """
    Factory class for creating common room types with default parameters.
    """
    
    @staticmethod
    def create_guest_room(
        width: float = 4.0,
        length: float = 8.0,
        height: float = 3.0,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a standard guest room"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='guest_room',
            name=name or f"GuestRoom_{uuid.uuid4().hex[:8]}",
            requires_natural_light=True,
            requires_exterior_access=True,
            preferred_adjacencies=['vertical_circulation'],
            avoid_adjacencies=['service_areas', 'back_of_house'],
            **kwargs
        )
    
    @staticmethod
    def create_lobby(
        width: float = 15.0,
        length: float = 20.0,
        height: float = 4.5,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a hotel lobby"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='lobby',
            name=name or "Main Lobby",
            floor=0,
            requires_natural_light=True,
            preferred_adjacencies=['entrance', 'vertical_circulation', 'restaurant'],
            avoid_adjacencies=['back_of_house'],
            **kwargs
        )
    
    @staticmethod
    def create_restaurant(
        width: float = 12.0,
        length: float = 15.0,
        height: float = 4.0,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a hotel restaurant"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='restaurant',
            name=name or "Restaurant",
            floor=0,
            requires_natural_light=True,
            preferred_adjacencies=['lobby', 'kitchen'],
            avoid_adjacencies=['back_of_house', 'service_areas'],
            **kwargs
        )
    
    @staticmethod
    def create_meeting_room(
        width: float = 8.0,
        length: float = 12.0,
        height: float = 3.5,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a meeting room"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='meeting_room',
            name=name or f"MeetingRoom_{uuid.uuid4().hex[:8]}",
            floor=0,
            requires_natural_light=True,
            preferred_adjacencies=['lobby', 'vertical_circulation'],
            avoid_adjacencies=['service_areas', 'back_of_house'],
            **kwargs
        )
    
    @staticmethod
    def create_vertical_circulation(
        width: float = 8.0,
        length: float = 8.0,
        height: float = 4.0,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a vertical circulation core (stairs, elevators)"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='vertical_circulation',
            name=name or "Vertical Circulation Core",
            preferred_adjacencies=['lobby', 'guest_rooms'],
            **kwargs
        )
    
    @staticmethod
    def create_service_area(
        width: float = 8.0,
        length: float = 10.0,
        height: float = 3.5,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a service area"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='service_area',
            name=name or f"ServiceArea_{uuid.uuid4().hex[:8]}",
            preferred_adjacencies=['back_of_house', 'vertical_circulation'],
            avoid_adjacencies=['lobby', 'guest_rooms'],
            **kwargs
        )
    
    @staticmethod
    def create_entrance(
        width: float = 8.0,
        length: float = 6.0,
        height: float = 4.5,
        name: Optional[str] = None,
        **kwargs
    ) -> Room:
        """Create a hotel entrance"""
        return Room(
            width=width,
            length=length,
            height=height,
            room_type='entrance',
            name=name or "Main Entrance",
            floor=0,
            requires_exterior_access=True,
            preferred_adjacencies=['lobby'],
            avoid_adjacencies=['back_of_house', 'service_areas'],
            **kwargs
        )
