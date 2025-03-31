"""
Utilities for building configuration compatibility.
This helps maintain compatibility with older components when using the new
podium/standard floor building configuration format.
"""


def convert_new_to_old_building_config(new_config):
    """
    Convert new building config format with podium/standard floor sections
    to the old format for backward compatibility.

    Args:
        new_config: Building config in new format

    Returns:
        Dict: Building config in old format
    """
    # Start with a copy of the original config
    old_config = new_config.copy()

    # Remove podium and standard_floor sections
    if "podium" in old_config:
        del old_config["podium"]
    if "standard_floor" in old_config:
        del old_config["standard_floor"]

    return old_config


def is_podium_floor(floor, building_config):
    """
    Check if a floor is in the podium section.

    Args:
        floor: Floor number to check
        building_config: Building configuration

    Returns:
        bool: True if it's a podium floor
    """
    podium_config = building_config.get("podium", {})
    min_floor = podium_config.get("min_floor", building_config.get("min_floor", -2))
    max_floor = podium_config.get("max_floor", 4)

    return min_floor <= floor <= max_floor


def is_standard_floor(floor, building_config):
    """
    Check if a floor is in the standard floor (tower) section.

    Args:
        floor: Floor number to check
        building_config: Building configuration

    Returns:
        bool: True if it's a standard floor
    """
    std_floor_config = building_config.get("standard_floor", {})
    min_floor = std_floor_config.get("start_floor", 5)
    max_floor = std_floor_config.get("end_floor", 20)

    return min_floor <= floor <= max_floor


def get_room_section(room, building_config):
    """
    Determine which section (podium or standard floor) a room belongs to.

    Args:
        room: Room object
        building_config: Building configuration

    Returns:
        str: 'podium', 'standard_floor', or 'unknown'
    """
    # Check if floor is specified
    if not hasattr(room, "floor") or room.floor is None:
        # Default to podium for rooms without floor
        return "podium"

    # Get floor
    floor = room.floor

    # Check if it's a podium floor
    if is_podium_floor(floor, building_config):
        return "podium"

    # Check if it's a standard floor
    if is_standard_floor(floor, building_config):
        return "standard_floor"

    # Unknown section
    return "unknown"


def tag_rooms_with_section(rooms, building_config):
    """
    Tag rooms with their section information in metadata.

    Args:
        rooms: List of Room objects
        building_config: Building configuration

    Returns:
        List[Room]: Same rooms with updated metadata
    """
    for room in rooms:
        # Initialize metadata if needed
        if not hasattr(room, "metadata") or room.metadata is None:
            room.metadata = {}

        # Add section tag
        section = get_room_section(room, building_config)
        room.metadata["section"] = section

        # Add flag for standard floor rooms
        if section == "standard_floor":
            room.metadata["is_standard_floor_room"] = True
        elif section == "podium":
            room.metadata["is_podium_room"] = True

    return rooms
