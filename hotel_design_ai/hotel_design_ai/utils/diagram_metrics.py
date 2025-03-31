"""
Evaluation metrics for hotel layouts.
"""

from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
import numpy as np
import math

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.utils.geometry import distance, manhattan_distance, room_centroid


class LayoutMetrics:
    """
    Class for evaluating hotel layout metrics.
    """

    def __init__(self, layout: SpatialGrid, building_config: Dict[str, Any] = None):
        """
        Initialize with a spatial grid layout.

        Args:
            layout: The spatial grid layout to evaluate
            building_config: Building configuration parameters
        """
        self.layout = layout
        self.building_config = building_config or {"floor_height": 4.0}
        self.floor_height = self.building_config.get("floor_height", 4.0)

        # Room types that typically need natural light
        self.natural_light_types = [
            "guest_room",
            "restaurant",
            "meeting_room",
            "lobby",
            "office",
            "ballroom",
            "retail",
        ]
        self.min_floor = self.building_config.get("min_floor", -1)
        self.max_floor = self.building_config.get("max_floor", 3)

    # Helper methods
    def _get_room_floor(self, room_data):
        """Get the floor number for a room based on z-coordinate."""
        z = room_data["position"][2]
        return int(z / self.floor_height)

    def _get_room_center(self, room_data):
        """Calculate the center point of a room."""
        x, y, z = room_data["position"]
        w, l, h = room_data["dimensions"]
        return (x + w / 2, y + l / 2, z + h / 2)

    def _iterate_rooms(self, filter_func=None):
        """
        Helper to iterate through rooms with optional filtering.

        Args:
            filter_func: Optional function taking room_id and room_data
                         and returning a boolean

        Yields:
            (room_id, room_data)
        """
        for room_id, room_data in self.layout.rooms.items():
            if filter_func is None or filter_func(room_id, room_data):
                yield room_id, room_data

    def _iterate_rooms_by_type(self, room_types=None):
        """
        Helper to iterate through rooms of specific types.

        Args:
            room_types: List of room types to include, or None for all

        Yields:
            (room_id, room_data, room_type)
        """
        for room_id, room_data in self.layout.rooms.items():
            room_type = room_data["type"]
            if room_types is None or room_type in room_types:
                yield room_id, room_data, room_type

    def _calculate_weighted_score(self, metrics, weights):
        """Calculate weighted average of metrics."""
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric in weights:
                weight = weights.get(metric, 0)
                weighted_sum += value * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def space_utilization(self) -> float:
        """
        Calculate the space utilization ratio (occupied volume / total volume).

        Returns:
            float: Space utilization ratio (0.0 to 1.0)
        """
        return self.layout.calculate_space_utilization()

    def floor_area_ratio(self) -> float:
        """
        Calculate Floor Area Ratio (FAR) - ratio of floor area to site area.

        Returns:
            float: Floor Area Ratio
        """
        # Calculate total floor area
        total_floor_area = sum(
            room_data["dimensions"][0] * room_data["dimensions"][1]
            for room_data in self.layout.rooms.values()
        )

        # Site area
        site_area = self.layout.width * self.layout.length

        # FAR
        return total_floor_area / site_area if site_area > 0 else 0.0

    def circulation_efficiency(self) -> Dict[str, float]:
        """
        Calculate circulation efficiency metrics.

        Returns:
            Dict: Circulation metrics including:
                - avg_corridor_length: Average corridor length
                - circulation_ratio: Ratio of circulation to total area
        """
        # Identify circulation spaces
        circulation_rooms = [
            room_data
            for room_id, room_data in self._iterate_rooms()
            if room_data["type"] in ["vertical_circulation", "circulation"]
        ]

        # Calculate circulation area and total floor area
        circulation_area = sum(
            room["dimensions"][0] * room["dimensions"][1] for room in circulation_rooms
        )

        total_floor_area = sum(
            room["dimensions"][0] * room["dimensions"][1]
            for room in self.layout.rooms.values()
        )

        # Calculate circulation ratio
        circulation_ratio = (
            circulation_area / total_floor_area if total_floor_area > 0 else 0.0
        )

        # Calculate corridor metrics if we have vertical circulation
        vertical_circulation_rooms = [
            room for room in circulation_rooms if room["type"] == "vertical_circulation"
        ]

        avg_corridor_length = 0.0
        if vertical_circulation_rooms:
            # Get circulation centers
            vertical_circulation_centers = [
                self._get_room_center(room) for room in vertical_circulation_rooms
            ]

            if vertical_circulation_centers:
                # Calculate average distance from each room to nearest circulation
                distances = []

                for room_id, room_data in self._iterate_rooms():
                    # Skip circulation rooms themselves
                    if room_data["type"] in ["vertical_circulation", "circulation"]:
                        continue

                    # Calculate room center
                    room_center = self._get_room_center(room_data)

                    # Find closest circulation on same floor
                    min_dist = float("inf")
                    for circ_center in vertical_circulation_centers:
                        # Only consider distance within same floor
                        if abs(room_center[2] - circ_center[2]) < 0.1:
                            dist = manhattan_distance(
                                (room_center[0], room_center[1], 0),
                                (circ_center[0], circ_center[1], 0),
                            )
                            min_dist = min(min_dist, dist)

                    if min_dist < float("inf"):
                        distances.append(min_dist)

                if distances:
                    avg_corridor_length = sum(distances) / len(distances)

        return {
            "circulation_ratio": circulation_ratio,
            "avg_corridor_length": avg_corridor_length,
        }

    def natural_light_access(self) -> float:
        """
        Calculate the ratio of rooms with natural light (exterior access).

        Returns:
            float: Ratio of rooms with natural light (0.0 to 1.0)
        """
        # Get all exterior rooms
        exterior_rooms = set(self.layout.get_exterior_rooms())

        # Count rooms needing light and those that have it
        rooms_needing_light = []
        rooms_with_light = []

        for room_id, _, room_type in self._iterate_rooms_by_type(
            self.natural_light_types
        ):
            rooms_needing_light.append(room_id)
            if room_id in exterior_rooms:
                rooms_with_light.append(room_id)

        # Calculate ratio
        if not rooms_needing_light:
            return 1.0  # No rooms need light

        return len(rooms_with_light) / len(rooms_needing_light)

    def structural_alignment(self, structural_grid: Tuple[float, float]) -> float:
        """
        Calculate how well rooms align with the structural grid.

        Args:
            structural_grid: (x_spacing, y_spacing) of the structural grid

        Returns:
            float: Ratio of rooms aligned with grid (0.0 to 1.0)
        """
        grid_x, grid_y = structural_grid
        aligned_rooms = 0
        total_rooms = len(self.layout.rooms)

        if total_rooms == 0:
            return 1.0  # No rooms to check

        # Define tolerance as a percentage of grid spacing
        tolerance_x = 0.05 * grid_x
        tolerance_y = 0.05 * grid_y

        for _, room_data in self._iterate_rooms():
            x, y, _ = room_data["position"]

            # Check if room position aligns with grid lines
            x_offset = x % grid_x
            y_offset = y % grid_y

            # Room is aligned if it's close to a grid line
            x_aligned = min(x_offset, grid_x - x_offset) < tolerance_x
            y_aligned = min(y_offset, grid_y - y_offset) < tolerance_y

            if x_aligned and y_aligned:
                aligned_rooms += 1

        return aligned_rooms / total_rooms

    def department_clustering(self, room_departments: Dict[str, str]) -> float:
        """
        Calculate how well rooms of the same department are clustered together.

        Args:
            room_departments: Dict mapping room_id to department

        Returns:
            float: Clustering score (0.0 to 1.0)
        """
        # Group rooms by department
        departments = {}

        for room_id, room_data in self._iterate_rooms():
            if room_id in room_departments:
                dept = room_departments[room_id]
                if dept not in departments:
                    departments[dept] = []
                departments[dept].append(room_id)

        # Calculate clustering score for each department
        dept_scores = []

        for dept, room_ids in departments.items():
            if len(room_ids) <= 1:
                # Only one room in this department, so it's trivially clustered
                dept_scores.append(1.0)
                continue

            # Calculate room centers
            centers = [
                self._get_room_center(self.layout.rooms[room_id])
                for room_id in room_ids
            ]

            # Calculate average internal distance (within department)
            internal_distances = [
                manhattan_distance(centers[i], centers[j])
                for i in range(len(centers))
                for j in range(i + 1, len(centers))
            ]

            avg_internal_distance = (
                sum(internal_distances) / len(internal_distances)
                if internal_distances
                else 0
            )

            # Calculate average distance to other departments
            other_centers = []
            for other_dept, other_room_ids in departments.items():
                if other_dept != dept:
                    other_centers.extend(
                        [
                            self._get_room_center(self.layout.rooms[room_id])
                            for room_id in other_room_ids
                        ]
                    )

            # Calculate min distance from each room in this dept to any room in other depts
            if other_centers:
                min_external_distances = [
                    min(
                        manhattan_distance(center, other_center)
                        for other_center in other_centers
                    )
                    for center in centers
                ]

                avg_external_distance = (
                    sum(min_external_distances) / len(min_external_distances)
                    if min_external_distances
                    else 0
                )

                # Score is ratio of external to internal distances (capped at 1.0)
                dept_score = min(
                    1.0, avg_external_distance / (avg_internal_distance + 0.1)
                )
            else:
                dept_score = 1.0  # No other departments to compare with

            dept_scores.append(dept_score)

        # Overall score is average of department scores
        return sum(dept_scores) / len(dept_scores) if dept_scores else 1.0

    def vertical_stacking(self) -> float:
        """
        Calculate how well rooms are stacked vertically (for multi-floor buildings).

        Returns:
            float: Stacking score (0.0 to 1.0)
        """
        # Group rooms by type
        rooms_by_type = {}

        for room_id, room_data, room_type in self._iterate_rooms_by_type():
            if room_type not in rooms_by_type:
                rooms_by_type[room_type] = []
            rooms_by_type[room_type].append(room_data)

        # Only consider room types that should typically be stacked
        stacking_types = ["guest_room", "meeting_room", "office"]
        stacking_scores = []

        for room_type in stacking_types:
            if room_type not in rooms_by_type or len(rooms_by_type[room_type]) <= 1:
                continue

            rooms = rooms_by_type[room_type]

            # Group rooms by (x, y) position (ignoring z)
            position_groups = {}

            for room in rooms:
                x, y, _ = room["position"]
                w, l, _ = room["dimensions"]

                # Use center point's x,y as key
                center_x = x + w / 2
                center_y = y + l / 2

                # Round to reduce floating point precision issues
                key = (round(center_x, 1), round(center_y, 1))

                if key not in position_groups:
                    position_groups[key] = []
                position_groups[key].append(room)

            # Calculate stacking score
            total_rooms = len(rooms)
            stacked_rooms = sum(
                len(group) for group in position_groups.values() if len(group) > 1
            )

            # Score is ratio of stacked rooms to total rooms
            if total_rooms > 0:
                stacking_scores.append(stacked_rooms / total_rooms)

        # Overall score is average of stacking scores
        return sum(stacking_scores) / len(stacking_scores) if stacking_scores else 1.0

    def calculate_space_utilization(self) -> float:
        """
        Calculate the space utilization ratio (occupied volume / total volume).
        Enhanced to focus on efficiency in no-corridor layouts.

        Returns:
            float: Space utilization ratio (0.0 to 1.0)
        """
        return self.layout.calculate_space_utilization()

    def calculate_compactness(self) -> float:
        """
        Calculate layout compactness as a measure of tightness of packing.

        Returns:
            float: Compactness score (0.0 to 1.0)
        """
        # Calculate bounding box area of all rooms
        min_x, min_y, max_x, max_y = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )

        # Get bounding box for all rooms per floor
        floor_rooms = defaultdict(list)
        for room_id, room_data in self.layout.rooms.items():
            x, y, z = room_data["position"]
            w, l, h = room_data["dimensions"]

            # Determine floor
            floor = int(z / self.floor_height)
            floor_rooms[floor].append((x, y, w, l))

            # Update overall bounding box
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + l)

        # Calculate overall compactness
        building_area = (max_x - min_x) * (max_y - min_y)
        room_area = sum(
            room_data["dimensions"][0] * room_data["dimensions"][1]
            for room_data in self.layout.rooms.values()
        )

        overall_compactness = room_area / building_area if building_area > 0 else 0.0

        # Calculate per-floor compactness and take the average
        floor_compactness_values = []
        for floor, rooms in floor_rooms.items():
            if not rooms:
                continue

            # Calculate floor bounding box
            floor_min_x = min(x for x, _, _, _ in rooms)
            floor_min_y = min(y for _, y, _, _ in rooms)
            floor_max_x = max(x + w for x, y, w, _ in rooms)
            floor_max_y = max(y + l for _, y, _, l in rooms)

            floor_bbox_area = (floor_max_x - floor_min_x) * (floor_max_y - floor_min_y)
            floor_room_area = sum(w * l for _, _, w, l in rooms)

            floor_compactness = (
                floor_room_area / floor_bbox_area if floor_bbox_area > 0 else 0.0
            )
            floor_compactness_values.append(floor_compactness)

        # Combine overall and per-floor metrics
        if floor_compactness_values:
            avg_floor_compactness = sum(floor_compactness_values) / len(
                floor_compactness_values
            )
            # Weight overall compactness more, but consider per-floor compactness
            return 0.7 * overall_compactness + 0.3 * avg_floor_compactness
        else:
            return overall_compactness

    def wall_sharing_efficiency(self) -> float:
        """
        Calculate the efficiency of wall sharing between rooms.
        Higher score means more shared walls and better space efficiency.

        Returns:
            float: Wall sharing efficiency (0.0 to 1.0)
        """
        # Track shared vs. non-shared wall lengths
        total_wall_length = 0.0
        shared_wall_length = 0.0

        # Analyze each floor separately
        for floor in range(self.min_floor, self.max_floor + 1):
            # Get rooms on this floor
            z_min = floor * self.floor_height
            z_max = (floor + 1) * self.floor_height

            floor_rooms = []
            for room_id, room_data in self.layout.rooms.items():
                x, y, z = room_data["position"]
                if z_min <= z < z_max:
                    floor_rooms.append(room_data)

            # Skip if floor is empty
            if not floor_rooms:
                continue

            # Calculate all walls and check for sharing
            walls = []  # (x1, y1, x2, y2, room_id) format

            for room_data in floor_rooms:
                room_id = room_data["id"]
                x, y, z = room_data["position"]
                w, l, h = room_data["dimensions"]

                # Create the four walls
                walls.append(((x, y), (x + w, y), room_id))  # Bottom
                walls.append(((x + w, y), (x + w, y + l), room_id))  # Right
                walls.append(((x, y + l), (x + w, y + l), room_id))  # Top
                walls.append(((x, y), (x, y + l), room_id))  # Left

                # Add to total wall length
                total_wall_length += 2 * (w + l)

            # Check for shared walls
            for i, ((x1, y1), (x2, y2), id1) in enumerate(walls):
                wall_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                # Look for matching walls from other rooms
                for j, ((x3, y3), (x4, y4), id2) in enumerate(walls):
                    if id1 == id2:  # Skip same room
                        continue

                    # Check if walls are collinear and overlapping
                    if abs(x1 - x2) < 0.01 and abs(x3 - x4) < 0.01:
                        # Vertical walls
                        if abs(x1 - x3) < 0.01:
                            # Calculate overlap
                            y_min = max(min(y1, y2), min(y3, y4))
                            y_max = min(max(y1, y2), max(y3, y4))
                            overlap = max(0, y_max - y_min)
                            shared_wall_length += overlap

                    elif abs(y1 - y2) < 0.01 and abs(y3 - y4) < 0.01:
                        # Horizontal walls
                        if abs(y1 - y3) < 0.01:
                            # Calculate overlap
                            x_min = max(min(x1, x2), min(x3, x4))
                            x_max = min(max(x1, x2), max(x3, x4))
                            overlap = max(0, x_max - x_min)
                            shared_wall_length += overlap

        # Avoid counting walls twice
        shared_wall_length /= 2

        # Calculate efficiency - what percentage of walls are shared
        if total_wall_length > 0:
            return shared_wall_length / total_wall_length
        else:
            return 0.0

    def adjacency_satisfaction(
        self, adjacency_preferences: Dict[str, List[str]]
    ) -> float:
        """
        Enhanced adjacency satisfaction metric that focuses on direct adjacency.

        Args:
            adjacency_preferences: Dict mapping room types to list of preferred adjacent types

        Returns:
            float: Ratio of satisfied adjacencies (0.0 to 1.0)
        """
        # Track satisfied adjacencies vs. total requirements
        satisfied_adjacencies = 0
        total_adjacencies = 0

        # For each room, check if its preferred adjacencies are satisfied
        for room_id, room_data in self.layout.rooms.items():
            room_type = room_data["type"]
            if room_type not in adjacency_preferences:
                continue

            # Get preferred adjacencies for this room type
            preferred_types = adjacency_preferences[room_type]

            # Skip if no preferences
            if not preferred_types:
                continue

            # Get actual neighbors
            neighbors = self.layout.get_room_neighbors(room_id)
            neighbor_types = [
                self.layout.rooms[n]["type"]
                for n in neighbors
                if n in self.layout.rooms
            ]

            # For each preferred type, check if satisfied
            for pref_type in preferred_types:
                total_adjacencies += 1

                if pref_type in neighbor_types:
                    satisfied_adjacencies += 1

        # Calculate satisfaction ratio
        if total_adjacencies > 0:
            return satisfied_adjacencies / total_adjacencies
        else:
            return 1.0  # No preferences = fully satisfied

    def departmental_connectivity(self, room_departments: Dict[int, str]) -> float:
        """
        Evaluate how well departments are internally connected.

        Args:
            room_departments: Dict mapping room_id to department

        Returns:
            float: Department connectivity score (0.0 to 1.0)
        """
        # Group rooms by department
        departments = defaultdict(list)
        for room_id, dept in room_departments.items():
            if room_id in self.layout.rooms:
                departments[dept].append(room_id)

        # Calculate connectivity by department
        dept_scores = []

        for dept, room_ids in departments.items():
            if len(room_ids) <= 1:
                # Single room departments are fully connected
                dept_scores.append(1.0)
                continue

            # Check adjacencies within department
            internal_connections = 0
            possible_connections = len(room_ids) * (len(room_ids) - 1) / 2  # n choose 2

            for i, room1 in enumerate(room_ids):
                for room2 in room_ids[i + 1 :]:
                    if self.layout.are_adjacent(room1, room2):
                        internal_connections += 1

            # Calculate department score
            dept_score = (
                internal_connections / possible_connections
                if possible_connections > 0
                else 1.0
            )
            dept_scores.append(dept_score)

        # Average department scores
        if dept_scores:
            return sum(dept_scores) / len(dept_scores)
        else:
            return 0.0

    def evaluate_all(
        self,
        adjacency_preferences: Optional[Dict[str, List[str]]] = None,
        room_departments: Optional[Dict[str, str]] = None,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for space-focused layouts.

        Args:
            adjacency_preferences: Dict of preferred adjacencies
            room_departments: Dict mapping room_id to department
            structural_grid: Structural grid spacing
            weights: Dict of metric weights

        Returns:
            Dict: All metrics and weighted score
        """
        # Default weights for space-focused layout
        default_weights = {
            "space_utilization": 1.0,
            "compactness": 2.0,  # New: high priority for compactness
            "wall_sharing": 1.5,  # New: wall sharing efficiency
            "adjacency_satisfaction": 2.0,
            "departmental_connectivity": 1.5,  # New: internal dept connectivity
            "natural_light_access": 1.0,  # Lower priority in space-focused layout
            "structural_alignment": 0.8,  # Lower priority
        }

        # Use provided weights or defaults
        metric_weights = weights or default_weights

        # Calculate metrics
        metrics = {
            "space_utilization": self.space_utilization(),
            "compactness": self.calculate_compactness(),
            "wall_sharing": self.wall_sharing_efficiency(),
            "natural_light_access": self.natural_light_access(),
            "structural_alignment": self.structural_alignment(structural_grid),
        }

        # Add metrics that require additional inputs
        if adjacency_preferences:
            metrics["adjacency_satisfaction"] = self.adjacency_satisfaction(
                adjacency_preferences
            )

        if room_departments:
            metrics["departmental_connectivity"] = self.departmental_connectivity(
                room_departments
            )

        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric in metric_weights:
                weight = metric_weights[metric]
                weighted_sum += value * weight
                total_weight += weight

        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Add overall score to results
        metrics["overall_score"] = overall_score

        return metrics


def calculate_occupant_loads(layout: SpatialGrid) -> Dict[int, float]:
    """
    Calculate the occupant load for each room based on standard occupancy factors.

    Args:
        layout: Spatial grid layout

    Returns:
        Dict[int, float]: Mapping of room_id to occupant load
    """
    # Occupancy factors in square meters per person
    # Based on IBC (International Building Code) standards
    occupancy_factors = {
        "guest_room": 18.6,  # Hotel rooms
        "lobby": 1.4,  # Assembly areas
        "restaurant": 1.4,  # Dining areas
        "kitchen": 18.6,  # Commercial kitchens
        "meeting_room": 1.9,  # Conference rooms
        "ballroom": 0.7,  # Assembly areas
        "office": 9.3,  # Business areas
        "retail": 5.6,  # Mercantile
        "fitness": 4.6,  # Exercise rooms
        "pool": 4.6,  # Swimming pools
        "lounge": 1.4,  # Assembly areas
        # Default for other areas
        "default": 9.3,
    }

    # Calculate occupant load for each room
    occupant_loads = {}

    for room_id, room_data in layout.rooms.items():
        room_type = room_data["type"]
        width, length, _ = room_data["dimensions"]

        # Calculate floor area
        area = width * length

        # Get occupancy factor
        factor = occupancy_factors.get(room_type, occupancy_factors["default"])

        # Calculate occupant load (round up)
        load = math.ceil(area / factor)

        occupant_loads[room_id] = load

    return occupant_loads


def calculate_egress_capacity(
    layout: SpatialGrid, building_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate egress capacity requirements based on occupant loads.

    Args:
        layout: Spatial grid layout
        building_config: Building configuration parameters

    Returns:
        Dict: Egress capacity metrics
    """
    # Set default building config if not provided
    building_config = building_config or {"floor_height": 4.0}
    floor_height = building_config.get("floor_height", 4.0)

    # Calculate occupant loads
    occupant_loads = calculate_occupant_loads(layout)

    # Sum total occupants
    total_occupants = sum(occupant_loads.values())

    # Group by floor
    occupants_by_floor = {}

    for room_id, load in occupant_loads.items():
        room_data = layout.rooms[room_id]
        z = room_data["position"][2]

        # Determine floor using floor height from building config
        floor = int(z / floor_height)

        if floor not in occupants_by_floor:
            occupants_by_floor[floor] = 0

        occupants_by_floor[floor] += load

    # Calculate required exits and widths
    # Based on IBC requirements:
    # - Occupant load 1-500: 2 exits required
    # - Occupant load 501-1000: 3 exits required
    # - Occupant load >1000: 4 exits required
    # - Exit width: 5mm per occupant

    required_exits = {}
    required_exit_width = {}

    for floor, occupants in occupants_by_floor.items():
        # Determine number of required exits
        if occupants <= 500:
            num_exits = 2
        elif occupants <= 1000:
            num_exits = 3
        else:
            num_exits = 4

        # Calculate required exit width (in meters)
        # 5mm per occupant, converted to meters
        width = occupants * 0.005

        required_exits[floor] = num_exits
        required_exit_width[floor] = width

    return {
        "total_occupants": total_occupants,
        "occupants_by_floor": occupants_by_floor,
        "required_exits": required_exits,
        "required_exit_width": required_exit_width,
    }


def calculate_daylight_factor(layout: SpatialGrid) -> Dict[int, float]:
    """
    Estimate daylight factor for each room based on distance from exterior.
    This is a simplified estimation as real daylight factor requires complex simulation.

    Args:
        layout: Spatial grid layout

    Returns:
        Dict[int, float]: Mapping of room_id to estimated daylight factor
    """
    # Get exterior rooms
    exterior_rooms = set(layout.get_exterior_rooms())

    # Estimate daylight factor for each room
    daylight_factors = {}

    max_distance = math.sqrt(layout.width**2 + layout.length**2)

    for room_id, room_data in layout.rooms.items():
        # Exterior rooms get high daylight factor
        if room_id in exterior_rooms:
            # Calculate room center
            x, y, z = room_data["position"]
            w, l, h = room_data["dimensions"]
            center = (x + w / 2, y + l / 2, z + h / 2)

            # Estimate based on distance from center of room to exterior
            # This is a very simplified approach
            min_dist_to_edge = min(
                center[0],  # Distance to left edge
                layout.width - center[0],  # Distance to right edge
                center[1],  # Distance to bottom edge
                layout.length - center[1],  # Distance to top edge
            )

            # Normalize distance (0 = at edge, 1 = furthest from edge)
            normalized_dist = 1 - (min_dist_to_edge / (max_distance / 2))

            # Scale to typical daylight factor range (0-5%)
            factor = 5.0 * normalized_dist

            daylight_factors[room_id] = factor
        else:
            # Interior rooms get low daylight factor
            daylight_factors[room_id] = 0.5  # Minimal daylight factor

    return daylight_factors


def calculate_energy_efficiency(layout: SpatialGrid) -> Dict[str, float]:
    """
    Estimate energy efficiency metrics based on layout properties.

    Args:
        layout: Spatial grid layout

    Returns:
        Dict: Energy efficiency metrics
    """
    # Calculate exterior surface area
    building_surfaces = [
        # Bottom surface (z=0)
        (0, 0, 0, layout.width, layout.length),
        # Top surface (z=height)
        (0, 0, layout.height, layout.width, layout.length),
        # Front surface (y=0)
        (0, 0, 0, layout.width, layout.height),
        # Back surface (y=length)
        (0, layout.length, 0, layout.width, layout.height),
        # Left surface (x=0)
        (0, 0, 0, layout.length, layout.height),
        # Right surface (x=width)
        (layout.width, 0, 0, layout.length, layout.height),
    ]

    # Calculate areas of each surface
    exterior_area = sum(surface[3] * surface[4] for surface in building_surfaces)

    # Calculate total volume
    total_volume = layout.width * layout.length * layout.height

    # Calculate surface-to-volume ratio
    surface_to_volume = exterior_area / total_volume if total_volume > 0 else 0

    # Calculate window-to-wall ratio (estimating windows on exterior rooms)
    exterior_rooms = layout.get_exterior_rooms()

    # Estimate windows as 40% of exterior wall area for exterior rooms
    wall_area = 0.0

    for room_id in exterior_rooms:
        room_data = layout.rooms[room_id]
        x, y, z = room_data["position"]
        w, l, h = room_data["dimensions"]

        # Determine which faces are exterior
        is_at_left = abs(x) < 0.1
        is_at_right = abs(x + w - layout.width) < 0.1
        is_at_front = abs(y) < 0.1
        is_at_back = abs(y + l - layout.length) < 0.1

        # Calculate wall areas
        if is_at_left or is_at_right:
            wall_area += l * h
        if is_at_front or is_at_back:
            wall_area += w * h

    # Estimate window area (40% of exterior walls for habitable rooms)
    window_area = 0.4 * wall_area

    # Calculate window-to-wall ratio
    window_to_wall = window_area / wall_area if wall_area > 0 else 0

    return {
        "surface_to_volume": surface_to_volume,
        "window_to_wall": window_to_wall,
        "exterior_area": exterior_area,
        "total_volume": total_volume,
    }


def calculate_program_compliance(
    layout: SpatialGrid, program_requirements: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Calculate how well the layout complies with program requirements.

    Args:
        layout: Spatial grid layout
        program_requirements: Dict mapping room types to required areas

    Returns:
        Dict: Program compliance metrics
    """
    # Group rooms by type and calculate areas
    total_area_by_type = {}

    for room_id, room_data in layout.rooms.items():
        room_type = room_data["type"]
        w, l, _ = room_data["dimensions"]
        area = w * l

        if room_type not in total_area_by_type:
            total_area_by_type[room_type] = 0

        total_area_by_type[room_type] += area

    # Calculate compliance for each room type
    compliance_by_type = {}
    actual_vs_required = {}

    for room_type, required in program_requirements.items():
        # Extract required area
        if isinstance(required, dict) and "area" in required:
            required_area = required["area"]
        else:
            required_area = required

        # Get actual area
        actual_area = total_area_by_type.get(room_type, 0)

        # Calculate compliance percentage
        compliance = actual_area / required_area if required_area > 0 else 0
        compliance_by_type[room_type] = compliance

        # Track actual vs required
        actual_vs_required[room_type] = {
            "required": required_area,
            "actual": actual_area,
            "difference": actual_area - required_area,
            "compliance": compliance,
        }

    # Calculate overall compliance (weighted by area requirements)
    total_required = sum(
        req["area"] if isinstance(req, dict) else req
        for req in program_requirements.values()
    )

    # Calculate weighted compliance
    weighted_compliance = 0.0
    for room_type, required in program_requirements.items():
        required_area = required["area"] if isinstance(required, dict) else required
        compliance = compliance_by_type.get(room_type, 0)

        weight = required_area / total_required if total_required > 0 else 0
        weighted_compliance += compliance * weight

    return {
        "overall_compliance": weighted_compliance,
        "total_area_by_type": total_area_by_type,
        "compliance_by_type": compliance_by_type,
        "actual_vs_required": actual_vs_required,
    }
