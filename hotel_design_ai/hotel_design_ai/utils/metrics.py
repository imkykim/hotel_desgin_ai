"""
Evaluation metrics for hotel layouts.
"""

from typing import Dict, List, Tuple, Set, Any, Optional
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
        total_floor_area = 0.0

        for _, room_data in self.layout.rooms.items():
            width, length, _ = room_data["dimensions"]
            total_floor_area += width * length

        # Site area
        site_area = self.layout.width * self.layout.length

        # FAR
        if site_area > 0:
            return total_floor_area / site_area
        return 0.0

    def circulation_efficiency(self) -> Dict[str, float]:
        """
        Calculate circulation efficiency metrics.

        Returns:
            Dict: Circulation metrics including:
                - avg_corridor_length: Average corridor length
                - circulation_ratio: Ratio of circulation to total area
        """
        # Identify circulation spaces
        circulation_rooms = []

        for room_id, room_data in self.layout.rooms.items():
            if room_data["type"] in ["vertical_circulation", "circulation"]:
                circulation_rooms.append(room_data)

        # Calculate circulation area
        circulation_area = sum(
            room["dimensions"][0] * room["dimensions"][1] for room in circulation_rooms
        )

        # Calculate total floor area
        total_floor_area = sum(
            room["dimensions"][0] * room["dimensions"][1]
            for room in self.layout.rooms.values()
        )

        # Calculate circulation ratio
        circulation_ratio = 0.0
        if total_floor_area > 0:
            circulation_ratio = circulation_area / total_floor_area

        # Calculate corridor metrics if we have vertical circulation
        vertical_circulation_rooms = [
            room for room in circulation_rooms if room["type"] == "vertical_circulation"
        ]

        avg_corridor_length = 0.0
        if vertical_circulation_rooms:
            # Get all room centers and calculate distances
            vertical_circulation_centers = []
            for room in vertical_circulation_rooms:
                x, y, z = room["position"]
                w, l, h = room["dimensions"]
                center = (x + w / 2, y + l / 2, z + h / 2)
                vertical_circulation_centers.append(center)

            if vertical_circulation_centers:
                # Calculate average distance from each room to nearest circulation
                total_distance = 0.0
                room_count = 0

                for room_id, room_data in self.layout.rooms.items():
                    # Skip circulation rooms themselves
                    if room_data["type"] in ["vertical_circulation", "circulation"]:
                        continue

                    # Calculate room center
                    x, y, z = room_data["position"]
                    w, l, h = room_data["dimensions"]
                    room_center = (x + w / 2, y + l / 2, z + h / 2)

                    # Find closest circulation
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
                        total_distance += min_dist
                        room_count += 1

                if room_count > 0:
                    avg_corridor_length = total_distance / room_count

        return {
            "circulation_ratio": circulation_ratio,
            "avg_corridor_length": avg_corridor_length,
        }

    def adjacency_satisfaction(
        self, adjacency_preferences: Dict[str, List[str]]
    ) -> float:
        """
        Calculate how well adjacency relationships are satisfied.

        Args:
            adjacency_preferences: Dict mapping room types to list of preferred adjacent types

        Returns:
            float: Ratio of satisfied adjacencies (0.0 to 1.0)
        """
        # Track satisfied and total adjacency requirements
        satisfied = 0
        total = 0

        # For each room type with adjacency preferences
        for room_type, preferred_adjacencies in adjacency_preferences.items():
            # Find all rooms of this type
            rooms_of_type = []

            for room_id, room_data in self.layout.rooms.items():
                if room_data["type"] == room_type:
                    rooms_of_type.append(room_id)

            # Skip if no rooms of this type
            if not rooms_of_type:
                continue

            # For each room of this type
            for room_id in rooms_of_type:
                # Get neighbors
                neighbors = self.layout.get_room_neighbors(room_id)

                # Check if neighbors include the preferred types
                for preferred_type in preferred_adjacencies:
                    # Increment total requirements
                    total += 1

                    # Check if any neighbor is of the preferred type
                    satisfied_for_this_room = False
                    for neighbor_id in neighbors:
                        if neighbor_id in self.layout.rooms:
                            if self.layout.rooms[neighbor_id]["type"] == preferred_type:
                                satisfied_for_this_room = True
                                break

                    if satisfied_for_this_room:
                        satisfied += 1

        # Calculate ratio of satisfied adjacencies
        if total > 0:
            return satisfied / total
        return 1.0  # No adjacency requirements to satisfy

    def natural_light_access(self) -> float:
        """
        Calculate the ratio of rooms with natural light (exterior access).

        Returns:
            float: Ratio of rooms with natural light (0.0 to 1.0)
        """
        # Get all exterior rooms
        exterior_rooms = set(self.layout.get_exterior_rooms())

        # Count rooms that need natural light
        rooms_needing_light = []

        natural_light_types = [
            "guest_room",
            "restaurant",
            "meeting_room",
            "lobby",
            "office",
            "ballroom",
        ]

        for room_id, room_data in self.layout.rooms.items():
            if room_data["type"] in natural_light_types:
                rooms_needing_light.append(room_id)

        # Calculate ratio
        if not rooms_needing_light:
            return 1.0  # No rooms need light

        rooms_with_light = len([r for r in rooms_needing_light if r in exterior_rooms])
        return rooms_with_light / len(rooms_needing_light)

    def structural_alignment(self, structural_grid: Tuple[float, float]) -> float:
        """
        Calculate how well rooms align with the structural grid.

        Args:
            structural_grid: (x_spacing, y_spacing) of the structural grid

        Returns:
            float: Ratio of rooms aligned with grid (0.0 to 1.0)
        """
        aligned_rooms = 0
        total_rooms = len(self.layout.rooms)

        if total_rooms == 0:
            return 1.0  # No rooms to check

        grid_x, grid_y = structural_grid

        # Define tolerance as a percentage of grid spacing
        tolerance_x = 0.05 * grid_x
        tolerance_y = 0.05 * grid_y

        for room_id, room_data in self.layout.rooms.items():
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

        for room_id, room_data in self.layout.rooms.items():
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
            centers = []
            for room_id in room_ids:
                room_data = self.layout.rooms[room_id]
                pos = room_data["position"]
                dim = room_data["dimensions"]
                center = room_centroid(pos, dim)
                centers.append(center)

            # Calculate average distance between all pairs in this department
            total_distance = 0.0
            pair_count = 0

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = manhattan_distance(centers[i], centers[j])
                    total_distance += dist
                    pair_count += 1

            avg_distance = total_distance / pair_count if pair_count > 0 else 0

            # Calculate average distance to other departments
            other_centers = []
            for other_dept, other_room_ids in departments.items():
                if other_dept != dept:
                    for room_id in other_room_ids:
                        room_data = self.layout.rooms[room_id]
                        pos = room_data["position"]
                        dim = room_data["dimensions"]
                        center = room_centroid(pos, dim)
                        other_centers.append(center)

            # Calculate min distance from each room in this dept to any room in other depts
            min_distances = []

            for center in centers:
                if other_centers:
                    min_dist = min(
                        manhattan_distance(center, other_center)
                        for other_center in other_centers
                    )
                    min_distances.append(min_dist)

            avg_min_distance = (
                sum(min_distances) / len(min_distances) if min_distances else 0
            )

            # Score is ratio of external to internal distances (capped at 1.0)
            dept_score = min(1.0, avg_min_distance / (avg_distance + 0.1))
            dept_scores.append(dept_score)

        # Overall score is average of department scores
        if dept_scores:
            return sum(dept_scores) / len(dept_scores)
        return 1.0  # No departments to check

    def vertical_stacking(self) -> float:
        """
        Calculate how well rooms are stacked vertically (for multi-floor buildings).

        Returns:
            float: Stacking score (0.0 to 1.0)
        """
        # Group rooms by type
        rooms_by_type = {}

        for room_id, room_data in self.layout.rooms.items():
            room_type = room_data["type"]
            if room_type not in rooms_by_type:
                rooms_by_type[room_type] = []
            rooms_by_type[room_type].append(room_data)

        # Only consider guest rooms and similar repeating room types
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
        if stacking_scores:
            return sum(stacking_scores) / len(stacking_scores)
        return 1.0  # No rooms to check for stacking

    def evaluate_all(
        self,
        adjacency_preferences: Optional[Dict[str, List[str]]] = None,
        room_departments: Optional[Dict[str, str]] = None,
        structural_grid: Tuple[float, float] = (8.0, 8.0),
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all metrics with optional weighting.

        Args:
            adjacency_preferences: Dict of preferred adjacencies
            room_departments: Dict mapping room_id to department
            structural_grid: Structural grid spacing
            weights: Dict of metric weights

        Returns:
            Dict: All metrics and weighted score
        """
        # Default weights
        default_weights = {
            "space_utilization": 1.0,
            "circulation_efficiency": 1.5,
            "adjacency_satisfaction": 2.0,
            "natural_light_access": 1.5,
            "structural_alignment": 1.0,
            "department_clustering": 1.0,
            "vertical_stacking": 0.5,
        }

        # Use provided weights or defaults
        metric_weights = weights or default_weights

        # Calculate metrics
        metrics = {
            "space_utilization": self.space_utilization(),
            "floor_area_ratio": self.floor_area_ratio(),
            "circulation_efficiency": self.circulation_efficiency(),
            "natural_light_access": self.natural_light_access(),
            "structural_alignment": self.structural_alignment(structural_grid),
            "vertical_stacking": self.vertical_stacking(),
            "egress_capacity": calculate_egress_capacity(
                self.layout, self.building_config
            ),
        }

        # Add metrics that require additional inputs
        if adjacency_preferences:
            metrics["adjacency_satisfaction"] = self.adjacency_satisfaction(
                adjacency_preferences
            )

        if room_departments:
            metrics["department_clustering"] = self.department_clustering(
                room_departments
            )

        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            # Skip circulation efficiency dict
            if metric == "circulation_efficiency":
                continue

            # Skip metrics not in weights
            if metric not in metric_weights:
                continue

            weight = metric_weights[metric]
            weighted_sum += value * weight
            total_weight += weight

        # Add circulation metrics with appropriate weights
        if (
            "circulation_efficiency" in metrics
            and "circulation_efficiency" in metric_weights
        ):
            circ_metrics = metrics["circulation_efficiency"]
            circ_weight = metric_weights["circulation_efficiency"]

            # Normalize circulation ratio (lower is better)
            # Typical values: 0.15-0.25 is good, >0.3 is inefficient
            circ_ratio_score = max(0, 1 - (circ_metrics["circulation_ratio"] / 0.3))

            # Normalize corridor length (lower is better)
            # Scale based on building size - here using rough heuristic
            building_size = math.sqrt(self.layout.width**2 + self.layout.length**2)
            max_corridor = building_size * 0.2  # 20% of building diagonal
            corridor_score = max(
                0, 1 - (circ_metrics["avg_corridor_length"] / max_corridor)
            )

            # Add to weighted sum
            weighted_sum += circ_ratio_score * circ_weight * 0.5
            weighted_sum += corridor_score * circ_weight * 0.5

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
        floor = int(z / building_config["floor_height"])

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
    exterior_area = 0.0

    # Calculate the surfaces of the building that are exterior
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
    for surface in building_surfaces:
        if surface[2] == 0:  # Bottom surface
            area = surface[3] * surface[4]
        elif surface[1] == 0 or surface[1] == layout.length:  # Front or back
            area = surface[3] * surface[4]
        else:  # Left or right
            area = surface[3] * surface[4]
        exterior_area += area

    # Calculate total volume
    total_volume = layout.width * layout.length * layout.height

    # Calculate surface-to-volume ratio
    surface_to_volume = exterior_area / total_volume if total_volume > 0 else 0

    # Calculate window-to-wall ratio (estimating windows on exterior rooms)
    exterior_rooms = layout.get_exterior_rooms()

    # Estimate windows as 40% of exterior wall area for exterior rooms
    window_area = 0.0
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
    # Group rooms by type
    rooms_by_type = {}

    for room_id, room_data in layout.rooms.items():
        room_type = room_data["type"]
        if room_type not in rooms_by_type:
            rooms_by_type[room_type] = []
        rooms_by_type[room_type].append(room_data)

    # Calculate metrics
    total_area_by_type = {}
    compliance_by_type = {}
    actual_vs_required = {}

    # Calculate total area for each room type
    for room_type, rooms in rooms_by_type.items():
        total_area = sum(
            room["dimensions"][0] * room["dimensions"][1] for room in rooms
        )
        total_area_by_type[room_type] = total_area

    # Calculate compliance for each room type
    for room_type, required in program_requirements.items():
        if isinstance(required, dict) and "area" in required:
            required_area = required["area"]
        else:
            required_area = required  # If just a number is provided

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
