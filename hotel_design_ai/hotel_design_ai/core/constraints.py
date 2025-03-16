"""
Constraint system for Hotel Design AI.
This file defines constraints that can be applied to hotel layouts.
"""

from typing import List, Dict, Tuple, Set, Any, Optional, Callable
import numpy as np

from hotel_design_ai.core.spatial_grid import SpatialGrid
from hotel_design_ai.models.room import Room


class Constraint:
    """Base class for all constraints"""

    def __init__(
        self, weight: float = 1.0, name: Optional[str] = None, is_hard: bool = False
    ):
        """
        Initialize a constraint.

        Args:
            weight: Importance weight of this constraint (higher = more important)
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint (must be satisfied)
        """
        self.weight = weight
        self.name = name or self.__class__.__name__
        self.is_hard = is_hard

    def evaluate(self, layout: SpatialGrid) -> float:
        """
        Evaluate how well the layout satisfies this constraint.

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            float: Score between 0.0 (not satisfied) and 1.0 (fully satisfied)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def is_satisfied(self, layout: SpatialGrid) -> bool:
        """
        Check if the constraint is satisfied (used for hard constraints).

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        return self.evaluate(layout) >= 0.99  # Allow for small numerical errors

    def __str__(self) -> str:
        return (
            f"{self.name} (weight={self.weight}, {'hard' if self.is_hard else 'soft'})"
        )


class AdjacencyConstraint(Constraint):
    """Constraint that requires two room types to be adjacent"""

    def __init__(
        self,
        room_type1: str,
        room_type2: str,
        weight: float = 1.0,
        name: Optional[str] = None,
        is_hard: bool = False,
    ):
        """
        Initialize an adjacency constraint.

        Args:
            room_type1: First room type
            room_type2: Second room type
            weight: Importance weight
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint
        """
        super().__init__(
            weight, name or f"Adjacency({room_type1},{room_type2})", is_hard
        )
        self.room_type1 = room_type1
        self.room_type2 = room_type2

    def evaluate(self, layout: SpatialGrid) -> float:
        """Evaluate how well the adjacency constraint is satisfied"""
        # Find all rooms of each type
        rooms_of_type1 = []
        rooms_of_type2 = []

        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]
            if room_type == self.room_type1:
                rooms_of_type1.append(room_id)
            elif room_type == self.room_type2:
                rooms_of_type2.append(room_id)

        # If either type is missing, constraint can't be satisfied
        if not rooms_of_type1 or not rooms_of_type2:
            return 0.0

        # Check if any rooms of type1 are adjacent to any rooms of type2
        for room1_id in rooms_of_type1:
            neighbors = layout.get_room_neighbors(room1_id)
            for room2_id in rooms_of_type2:
                if room2_id in neighbors:
                    return 1.0  # Found at least one adjacency

        # No adjacencies found
        return 0.0


class SeparationConstraint(Constraint):
    """Constraint that requires two room types to be separated (not adjacent)"""

    def __init__(
        self,
        room_type1: str,
        room_type2: str,
        weight: float = 1.0,
        name: Optional[str] = None,
        is_hard: bool = False,
    ):
        """
        Initialize a separation constraint.

        Args:
            room_type1: First room type
            room_type2: Second room type
            weight: Importance weight
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint
        """
        super().__init__(
            weight, name or f"Separation({room_type1},{room_type2})", is_hard
        )
        self.room_type1 = room_type1
        self.room_type2 = room_type2

    def evaluate(self, layout: SpatialGrid) -> float:
        """Evaluate how well the separation constraint is satisfied"""
        # If either room type doesn't exist, constraint is satisfied
        room_type1_exists = False
        room_type2_exists = False

        for _, room_data in layout.rooms.items():
            room_type = room_data["type"]
            if room_type == self.room_type1:
                room_type1_exists = True
            elif room_type == self.room_type2:
                room_type2_exists = True

            if room_type1_exists and room_type2_exists:
                break

        if not room_type1_exists or not room_type2_exists:
            return 1.0

        # Find all rooms of each type
        rooms_of_type1 = []
        rooms_of_type2 = []

        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]
            if room_type == self.room_type1:
                rooms_of_type1.append(room_id)
            elif room_type == self.room_type2:
                rooms_of_type2.append(room_id)

        # Check if any rooms of type1 are adjacent to any rooms of type2
        for room1_id in rooms_of_type1:
            neighbors = layout.get_room_neighbors(room1_id)
            for room2_id in rooms_of_type2:
                if room2_id in neighbors:
                    return 0.0  # Found an adjacency, constraint violated

        # No adjacencies found, constraint satisfied
        return 1.0


class ExteriorAccessConstraint(Constraint):
    """Constraint that requires a room type to have exterior access"""

    def __init__(
        self,
        room_type: str,
        weight: float = 1.0,
        name: Optional[str] = None,
        is_hard: bool = False,
    ):
        """
        Initialize an exterior access constraint.

        Args:
            room_type: Room type that requires exterior access
            weight: Importance weight
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint
        """
        super().__init__(weight, name or f"ExteriorAccess({room_type})", is_hard)
        self.room_type = room_type

    def evaluate(self, layout: SpatialGrid) -> float:
        """Evaluate how well the exterior access constraint is satisfied"""
        # Get all exterior rooms
        exterior_rooms = set(layout.get_exterior_rooms())

        # Find all rooms of the specified type
        rooms_of_type = []

        for room_id, room_data in layout.rooms.items():
            if room_data["type"] == self.room_type:
                rooms_of_type.append(room_id)

        # If no rooms of this type, constraint is satisfied by default
        if not rooms_of_type:
            return 1.0

        # Count how many have exterior access
        rooms_with_access = sum(
            1 for room_id in rooms_of_type if room_id in exterior_rooms
        )

        # Return ratio of rooms with access
        return rooms_with_access / len(rooms_of_type)


class FloorConstraint(Constraint):
    """Constraint that requires a room type to be on a specific floor"""

    def __init__(
        self,
        room_type: str,
        floor: int,
        weight: float = 1.0,
        name: Optional[str] = None,
        is_hard: bool = False,
    ):
        """
        Initialize a floor constraint.

        Args:
            room_type: Room type to constrain
            floor: Required floor (0 = ground floor, -1 = basement)
            weight: Importance weight
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint
        """
        super().__init__(weight, name or f"Floor({room_type},level={floor})", is_hard)
        self.room_type = room_type
        self.floor = floor

    def evaluate(self, layout: SpatialGrid) -> float:
        """Evaluate how well the floor constraint is satisfied"""
        # Find all rooms of the specified type
        rooms_of_type = []

        for room_id, room_data in layout.rooms.items():
            if room_data["type"] == self.room_type:
                rooms_of_type.append((room_id, room_data))

        # If no rooms of this type, constraint is satisfied by default
        if not rooms_of_type:
            return 1.0

        # Count how many are on the correct floor
        # Get floor height from building configuration
        correct_floor_count = 0

        for _, room_data in rooms_of_type:
            # Get floor from z coordinate based on building config
            # This should use the building_config floor_height
            # For now, use the default floor height
            floor_height = (
                5.0  # This will be replaced with building_config["floor_height"]
            )
            _, _, z = room_data["position"]
            floor = int(z / floor_height)

            if floor == self.floor:
                correct_floor_count += 1

        # Return ratio of rooms on correct floor
        return correct_floor_count / len(rooms_of_type)


class VerticalAlignmentConstraint(Constraint):
    """
    Hard constraint that requires vertical circulation elements to be aligned across floors.
    """

    def __init__(self, weight: float = 3.0, name: Optional[str] = None):
        """Initialize with higher default weight and as a hard constraint"""
        super().__init__(weight, name or "VerticalAlignment", is_hard=True)

    def evaluate(self, layout: SpatialGrid) -> float:
        """
        Evaluate if vertical circulation elements are aligned across floors.

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            float: 1.0 if all vertical circulation is aligned, 0.0 otherwise
        """
        # Group vertical circulation elements by their (x,y) position
        position_groups = {}
        floor_count = {}

        # This should be using building_config["floor_height"]
        floor_height = 5.0

        for room_id, room_data in layout.rooms.items():
            if room_data["type"] == "vertical_circulation":
                x, y, z = room_data["position"]
                # Round to avoid floating-point comparison issues
                key = (round(x, 1), round(y, 1))

                if key not in position_groups:
                    position_groups[key] = []
                    floor_count[key] = set()

                position_groups[key].append(room_id)
                # Determine floor from z-coordinate
                floor = int(z / floor_height)
                floor_count[key].add(floor)

        # Check if we even have vertical circulation
        if not position_groups:
            return 0.0  # No vertical circulation found

        # Count elements that are properly aligned
        # This should use building_config["min_floor"] and building_config["max_floor"]
        floor_range = list(range(-1, 4))  # Default from -1 to 3
        total_positions = len(position_groups)
        aligned_positions = 0

        for key, floors in floor_count.items():
            # Vertical circulation should exist on all floors
            if len(floors) == len(floor_range):
                aligned_positions += 1

        # Return proportion of aligned positions
        return aligned_positions / total_positions if total_positions > 0 else 0.0


class ConstraintSystem:
    """System for managing and evaluating multiple constraints"""

    def __init__(self, building_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the constraint system.

        Args:
            building_config: Building configuration parameters
        """
        self.constraints: List[Constraint] = []
        self.building_config = building_config or {
            "floor_height": 5.0,
            "min_floor": -1,
            "max_floor": 3,
        }

    @property
    def hard_constraints(self) -> List[Constraint]:
        """Get list of hard constraints"""
        return [c for c in self.constraints if c.is_hard]

    @property
    def soft_constraints(self) -> List[Constraint]:
        """Get list of soft constraints"""
        return [c for c in self.constraints if not c.is_hard]

    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the system"""
        self.constraints.append(constraint)

    def add_constraints(self, constraints: List[Constraint]):
        """Add multiple constraints to the system"""
        self.constraints.extend(constraints)

    def check_hard_constraints(self, layout: SpatialGrid) -> bool:
        """
        Check if all hard constraints are satisfied.

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            bool: True if all hard constraints are satisfied
        """
        for constraint in self.hard_constraints:
            if not constraint.is_satisfied(layout):
                return False
        return True

    def evaluate(self, layout: SpatialGrid) -> Dict[str, float]:
        """
        Evaluate all constraints for a layout.

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            Dict[str, float]: Scores for each constraint by name
        """
        results = {}

        for constraint in self.constraints:
            score = constraint.evaluate(layout)
            results[constraint.name] = score

        return results

    def evaluate_weighted(self, layout: SpatialGrid) -> float:
        """
        Calculate a weighted average of all soft constraint scores.

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            float: Weighted average score (0.0 to 1.0)
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for constraint in self.soft_constraints:
            score = constraint.evaluate(layout)
            weighted_sum += score * constraint.weight
            total_weight += constraint.weight

        if total_weight == 0.0:
            return 1.0  # No constraints

        return weighted_sum / total_weight

    def detailed_evaluation(self, layout: SpatialGrid) -> Dict[str, Any]:
        """
        Provide a detailed evaluation with individual and overall scores.

        Args:
            layout: The spatial grid layout to evaluate

        Returns:
            Dict: Detailed evaluation results
        """
        individual_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # Process hard constraints
        hard_satisfied = self.check_hard_constraints(layout)

        # Process soft constraints
        for constraint in self.soft_constraints:
            score = constraint.evaluate(layout)
            individual_scores[constraint.name] = {
                "score": score,
                "weight": constraint.weight,
                "weighted_score": score * constraint.weight,
                "is_hard": False,
            }
            weighted_sum += score * constraint.weight
            total_weight += constraint.weight

        # Add hard constraint results
        for constraint in self.hard_constraints:
            score = constraint.evaluate(layout)
            satisfied = constraint.is_satisfied(layout)
            individual_scores[constraint.name] = {
                "score": score,
                "weight": constraint.weight,
                "satisfied": satisfied,
                "is_hard": True,
            }

        overall_score = weighted_sum / total_weight if total_weight > 0 else 1.0

        return {
            "overall_score": overall_score,
            "hard_constraints_satisfied": hard_satisfied,
            "total_weight": total_weight,
            "constraints": individual_scores,
        }


def create_hard_constraints(building_config: Dict[str, Any]) -> List[Constraint]:
    """
    Create standard hard constraints for hotel layouts.

    Args:
        building_config: Building configuration parameters

    Returns:
        List[Constraint]: List of hard constraints
    """
    constraints = []

    # Entrance must be on ground floor
    constraints.append(
        FloorConstraint(
            "entrance", 0, weight=3.0, name="EntranceOnGround", is_hard=True
        )
    )

    # Entrance must have exterior access
    constraints.append(
        ExteriorAccessConstraint(
            "entrance", weight=3.0, name="EntranceExteriorAccess", is_hard=True
        )
    )

    # Vertical circulation must be aligned
    constraints.append(VerticalAlignmentConstraint(weight=3.0))

    return constraints


def create_default_constraints(
    building_config: Dict[str, Any],
    structural_grid: Tuple[float, float] = (8.0, 8.0),
    include_hard_constraints: bool = True,
) -> ConstraintSystem:
    """
    Create a default set of constraints for hotel layouts.

    Args:
        building_config: Building configuration parameters
        structural_grid: (x_spacing, y_spacing) of the structural grid
        include_hard_constraints: Whether to include hard constraints

    Returns:
        ConstraintSystem: Initialized with common hotel design constraints
    """
    constraint_system = ConstraintSystem(building_config)

    # Add hard constraints if requested
    if include_hard_constraints:
        hard_constraints = create_hard_constraints(building_config)
        constraint_system.add_constraints(hard_constraints)

    # Adjacency constraints (rooms that should be adjacent)
    adjacency_constraints = [
        AdjacencyConstraint("entrance", "lobby", weight=2.0),
        AdjacencyConstraint("lobby", "vertical_circulation", weight=1.5),
        AdjacencyConstraint("restaurant", "lobby", weight=1.0),
        AdjacencyConstraint("kitchen", "restaurant", weight=2.0),
        AdjacencyConstraint("meeting_room", "pre_function", weight=1.5),
        AdjacencyConstraint("ballroom", "pre_function", weight=2.0),
    ]

    # Separation constraints (rooms that should not be adjacent)
    separation_constraints = [
        SeparationConstraint("kitchen", "guest_room", weight=1.5),
        SeparationConstraint("back_of_house", "ballroom", weight=1.0),
        SeparationConstraint("mechanical", "meeting_room", weight=1.5),
    ]

    # Floor constraints
    floor_constraints = [
        # Most floor constraints are soft (preferences rather than requirements)
        FloorConstraint("lobby", 0, weight=2.0),
        FloorConstraint("parking", -1, weight=2.0),
    ]

    # Other constraints
    other_constraints = [
        ExteriorAccessConstraint("guest_room", weight=1.5),
        NaturalLightConstraint(
            ["guest_room", "restaurant", "meeting_room", "office"], weight=1.5
        ),
        ProximityConstraint(
            "vertical_circulation", "guest_room", max_distance=30.0, weight=1.0
        ),
    ]

    # Add all soft constraints
    constraint_system.add_constraints(adjacency_constraints)
    constraint_system.add_constraints(separation_constraints)
    constraint_system.add_constraints(floor_constraints)
    constraint_system.add_constraints(other_constraints)

    return constraint_system


class ProximityConstraint(Constraint):
    """Constraint that rewards rooms being close to each other"""

    def __init__(
        self,
        room_type1: str,
        room_type2: str,
        max_distance: float = 20.0,
        weight: float = 1.0,
        name: Optional[str] = None,
        is_hard: bool = False,
    ):
        """
        Initialize a proximity constraint.

        Args:
            room_type1: First room type
            room_type2: Second room type
            max_distance: Maximum distance for full satisfaction (meters)
            weight: Importance weight
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint
        """
        super().__init__(
            weight, name or f"Proximity({room_type1},{room_type2})", is_hard
        )
        self.room_type1 = room_type1
        self.room_type2 = room_type2
        self.max_distance = max_distance

    def evaluate(self, layout: SpatialGrid) -> float:
        """Evaluate how well the proximity constraint is satisfied"""
        # Find all rooms of each type
        rooms_of_type1 = []
        rooms_of_type2 = []

        for room_id, room_data in layout.rooms.items():
            room_type = room_data["type"]
            if room_type == self.room_type1:
                rooms_of_type1.append(room_data)
            elif room_type == self.room_type2:
                rooms_of_type2.append(room_data)

        # If either type is missing, constraint can't be satisfied
        if not rooms_of_type1 or not rooms_of_type2:
            return 0.0

        # Calculate minimum distance between any room of type1 and any room of type2
        min_distance = float("inf")

        for room1 in rooms_of_type1:
            pos1 = room1["position"]
            center1 = (
                pos1[0] + room1["dimensions"][0] / 2,
                pos1[1] + room1["dimensions"][1] / 2,
                pos1[2] + room1["dimensions"][2] / 2,
            )

            for room2 in rooms_of_type2:
                pos2 = room2["position"]
                center2 = (
                    pos2[0] + room2["dimensions"][0] / 2,
                    pos2[1] + room2["dimensions"][1] / 2,
                    pos2[2] + room2["dimensions"][2] / 2,
                )

                # Calculate Euclidean distance between centers
                distance = np.sqrt(
                    (center1[0] - center2[0]) ** 2
                    + (center1[1] - center2[1]) ** 2
                    + (center1[2] - center2[2]) ** 2
                )

                min_distance = min(min_distance, distance)

        # Normalize distance: 0 = max_distance or greater, 1 = touching
        normalized_score = max(0.0, 1.0 - (min_distance / self.max_distance))
        return normalized_score


class NaturalLightConstraint(Constraint):
    """Constraint that requires certain room types to have natural light (exterior walls)"""

    def __init__(
        self,
        room_types: List[str],
        weight: float = 1.0,
        name: Optional[str] = None,
        is_hard: bool = False,
    ):
        """
        Initialize a natural light constraint.

        Args:
            room_types: List of room types that require natural light
            weight: Importance weight
            name: Optional name for the constraint
            is_hard: Whether this is a hard constraint
        """
        super().__init__(weight, name or "NaturalLight", is_hard)
        self.room_types = room_types

    def evaluate(self, layout: SpatialGrid) -> float:
        """Evaluate how well the natural light constraint is satisfied"""
        # Get all exterior rooms
        exterior_rooms = set(layout.get_exterior_rooms())

        # Find rooms of specified types
        rooms_needing_light = []

        for room_id, room_data in layout.rooms.items():
            if room_data["type"] in self.room_types:
                rooms_needing_light.append(room_id)

        # If no rooms need light, constraint is satisfied
        if not rooms_needing_light:
            return 1.0

        # Count how many have exterior access
        rooms_with_light = sum(
            1 for room_id in rooms_needing_light if room_id in exterior_rooms
        )

        # Return ratio
        return rooms_with_light / len(rooms_needing_light)


def from_adjacency_matrix(
    room_types: List[str], adjacency_matrix: List[List[int]], base_weight: float = 1.0
) -> List[Constraint]:
    """
    Create constraints from an adjacency matrix.

    Args:
        room_types: List of room types (in order matching matrix)
        adjacency_matrix: Matrix where:
            2 = must be adjacent
            1 = preferred adjacent
            0 = no preference
            -1 = preferred separate
            -2 = must be separate
        base_weight: Base weight for constraints

    Returns:
        List[Constraint]: Generated constraints
    """
    constraints = []

    for i in range(len(room_types)):
        for j in range(
            i + 1, len(room_types)
        ):  # Only upper triangle to avoid duplicates
            value = adjacency_matrix[i][j]

            if value == 2:
                # Must be adjacent
                constraints.append(
                    AdjacencyConstraint(
                        room_types[i],
                        room_types[j],
                        weight=base_weight * 2.0,
                        is_hard=True,
                    )
                )
            elif value == 1:
                # Preferred adjacent
                constraints.append(
                    AdjacencyConstraint(
                        room_types[i], room_types[j], weight=base_weight
                    )
                )
            elif value == -1:
                # Preferred separate
                constraints.append(
                    SeparationConstraint(
                        room_types[i], room_types[j], weight=base_weight
                    )
                )
            elif value == -2:
                # Must be separate
                constraints.append(
                    SeparationConstraint(
                        room_types[i],
                        room_types[j],
                        weight=base_weight * 2.0,
                        is_hard=True,
                    )
                )

    return constraints
