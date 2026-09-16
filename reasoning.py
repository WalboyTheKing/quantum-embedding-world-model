"""
Quantum-Embedding World Model
V1.1 - Spatial Reasoning

Purpose:
    Reason about navigability and spatial state using the
    existing 3D sparse world model.

Architecture:

    Sensors
       |
       v
    Sparse 3D World
       |
       v
    Spatial Queries
       |
       v
    V1.1 Spatial Reasoning
       |
       +-- PASSABLE
       +-- BLOCKED
       +-- UNCERTAIN

Important principles:

    UNKNOWN != FREE
    UNKNOWN != OBSTACLE

    Reasoning must not invent information.

    This module does NOT modify the world.
"""

from world_model import SparseWorldModel


class SpatialReasoner:
    """
    First reasoning layer for the Quantum-Embedding World Model.

    The reasoner evaluates a 3D path using observed matter,
    confidence and UNKNOWN regions.
    """

    def __init__(self, world):
        self.world = world

    # =========================================================
    # BASIC HELPERS
    # =========================================================

    def _inside_bounds(self, coordinate):
        x, y, z = coordinate
        sx, sy, sz = self.world.size

        return (
            0 <= x < sx
            and 0 <= y < sy
            and 0 <= z < sz
        )

    def _get_matter(self, coordinate):
        x, y, z = coordinate

        result = self.world.query(
            x,
            y,
            z,
            "matter",
        )

        if not result:
            return None

        if result.get("state") == "UNKNOWN":
            return None

        return result

    # =========================================================
    # VOXEL REASONING
    # =========================================================

    def classify_voxel(
        self,
        coordinate,
        min_confidence=0.5,
    ):
        """
        Classify one voxel as:

            FREE
            BLOCKED
            UNCERTAIN
            OUT_OF_BOUNDS
        """

        if not self._inside_bounds(coordinate):
            return {
                "coordinate": coordinate,
                "status": "OUT_OF_BOUNDS",
                "reason": "Coordinate is outside world bounds.",
            }

        matter = self._get_matter(coordinate)

        # UNKNOWN is never considered free.
        if matter is None:
            return {
                "coordinate": coordinate,
                "status": "UNCERTAIN",
                "reason": "Matter is UNKNOWN.",
                "confidence": 0.0,
                "uncertainty": 1.0,
            }

        value = matter.get("value")
        confidence = matter.get("confidence", 0.0)

        if confidence < min_confidence:
            return {
                "coordinate": coordinate,
                "status": "UNCERTAIN",
                "reason": "Matter confidence is too low.",
                "value": value,
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
            }

        if value is not None and value > 0:
            return {
                "coordinate": coordinate,
                "status": "BLOCKED",
                "reason": "Observed matter occupies this voxel.",
                "value": value,
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
            }

        return {
            "coordinate": coordinate,
            "status": "FREE",
            "reason": "Observed absence of matter.",
            "value": value,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
        }

    # =========================================================
    # PATH REASONING
    # =========================================================

    def reason_path(
        self,
        start,
        direction,
        distance,
        min_confidence=0.5,
    ):
        """
        Evaluate a straight 3D path.

        Returns:

            PASSABLE
            BLOCKED
            UNCERTAIN

        Rules:

            BLOCKED:
                At least one observed voxel contains matter.

            UNCERTAIN:
                No known obstacle exists, but at least one
                voxel is UNKNOWN or has insufficient confidence.

            PASSABLE:
                Every voxel along the path is observed free.
        """

        if distance < 1:
            raise ValueError(
                "distance must be >= 1"
            )

        if not self._inside_bounds(start):
            raise ValueError(
                "Start coordinate is outside world bounds."
            )

        dx, dy, dz = direction

        if (dx, dy, dz) == (0, 0, 0):
            raise ValueError(
                "Direction cannot be zero."
            )

        checked_voxels = []
        uncertain_voxels = []
        blocked_voxels = []

        for step in range(1, distance + 1):

            coordinate = (
                start[0] + dx * step,
                start[1] + dy * step,
                start[2] + dz * step,
            )

            classification = self.classify_voxel(
                coordinate,
                min_confidence=min_confidence,
            )

            checked_voxels.append(
                classification
            )

            if classification["status"] == "OUT_OF_BOUNDS":
                return {
                    "status": "BLOCKED",
                    "reason": "Path leaves the world bounds.",
                    "checked_voxels": checked_voxels,
                    "blocked_voxels": blocked_voxels,
                    "uncertain_voxels": uncertain_voxels,
                }

            if classification["status"] == "BLOCKED":
                blocked_voxels.append(
                    classification
                )

            elif classification["status"] == "UNCERTAIN":
                uncertain_voxels.append(
                    classification
                )

        if blocked_voxels:
            status = "BLOCKED"
            reason = (
                "Observed matter blocks the path."
            )

        elif uncertain_voxels:
            status = "UNCERTAIN"
            reason = (
                "The path contains UNKNOWN or "
                "low-confidence space."
            )

        else:
            status = "PASSABLE"
            reason = (
                "All path voxels are observed free "
                "with sufficient confidence."
            )

        return {
            "status": status,
            "reason": reason,
            "start": start,
            "direction": direction,
            "distance": distance,
            "checked_voxels": checked_voxels,
            "blocked_voxels": blocked_voxels,
            "uncertain_voxels": uncertain_voxels,
        }

    # =========================================================
    # UNKNOWN-AWARE REASONING
    # =========================================================

    def can_pass(
        self,
        start,
        direction,
        distance,
        min_confidence=0.5,
    ):
        """
        High-level navigation decision.

        Returns a structured decision instead of
        assuming UNKNOWN means FREE.
        """

        result = self.reason_path(
            start=start,
            direction=direction,
            distance=distance,
            min_confidence=min_confidence,
        )

        if result["status"] == "PASSABLE":
            decision = "YES"

        elif result["status"] == "BLOCKED":
            decision = "NO"

        else:
            decision = "UNKNOWN"

        return {
            "decision": decision,
            "status": result["status"],
            "reason": result["reason"],
            "path": result,
        }

    # =========================================================
    # WORLD MUTATION CHECK
    # =========================================================

    def reasoning_does_not_modify_world(
        self,
        reasoning_function,
        *args,
        **kwargs,
    ):
        """
        Verify that reasoning only reads the world.
        """

        before = len(self.world.voxels)

        result = reasoning_function(
            *args,
            **kwargs,
        )

        after = len(self.world.voxels)

        return {
            "result": result,
            "world_modified": before != after,
            "voxels_before": before,
            "voxels_after": after,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self):
        return {
            "world_size": self.world.size,
            "observed_voxels": len(
                self.world.voxels
            ),
            "reasoning_type": (
                "unknown-aware 3D spatial reasoning"
            ),
            "world_mutation": False,
            "unknown_is_free": False,
            "unknown_is_obstacle": False,
        }


# =============================================================
# DEMO WORLD
# =============================================================

def create_demo_world():

    world = SparseWorldModel(
        size=(8, 8, 8)
    )

    # ---------------------------------------------------------
    # Observed free path
    #
    # x = 1, 2, 3
    # These voxels explicitly contain zero matter.
    # ---------------------------------------------------------

    world.update_sensor(
        1, 1, 1,
        "lidar",
        0.0,
        0.95,
    )

    world.update_sensor(
        2, 1, 1,
        "lidar",
        0.0,
        0.95,
    )

    world.update_sensor(
        3, 1, 1,
        "lidar",
        0.0,
        0.95,
    )

    # ---------------------------------------------------------
    # Observed obstacle
    # ---------------------------------------------------------

    world.update_sensor(
        4, 1, 1,
        "lidar",
        1.0,
        0.95,
    )

    # ---------------------------------------------------------
    # Another obstacle in 3D
    # ---------------------------------------------------------

    world.update_sensor(
        3, 2, 2,
        "lidar",
        1.0,
        0.90,
    )

    return world


# =============================================================
# TESTS
# =============================================================

def run_tests():

    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V1.1")
    print("SPATIAL REASONING")
    print("=" * 60)

    print()
    print("Creating reasoning test world...")

    world = create_demo_world()

    reasoner = SpatialReasoner(world)

    print()
    print("World size:", world.size)
    print(
        "Observed voxels:",
        len(world.voxels),
    )

    # =========================================================
    # TEST 1
    # =========================================================

    summary = reasoner.summary()

    assert (
        summary["unknown_is_free"]
        is False
    )

    assert (
        summary["unknown_is_obstacle"]
        is False
    )

    print()
    print(
        "Reasoning engine initialization test: PASS"
    )

    # =========================================================
    # TEST 2
    # =========================================================

    free = reasoner.classify_voxel(
        (2, 1, 1)
    )

    assert free["status"] == "FREE"

    print(
        "Observed free voxel reasoning test: PASS"
    )

    # =========================================================
    # TEST 3
    # =========================================================

    blocked = reasoner.classify_voxel(
        (4, 1, 1)
    )

    assert blocked["status"] == "BLOCKED"

    print(
        "Observed obstacle reasoning test: PASS"
    )

    # =========================================================
    # TEST 4
    # =========================================================

    unknown = reasoner.classify_voxel(
        (7, 7, 7)
    )

    assert unknown["status"] == "UNCERTAIN"

    print(
        "UNKNOWN voxel reasoning test: PASS"
    )

    # =========================================================
    # TEST 5
    # =========================================================

    passable = reasoner.reason_path(
        start=(0, 1, 1),
        direction=(1, 0, 0),
        distance=3,
    )

    assert passable["status"] == "PASSABLE"

    print(
        "PASSABLE path reasoning test: PASS"
    )

    # =========================================================
    # TEST 6
    # =========================================================

    blocked_path = reasoner.reason_path(
        start=(0, 1, 1),
        direction=(1, 0, 0),
        distance=4,
    )

    assert blocked_path["status"] == "BLOCKED"

    print(
        "BLOCKED path reasoning test: PASS"
    )

    # =========================================================
    # TEST 7
    # =========================================================

    unknown_path = reasoner.reason_path(
        start=(5, 5, 5),
        direction=(1, 0, 0),
        distance=2,
    )

    assert unknown_path["status"] == "UNCERTAIN"

    print(
        "UNKNOWN path reasoning test: PASS"
    )

    # =========================================================
    # TEST 8
    # =========================================================

    yes_decision = reasoner.can_pass(
        start=(0, 1, 1),
        direction=(1, 0, 0),
        distance=3,
    )

    assert yes_decision["decision"] == "YES"

    print(
        "YES navigation decision test: PASS"
    )

    # =========================================================
    # TEST 9
    # =========================================================

    no_decision = reasoner.can_pass(
        start=(0, 1, 1),
        direction=(1, 0, 0),
        distance=4,
    )

    assert no_decision["decision"] == "NO"

    print(
        "NO navigation decision test: PASS"
    )

    # =========================================================
    # TEST 10
    # =========================================================

    unknown_decision = reasoner.can_pass(
        start=(5, 5, 5),
        direction=(1, 0, 0),
        distance=2,
    )

    assert (
        unknown_decision["decision"]
        == "UNKNOWN"
    )

    print(
        "UNKNOWN navigation decision test: PASS"
    )

    # =========================================================
    # TEST 11
    # =========================================================

    diagonal_blocked = reasoner.reason_path(
        start=(3, 1, 1),
        direction=(0, 1, 1),
        distance=1,
    )

    assert (
        diagonal_blocked["status"]
        == "BLOCKED"
    )

    print(
        "3D diagonal obstacle reasoning test: PASS"
    )

    # =========================================================
    # TEST 12
    # =========================================================

    before = len(world.voxels)

    check = (
        reasoner.reasoning_does_not_modify_world(
            reasoner.can_pass,
            (0, 1, 1),
            (1, 0, 0),
            3,
        )
    )

    after = len(world.voxels)

    assert before == after
    assert (
        check["world_modified"]
        is False
    )

    print(
        "Reasoning immutability test: PASS"
    )

    # =========================================================
    # TEST 13
    # =========================================================

    out_of_bounds = reasoner.reason_path(
        start=(7, 7, 7),
        direction=(1, 0, 0),
        distance=2,
    )

    assert (
        out_of_bounds["status"]
        == "BLOCKED"
    )

    print(
        "World-boundary reasoning test: PASS"
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("-" * 60)
    print("SPATIAL REASONING SUMMARY")
    print("-" * 60)

    print(
        "Observed voxels :",
        len(world.voxels),
    )

    print(
        "Free path       : PASSABLE"
    )

    print(
        "Obstacle path   : BLOCKED"
    )

    print(
        "Unknown path    : UNCERTAIN"
    )

    print()
    print("=" * 60)
    print("V1.1 TEST COMPLETE")
    print("=" * 60)

    print()
    print("All V1.1 tests passed.")

    print()
    print("Architecture validated:")
    print(
        "  [OK] 3D voxel classification"
    )
    print(
        "  [OK] Observed free-space reasoning"
    )
    print(
        "  [OK] Observed obstacle reasoning"
    )
    print(
        "  [OK] UNKNOWN-aware reasoning"
    )
    print(
        "  [OK] PASSABLE path reasoning"
    )
    print(
        "  [OK] BLOCKED path reasoning"
    )
    print(
        "  [OK] UNCERTAIN path reasoning"
    )
    print(
        "  [OK] 3D diagonal reasoning"
    )
    print(
        "  [OK] Navigation decisions"
    )
    print(
        "  [OK] World-boundary handling"
    )
    print(
        "  [OK] Reasoning does not mutate world"
    )
    print(
        "  [OK] UNKNOWN != FREE"
    )
    print(
        "  [OK] UNKNOWN != OBSTACLE"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_tests()