"""
Quantum-Embedding World Model
V0.8 - Spatial Queries

Builds spatial reasoning queries on top of the existing
SparseWorldModel without replacing the V0.7 foundation.

Features:
- Neighbor voxel queries
- Solid matter queries
- Hot zone queries
- Motion zone queries
- UNKNOWN region detection
- Local uncertainty analysis
- Directional obstacle checks
- 3D spatial inspection
- Automated V0.8 tests
"""

from world_model import SparseWorldModel


class SpatialQueries:
    """
    Spatial query layer built on top of SparseWorldModel.

    The underlying world remains sparse:
    only observed voxels are stored.
    """

    def __init__(self, world):
        self.world = world

        if not hasattr(world, "size"):
            raise ValueError("World must expose a 'size' attribute.")

        self.size = world.size

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def _inside_bounds(self, x, y, z):
        """Check whether a voxel coordinate is inside the 3D world."""
        sx, sy, sz = self.size

        return (
            0 <= x < sx
            and 0 <= y < sy
            and 0 <= z < sz
        )

    def _get_embedding(self, x, y, z):
        """Return the embedding if the voxel exists, otherwise None."""
        if not self._inside_bounds(x, y, z):
            return None

        return self.world.get(x, y, z)

    def _get_property(self, x, y, z, property_name):
        """
        Safely query a property.

        Returns None when the voxel/property is unknown.
        """
        if not self._inside_bounds(x, y, z):
            return None

        result = self.world.query(
            x,
            y,
            z,
            property_name,
        )

        if not result:
            return None

        if result.get("state") == "UNKNOWN":
            return None

        return result

    # ============================================================
    # 1. NEIGHBOR QUERIES
    # ============================================================

    def neighbors(
        self,
        x,
        y,
        z,
        radius=1,
        include_unknown=True,
    ):
        """
        Return neighboring voxels around a coordinate.

        radius=1:
            3x3x3 neighborhood

        The center voxel itself is excluded.

        Returns a list of dictionaries.
        """

        if radius < 1:
            raise ValueError("radius must be >= 1")

        results = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):

                    # Skip center voxel
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    nx = x + dx
                    ny = y + dy
                    nz = z + dz

                    if not self._inside_bounds(nx, ny, nz):
                        continue

                    embedding = self._get_embedding(nx, ny, nz)

                    if embedding is None:
                        if include_unknown:
                            results.append({
                                "coordinate": (nx, ny, nz),
                                "observed": False,
                                "state": "UNKNOWN",
                            })

                        continue

                    results.append({
                        "coordinate": (nx, ny, nz),
                        "observed": True,
                        "state": "OBSERVED",
                        "known_properties": embedding.known_count(),
                        "average_uncertainty": (
                            embedding.average_uncertainty()
                        ),
                    })

        return results

    # ============================================================
    # 2. SOLID MATTER
    # ============================================================

    def solid_voxels(
        self,
        min_confidence=0.0,
    ):
        """
        Find observed voxels containing solid matter.

        matter > 0 means matter is present.
        """

        results = []

        for coordinate, embedding in self.world.voxels.items():

            x, y, z = coordinate

            result = self._get_property(
                x,
                y,
                z,
                "matter",
            )

            if result is None:
                continue

            value = result.get("value")
            confidence = result.get("confidence", 0.0)

            if value is None:
                continue

            if value > 0 and confidence >= min_confidence:
                results.append({
                    "coordinate": coordinate,
                    "value": value,
                    "confidence": confidence,
                    "uncertainty": result.get(
                        "uncertainty",
                        1.0 - confidence,
                    ),
                })

        return results

    # ============================================================
    # 3. HOT ZONES
    # ============================================================

    def hot_zones(
        self,
        threshold=30.0,
        min_confidence=0.0,
    ):
        """
        Find voxels whose observed temperature is above threshold.
        """

        results = []

        for coordinate, embedding in self.world.voxels.items():

            x, y, z = coordinate

            result = self._get_property(
                x,
                y,
                z,
                "temperature",
            )

            if result is None:
                continue

            temperature = result.get("value")
            confidence = result.get("confidence", 0.0)

            if temperature is None:
                continue

            if (
                temperature >= threshold
                and confidence >= min_confidence
            ):
                results.append({
                    "coordinate": coordinate,
                    "temperature": temperature,
                    "confidence": confidence,
                    "uncertainty": result.get(
                        "uncertainty",
                        1.0 - confidence,
                    ),
                })

        return results

    # ============================================================
    # 4. MOTION ZONES
    # ============================================================

    def motion_zones(
        self,
        threshold=0.0,
        min_confidence=0.0,
    ):
        """
        Find voxels containing observed motion.
        """

        results = []

        for coordinate, embedding in self.world.voxels.items():

            x, y, z = coordinate

            result = self._get_property(
                x,
                y,
                z,
                "motion",
            )

            if result is None:
                continue

            motion = result.get("value")
            confidence = result.get("confidence", 0.0)

            if motion is None:
                continue

            if (
                abs(motion) > threshold
                and confidence >= min_confidence
            ):
                results.append({
                    "coordinate": coordinate,
                    "motion": motion,
                    "confidence": confidence,
                    "uncertainty": result.get(
                        "uncertainty",
                        1.0 - confidence,
                    ),
                })

        return results

    # ============================================================
    # 5. UNKNOWN VOXELS
    # ============================================================

    def unknown_voxels(self):
        """
        Return virtual voxels that have never been observed.

        IMPORTANT:
        UNKNOWN is not represented by stored zero values.

        Because the world is sparse, an absent voxel is unknown.
        """

        sx, sy, sz = self.size

        results = []

        for x in range(sx):
            for y in range(sy):
                for z in range(sz):

                    if (x, y, z) not in self.world.voxels:
                        results.append(
                            (x, y, z)
                        )

        return results

    # ============================================================
    # 6. UNKNOWN REGION AROUND A POINT
    # ============================================================

    def unknown_neighbors(
        self,
        x,
        y,
        z,
        radius=1,
    ):
        """
        Return neighboring voxels that have never been observed.
        """

        neighbors = self.neighbors(
            x,
            y,
            z,
            radius=radius,
            include_unknown=True,
        )

        return [
            item["coordinate"]
            for item in neighbors
            if not item["observed"]
        ]

    # ============================================================
    # 7. LOCAL UNCERTAINTY
    # ============================================================

    def local_uncertainty(
        self,
        x,
        y,
        z,
        radius=1,
    ):
        """
        Calculate uncertainty around a spatial region.

        Returns:
        - observed voxel count
        - unknown voxel count
        - average uncertainty
        - known property count
        """

        neighbors = self.neighbors(
            x,
            y,
            z,
            radius=radius,
            include_unknown=True,
        )

        observed_count = 0
        unknown_count = 0
        uncertainties = []
        known_properties = 0

        for item in neighbors:

            if not item["observed"]:
                unknown_count += 1
                continue

            observed_count += 1

            uncertainty = item.get(
                "average_uncertainty"
            )

            if uncertainty is not None:
                uncertainties.append(
                    uncertainty
                )

            known_properties += item.get(
                "known_properties",
                0,
            )

        if uncertainties:
            average_uncertainty = (
                sum(uncertainties)
                / len(uncertainties)
            )
        else:
            average_uncertainty = 1.0

        total = observed_count + unknown_count

        if total > 0:
            unknown_ratio = (
                unknown_count / total
            )
        else:
            unknown_ratio = 1.0

        return {
            "center": (x, y, z),
            "radius": radius,
            "observed_voxels": observed_count,
            "unknown_voxels": unknown_count,
            "unknown_ratio": unknown_ratio,
            "average_uncertainty": average_uncertainty,
            "known_properties": known_properties,
        }

    # ============================================================
    # 8. PROPERTY AROUND A LOCATION
    # ============================================================

    def property_near(
        self,
        x,
        y,
        z,
        property_name,
        radius=1,
    ):
        """
        Find nearby voxels containing a specific property.
        """

        results = []

        neighbors = self.neighbors(
            x,
            y,
            z,
            radius=radius,
            include_unknown=False,
        )

        for item in neighbors:

            nx, ny, nz = item["coordinate"]

            result = self._get_property(
                nx,
                ny,
                nz,
                property_name,
            )

            if result is None:
                continue

            results.append({
                "coordinate": (nx, ny, nz),
                "value": result.get("value"),
                "state": result.get("state"),
                "confidence": result.get(
                    "confidence",
                    0.0,
                ),
                "uncertainty": result.get(
                    "uncertainty",
                    1.0,
                ),
            })

        return results

    # ============================================================
    # 9. OBSTACLE CHECK
    # ============================================================

    def is_obstacle(
        self,
        x,
        y,
        z,
        min_confidence=0.5,
    ):
        """
        Determine whether a voxel contains solid matter.

        Returns a structured result instead of a simple boolean
        so uncertainty is preserved.
        """

        if not self._inside_bounds(x, y, z):
            return {
                "coordinate": (x, y, z),
                "obstacle": False,
                "state": "OUT_OF_BOUNDS",
                "confidence": 1.0,
                "uncertainty": 0.0,
            }

        result = self._get_property(
            x,
            y,
            z,
            "matter",
        )

        if result is None:
            return {
                "coordinate": (x, y, z),
                "obstacle": None,
                "state": "UNKNOWN",
                "confidence": 0.0,
                "uncertainty": 1.0,
            }

        value = result.get("value")
        confidence = result.get(
            "confidence",
            0.0,
        )

        if value is None:
            return {
                "coordinate": (x, y, z),
                "obstacle": None,
                "state": "UNKNOWN",
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
            }

        obstacle = (
            value > 0
            and confidence >= min_confidence
        )

        return {
            "coordinate": (x, y, z),
            "obstacle": obstacle,
            "state": result.get(
                "state",
                "UNKNOWN",
            ),
            "confidence": confidence,
            "uncertainty": result.get(
                "uncertainty",
                1.0 - confidence,
            ),
        }

    # ============================================================
    # 10. DIRECTIONAL PATH CHECK
    # ============================================================

    def check_direction(
        self,
        start,
        direction,
        distance,
        min_confidence=0.5,
    ):
        """
        Check a straight 3D path.

        Example:

            start=(4,4,2)
            direction=(1,0,0)
            distance=3

        Checks:

            (5,4,2)
            (6,4,2)
            (7,4,2)

        direction values should normally be -1, 0 or 1.
        """

        if distance < 1:
            raise ValueError(
                "distance must be >= 1"
            )

        sx, sy, sz = start
        dx, dy, dz = direction

        results = []

        for step in range(1, distance + 1):

            x = sx + dx * step
            y = sy + dy * step
            z = sz + dz * step

            if not self._inside_bounds(
                x,
                y,
                z,
            ):
                results.append({
                    "step": step,
                    "coordinate": (x, y, z),
                    "status": "OUT_OF_BOUNDS",
                    "obstacle": False,
                    "confidence": 1.0,
                    "uncertainty": 0.0,
                })

                continue

            obstacle = self.is_obstacle(
                x,
                y,
                z,
                min_confidence=min_confidence,
            )

            if obstacle["state"] == "UNKNOWN":
                status = "UNKNOWN"
            elif obstacle["obstacle"]:
                status = "OBSTACLE"
            else:
                status = "FREE"

            results.append({
                "step": step,
                "coordinate": (x, y, z),
                "status": status,
                "obstacle": obstacle["obstacle"],
                "confidence": obstacle[
                    "confidence"
                ],
                "uncertainty": obstacle[
                    "uncertainty"
                ],
            })

        return results

    # ============================================================
    # 11. SPATIAL SUMMARY
    # ============================================================

    def summary(self):
        """
        Generate a global spatial summary.
        """

        solid = self.solid_voxels()
        hot = self.hot_zones()
        motion = self.motion_zones()
        unknown = self.unknown_voxels()

        return {
            "world_size": self.size,
            "solid_voxels": len(solid),
            "hot_zones": len(hot),
            "motion_zones": len(motion),
            "unknown_voxels": len(unknown),
            "observed_voxels": len(
                self.world.voxels
            ),
        }


# ================================================================
# DEMO WORLD
# ================================================================

def create_demo_world():
    """
    Create the same type of shared 3D world used during V0.7.
    """

    world = SparseWorldModel(
        size=(8, 8, 8)
    )

    # ------------------------------------------------------------
    # Central object
    # ------------------------------------------------------------

    world.update_sensor(
        4,
        4,
        2,
        "camera",
        0.8,
        0.95,
    )

    world.update_sensor(
        4,
        4,
        2,
        "radar",
        1.2,
        0.70,
    )

    world.update_sensor(
        4,
        4,
        2,
        "contact",
        1.0,
        0.95,
    )

    world.update_sensor(
        4,
        4,
        2,
        "lidar",
        1.0,
        0.98,
    )

    # ------------------------------------------------------------
    # Hot region
    # ------------------------------------------------------------

    world.update_sensor(
        6,
        4,
        2,
        "thermal",
        42.0,
        0.85,
    )

    # ------------------------------------------------------------
    # Second object
    # ------------------------------------------------------------

    world.update_sensor(
        2,
        5,
        1,
        "camera",
        0.45,
        0.90,
    )

    world.update_sensor(
        2,
        5,
        1,
        "lidar",
        1.0,
        0.92,
    )

    # ------------------------------------------------------------
    # Moving object
    # ------------------------------------------------------------

    world.update_sensor(
        5,
        2,
        1,
        "radar",
        2.4,
        0.80,
    )

    return world


# ================================================================
# V0.8 TESTS
# ================================================================

def run_tests():
    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V0.8")
    print("SPATIAL QUERIES")
    print("=" * 60)

    print()
    print("Creating shared 3D world...")

    world = create_demo_world()

    spatial = SpatialQueries(world)

    print()
    print("World size:", world.size)
    print(
        "Observed voxels:",
        len(world.voxels),
    )

    # ============================================================
    # TEST 1 - NEIGHBORS
    # ============================================================

    neighbors = spatial.neighbors(
        4,
        4,
        2,
        radius=1,
    )

    assert len(neighbors) == 26

    print()
    print(
        "Neighbor query test: PASS"
    )
    print(
        "  Neighbors around (4,4,2):",
        len(neighbors),
    )

    # ============================================================
    # TEST 2 - SOLID MATTER
    # ============================================================

    solid = spatial.solid_voxels()

    assert len(solid) >= 2

    solid_coordinates = {
        item["coordinate"]
        for item in solid
    }

    assert (4, 4, 2) in solid_coordinates
    assert (2, 5, 1) in solid_coordinates

    print(
        "Solid matter query test: PASS"
    )
    print(
        "  Solid voxels:",
        len(solid),
    )

    # ============================================================
    # TEST 3 - HOT ZONES
    # ============================================================

    hot = spatial.hot_zones(
        threshold=40.0
    )

    assert len(hot) == 1
    assert hot[0]["coordinate"] == (
        6,
        4,
        2,
    )

    print(
        "Hot zone query test: PASS"
    )
    print(
        "  Hot zones:",
        len(hot),
    )

    # ============================================================
    # TEST 4 - MOTION
    # ============================================================

    motion = spatial.motion_zones(
        threshold=1.0
    )

    assert len(motion) >= 2

    print(
        "Motion zone query test: PASS"
    )
    print(
        "  Motion zones:",
        len(motion),
    )

    # ============================================================
    # TEST 5 - UNKNOWN
    # ============================================================

    unknown = spatial.unknown_voxels()

    assert len(unknown) == (
        512 - len(world.voxels)
    )

    print(
        "UNKNOWN voxel query test: PASS"
    )
    print(
        "  Unknown voxels:",
        len(unknown),
    )

    # ============================================================
    # TEST 6 - UNKNOWN != OBSERVED ZERO
    # ============================================================

    world.update_sensor(
        1,
        1,
        1,
        "thermal",
        0.0,
        0.90,
    )

    unknown_after_zero = spatial.unknown_voxels()

    assert (1, 1, 1) not in (
        unknown_after_zero
    )

    zero_result = world.query(
        1,
        1,
        1,
        "temperature",
    )

    assert zero_result["state"] == (
        "OBSERVED_ZERO"
    )

    print(
        "UNKNOWN != OBSERVED_ZERO test: PASS"
    )

    # ============================================================
    # TEST 7 - LOCAL UNCERTAINTY
    # ============================================================

    uncertainty = spatial.local_uncertainty(
        4,
        4,
        2,
        radius=1,
    )

    assert (
        uncertainty["unknown_voxels"] >= 0
    )

    assert (
        0.0
        <= uncertainty["average_uncertainty"]
        <= 1.0
    )

    print(
        "Local uncertainty test: PASS"
    )

    print(
        "  Average uncertainty:",
        round(
            uncertainty[
                "average_uncertainty"
            ],
            3,
        ),
    )

    # ============================================================
    # TEST 8 - OBSTACLE
    # ============================================================

    obstacle = spatial.is_obstacle(
        4,
        4,
        2,
    )

    assert obstacle["obstacle"] is True

    print(
        "Obstacle detection test: PASS"
    )

    # ============================================================
    # TEST 9 - UNKNOWN OBSTACLE
    # ============================================================

    unknown_obstacle = spatial.is_obstacle(
        0,
        0,
        0,
    )

    assert (
        unknown_obstacle["obstacle"]
        is None
    )

    assert (
        unknown_obstacle["state"]
        == "UNKNOWN"
    )

    print(
        "Unknown obstacle test: PASS"
    )

    # ============================================================
    # TEST 10 - DIRECTIONAL QUERY
    # ============================================================

    path = spatial.check_direction(
        start=(3, 4, 2),
        direction=(1, 0, 0),
        distance=3,
    )

    assert len(path) == 3

    assert path[0]["coordinate"] == (
        4,
        4,
        2,
    )

    assert path[0]["status"] == (
        "OBSTACLE"
    )

    print(
        "Directional obstacle test: PASS"
    )

    # ============================================================
    # TEST 11 - UNKNOWN PATH
    # ============================================================

    unknown_path = spatial.check_direction(
        start=(0, 0, 0),
        direction=(0, 1, 0),
        distance=2,
    )

    assert len(unknown_path) == 2

    assert unknown_path[0]["status"] == (
        "UNKNOWN"
    )

    print(
        "Unknown path test: PASS"
    )

    # ============================================================
    # TEST 12 - SPATIAL SUMMARY
    # ============================================================

    summary = spatial.summary()

    assert (
        summary["observed_voxels"]
        == len(world.voxels)
    )

    print(
        "Spatial summary test: PASS"
    )

    # ============================================================
    # DISPLAY
    # ============================================================

    print()
    print("-" * 60)
    print("SPATIAL WORLD SUMMARY")
    print("-" * 60)

    print(
        "World size       :",
        summary["world_size"],
    )

    print(
        "Observed voxels  :",
        summary["observed_voxels"],
    )

    print(
        "Unknown voxels   :",
        summary["unknown_voxels"],
    )

    print(
        "Solid voxels     :",
        summary["solid_voxels"],
    )

    print(
        "Hot zones        :",
        summary["hot_zones"],
    )

    print(
        "Motion zones     :",
        summary["motion_zones"],
    )

    print()
    print("=" * 60)
    print("V0.8 TEST COMPLETE")
    print("=" * 60)

    print()
    print("All V0.8 tests passed.")

    print()
    print("Architecture validated:")
    print("  [OK] 3D spatial neighbors")
    print("  [OK] Solid matter queries")
    print("  [OK] Hot zone queries")
    print("  [OK] Motion zone queries")
    print("  [OK] UNKNOWN region queries")
    print("  [OK] Local uncertainty")
    print("  [OK] Obstacle detection")
    print("  [OK] Directional path queries")
    print("  [OK] UNKNOWN != OBSERVED_ZERO")
    print("  [OK] Sparse spatial reasoning")

    print("=" * 60)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    run_tests()