"""
Quantum-Embedding World Model
V0.9 - Sparse Compute / Adaptive Resolution

Purpose:
    Reduce unnecessary computation by operating only on
    spatially relevant regions of the 3D world.

Architecture:
    V0.7 -> Multi-sensor shared world
    V0.8 -> Spatial queries
    V0.9 -> Sparse compute

Important:
    This module does NOT replace world_model.py.

The underlying world remains:
    - 3D
    - sparse
    - uncertainty-aware
    - UNKNOWN != 0
"""

from world_model import SparseWorldModel


class SparseCompute:
    """
    Sparse computation layer for the 3D world model.

    Instead of processing all virtual voxels, this layer
    identifies active/observed regions and computes only
    where information exists or where a query explicitly
    requests analysis.
    """

    def __init__(self, world):
        self.world = world
        self.size = world.size

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def _inside_bounds(self, x, y, z):
        """Check whether a coordinate belongs to the virtual world."""

        sx, sy, sz = self.size

        return (
            0 <= x < sx
            and 0 <= y < sy
            and 0 <= z < sz
        )

    def _distance_squared(self, a, b):
        """Squared Euclidean distance between two 3D coordinates."""

        ax, ay, az = a
        bx, by, bz = b

        return (
            (ax - bx) ** 2
            + (ay - by) ** 2
            + (az - bz) ** 2
        )

    # ============================================================
    # 1. OBSERVED VOXELS
    # ============================================================

    def observed_voxels(self):
        """
        Return only voxels physically stored in the sparse world.

        This is the primary sparse-compute set.
        """

        return list(self.world.voxels.keys())

    # ============================================================
    # 2. ACTIVE VOXELS
    # ============================================================

    def active_voxels(self):
        """
        Identify voxels containing useful physical information.

        A voxel is active when it has at least one known property.
        """

        active = []

        for coordinate, embedding in self.world.voxels.items():

            try:
                known_count = embedding.known_count()
            except AttributeError:
                known_count = 0

            if known_count > 0:
                active.append(coordinate)

        return active

    # ============================================================
    # 3. ACTIVE REGION
    # ============================================================

    def active_region(
        self,
        center,
        radius=1,
    ):
        """
        Return observed/active voxels close to a 3D center.

        Only stored voxels are examined.
        Unknown virtual voxels are never materialized.
        """

        if radius < 0:
            raise ValueError("radius must be >= 0")

        if not self._inside_bounds(*center):
            raise ValueError(
                f"Center {center} is outside world bounds."
            )

        radius_squared = radius ** 2

        results = []

        for coordinate in self.active_voxels():

            if (
                self._distance_squared(
                    coordinate,
                    center,
                )
                <= radius_squared
            ):
                results.append(coordinate)

        return results

    # ============================================================
    # 4. COMPUTATION CANDIDATES
    # ============================================================

    def computation_candidates(
        self,
        center=None,
        radius=None,
    ):
        """
        Determine which voxels should be processed.

        Cases:

        1. No center/radius:
           process all active voxels.

        2. Center + radius:
           process only active voxels inside region.
        """

        if center is None and radius is None:
            return self.active_voxels()

        if center is None or radius is None:
            raise ValueError(
                "center and radius must be provided together."
            )

        return self.active_region(
            center,
            radius,
        )

    # ============================================================
    # 5. PROPERTY COMPUTATION
    # ============================================================

    def compute_property(
        self,
        property_name,
        center=None,
        radius=None,
    ):
        """
        Compute a property only for sparse candidates.

        Returns structured results while preserving:
            value
            state
            confidence
            uncertainty
        """

        candidates = self.computation_candidates(
            center=center,
            radius=radius,
        )

        results = []

        for coordinate in candidates:

            x, y, z = coordinate

            result = self.world.query(
                x,
                y,
                z,
                property_name,
            )

            if not result:
                continue

            results.append({
                "coordinate": coordinate,
                "value": result.get("value"),
                "state": result.get(
                    "state",
                    "UNKNOWN",
                ),
                "confidence": result.get(
                    "confidence",
                    0.0,
                ),
                "uncertainty": result.get(
                    "uncertainty",
                    1.0,
                ),
                "source": result.get(
                    "source"
                ),
            })

        return results

    # ============================================================
    # 6. ACTIVE PROPERTY VOXELS
    # ============================================================

    def property_voxels(
        self,
        property_name,
    ):
        """
        Return only sparse voxels where a property is known.
        """

        results = self.compute_property(
            property_name
        )

        return [
            item
            for item in results
            if item["state"] != "UNKNOWN"
        ]

    # ============================================================
    # 7. HIGH-CONFIDENCE REGION
    # ============================================================

    def high_confidence_voxels(
        self,
        min_confidence=0.8,
    ):
        """
        Find sparse voxels containing at least one highly
        confident physical observation.
        """

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0 and 1."
            )

        results = []

        for coordinate, embedding in self.world.voxels.items():

            try:
                properties = embedding.known_properties()
            except AttributeError:
                properties = []

            high_confidence = False

            for property_name in properties:

                result = self.world.query(
                    coordinate[0],
                    coordinate[1],
                    coordinate[2],
                    property_name,
                )

                if not result:
                    continue

                confidence = result.get(
                    "confidence",
                    0.0,
                )

                if confidence >= min_confidence:
                    high_confidence = True
                    break

            if high_confidence:
                results.append(coordinate)

        return results

    # ============================================================
    # 8. UNCERTAINTY-FOCUSED REGION
    # ============================================================

    def uncertain_voxels(
        self,
        min_uncertainty=0.5,
    ):
        """
        Find sparse voxels where at least one known property
        has significant uncertainty.
        """

        if not 0.0 <= min_uncertainty <= 1.0:
            raise ValueError(
                "min_uncertainty must be between 0 and 1."
            )

        results = []

        for coordinate, embedding in self.world.voxels.items():

            try:
                properties = embedding.known_properties()
            except AttributeError:
                properties = []

            uncertain = False

            for property_name in properties:

                result = self.world.query(
                    coordinate[0],
                    coordinate[1],
                    coordinate[2],
                    property_name,
                )

                if not result:
                    continue

                uncertainty = result.get(
                    "uncertainty",
                    1.0,
                )

                if uncertainty >= min_uncertainty:
                    uncertain = True
                    break

            if uncertain:
                results.append(coordinate)

        return results

    # ============================================================
    # 9. SPATIAL CELL GROUPING
    # ============================================================

    def spatial_cells(
        self,
        cell_size=2,
    ):
        """
        Group sparse voxels into coarse 3D spatial cells.

        Example with cell_size=2:

            voxel (0,0,0) -> cell (0,0,0)
            voxel (1,1,1) -> cell (0,0,0)
            voxel (2,2,2) -> cell (1,1,1)

        This provides a lightweight adaptive-resolution
        foundation without changing the underlying voxel model.
        """

        if cell_size < 1:
            raise ValueError(
                "cell_size must be >= 1"
            )

        cells = {}

        for coordinate in self.active_voxels():

            x, y, z = coordinate

            cell = (
                x // cell_size,
                y // cell_size,
                z // cell_size,
            )

            if cell not in cells:
                cells[cell] = []

            cells[cell].append(
                coordinate
            )

        return cells

    # ============================================================
    # 10. ACTIVE CELL SUMMARY
    # ============================================================

    def cell_summary(
        self,
        cell_size=2,
    ):
        """
        Summarize coarse 3D cells.
        """

        cells = self.spatial_cells(
            cell_size
        )

        results = []

        for cell, coordinates in cells.items():

            results.append({
                "cell": cell,
                "active_voxels": len(
                    coordinates
                ),
                "coordinates": coordinates,
            })

        return results

    # ============================================================
    # 11. SPARSE COMPUTE METRICS
    # ============================================================

    def metrics(self):
        """
        Measure the difference between the full virtual
        computation space and the actual sparse computation set.
        """

        sx, sy, sz = self.size

        total_virtual = (
            sx * sy * sz
        )

        observed = len(
            self.world.voxels
        )

        active = len(
            self.active_voxels()
        )

        if total_virtual > 0:
            observed_ratio = (
                observed
                / total_virtual
            )

            active_ratio = (
                active
                / total_virtual
            )

            reduction_ratio = (
                1.0
                - active_ratio
            )
        else:
            observed_ratio = 0.0
            active_ratio = 0.0
            reduction_ratio = 0.0

        return {
            "virtual_voxels": total_virtual,
            "observed_voxels": observed,
            "active_voxels": active,
            "unknown_voxels": (
                total_virtual - observed
            ),
            "observed_ratio": observed_ratio,
            "active_ratio": active_ratio,
            "compute_reduction_ratio": reduction_ratio,
        }

    # ============================================================
    # 12. COMPUTE PLAN
    # ============================================================

    def compute_plan(
        self,
        center=None,
        radius=None,
    ):
        """
        Build an explicit sparse computation plan.

        This does not perform prediction or control yet.
        It identifies where computation should occur.
        """

        candidates = self.computation_candidates(
            center=center,
            radius=radius,
        )

        return {
            "center": center,
            "radius": radius,
            "candidate_count": len(
                candidates
            ),
            "candidates": candidates,
        }


# ================================================================
# DEMO WORLD
# ================================================================

def create_demo_world():
    """
    Create the same physical world structure used in V0.7/V0.8.
    """

    world = SparseWorldModel(
        size=(8, 8, 8)
    )

    # Central object
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

    # Thermal region
    world.update_sensor(
        6,
        4,
        2,
        "thermal",
        42.0,
        0.85,
    )

    # Second object
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

    # Moving object
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
# V0.9 TESTS
# ================================================================

def run_tests():

    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V0.9")
    print("SPARSE COMPUTE / ADAPTIVE RESOLUTION")
    print("=" * 60)

    print()
    print("Creating shared 3D world...")

    world = create_demo_world()

    sparse = SparseCompute(world)

    print()
    print("World size:", world.size)

    print(
        "Virtual voxels:",
        world.size[0]
        * world.size[1]
        * world.size[2],
    )

    print(
        "Observed voxels:",
        len(world.voxels),
    )

    # ============================================================
    # TEST 1 - OBSERVED VOXELS
    # ============================================================

    observed = sparse.observed_voxels()

    assert len(observed) == 4

    print()
    print(
        "Observed sparse set test: PASS"
    )

    # ============================================================
    # TEST 2 - ACTIVE VOXELS
    # ============================================================

    active = sparse.active_voxels()

    assert len(active) == 4

    print(
        "Active voxel detection test: PASS"
    )

    # ============================================================
    # TEST 3 - ACTIVE REGION
    # ============================================================

    region = sparse.active_region(
        center=(4, 4, 2),
        radius=2,
    )

    assert (4, 4, 2) in region

    print(
        "Active 3D region test: PASS"
    )
    print(
        "  Active voxels near center:",
        len(region),
    )

    # ============================================================
    # TEST 4 - PROPERTY COMPUTATION
    # ============================================================

    temperature = sparse.compute_property(
        "temperature"
    )

    known_temperature = [
        item
        for item in temperature
        if item["state"] != "UNKNOWN"
    ]

    assert len(known_temperature) == 1

    assert (
        known_temperature[0]["value"]
        == 42.0
    )

    print(
        "Sparse property computation test: PASS"
    )

    # ============================================================
    # TEST 5 - SOLID MATTER
    # ============================================================

    matter = sparse.property_voxels(
        "matter"
    )

    assert len(matter) == 2

    print(
        "Sparse matter computation test: PASS"
    )

    # ============================================================
    # TEST 6 - HIGH CONFIDENCE
    # ============================================================

    high_confidence = (
        sparse.high_confidence_voxels(
            min_confidence=0.90
        )
    )

    assert (
        (4, 4, 2)
        in high_confidence
    )

    print(
        "High-confidence region test: PASS"
    )

    # ============================================================
    # TEST 7 - UNCERTAINTY
    # ============================================================

    uncertain = sparse.uncertain_voxels(
        min_uncertainty=0.20
    )

    assert (
        (4, 4, 2)
        in uncertain
    )

    print(
        "Uncertainty-focused computation test: PASS"
    )

    # ============================================================
    # TEST 8 - SPATIAL CELLS
    # ============================================================

    cells = sparse.spatial_cells(
        cell_size=2
    )

    assert len(cells) > 0

    assert (
        (4 // 2, 4 // 2, 2 // 2)
        in cells
    )

    print(
        "3D spatial cell grouping test: PASS"
    )

    # ============================================================
    # TEST 9 - CELL SUMMARY
    # ============================================================

    cell_summary = sparse.cell_summary(
        cell_size=2
    )

    assert len(cell_summary) > 0

    print(
        "Adaptive-resolution cell summary test: PASS"
    )

    # ============================================================
    # TEST 10 - METRICS
    # ============================================================

    metrics = sparse.metrics()

    assert (
        metrics["virtual_voxels"]
        == 512
    )

    assert (
        metrics["observed_voxels"]
        == 4
    )

    assert (
        metrics["active_voxels"]
        == 4
    )

    assert (
        metrics["unknown_voxels"]
        == 508
    )

    assert (
        0.0
        <= metrics[
            "compute_reduction_ratio"
        ]
        <= 1.0
    )

    print(
        "Sparse compute metrics test: PASS"
    )

    # ============================================================
    # TEST 11 - COMPUTE PLAN
    # ============================================================

    plan = sparse.compute_plan(
        center=(4, 4, 2),
        radius=2,
    )

    assert (
        plan["candidate_count"]
        == len(region)
    )

    print(
        "Sparse computation plan test: PASS"
    )

    # ============================================================
    # TEST 12 - UNKNOWN SPACE IS NOT MATERIALIZED
    # ============================================================

    before = len(
        world.voxels
    )

    sparse.compute_property(
        "temperature"
    )

    after = len(
        world.voxels
    )

    assert before == after

    print(
        "No UNKNOWN materialization test: PASS"
    )

    # ============================================================
    # TEST 13 - ADD OBSERVATION
    # ============================================================

    world.update_sensor(
        1,
        1,
        1,
        "thermal",
        55.0,
        0.90,
    )

    active_after = (
        sparse.active_voxels()
    )

    assert (
        len(active_after) == 5
    )

    assert (
        (1, 1, 1)
        in active_after
    )

    print(
        "Dynamic sparse update test: PASS"
    )

    # ============================================================
    # FINAL METRICS
    # ============================================================

    final_metrics = sparse.metrics()

    print()
    print("-" * 60)
    print("SPARSE COMPUTE SUMMARY")
    print("-" * 60)

    print(
        "Virtual voxels       :",
        final_metrics[
            "virtual_voxels"
        ],
    )

    print(
        "Observed voxels      :",
        final_metrics[
            "observed_voxels"
        ],
    )

    print(
        "Active voxels        :",
        final_metrics[
            "active_voxels"
        ],
    )

    print(
        "Unknown voxels       :",
        final_metrics[
            "unknown_voxels"
        ],
    )

    print(
        "Observed ratio       :",
        f"{final_metrics['observed_ratio'] * 100:.2f}%",
    )

    print(
        "Active ratio         :",
        f"{final_metrics['active_ratio'] * 100:.2f}%",
    )

    print(
        "Compute reduction    :",
        f"{final_metrics['compute_reduction_ratio'] * 100:.2f}%",
    )

    print()
    print("=" * 60)
    print("V0.9 TEST COMPLETE")
    print("=" * 60)

    print()
    print("All V0.9 tests passed.")

    print()
    print("Architecture validated:")
    print("  [OK] Sparse observed voxel set")
    print("  [OK] Active voxel detection")
    print("  [OK] Active 3D regions")
    print("  [OK] Sparse property computation")
    print("  [OK] High-confidence regions")
    print("  [OK] Uncertainty-focused computation")
    print("  [OK] 3D spatial cell grouping")
    print("  [OK] Adaptive-resolution foundation")
    print("  [OK] Sparse compute metrics")
    print("  [OK] Explicit computation plans")
    print("  [OK] UNKNOWN space remains unmaterialized")
    print("  [OK] Dynamic sensor updates")

    print("=" * 60)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    run_tests()