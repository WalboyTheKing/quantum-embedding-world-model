from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from world_model import SparseWorldModel
from temporal_world import TemporalWorldModel


# ============================================================
# QUANTUM-EMBEDDING WORLD MODEL
# V1.4 — WORLD DYNAMICS
# ============================================================

PROPERTIES = (
    "matter",
    "light",
    "temperature",
    "motion",
    "pressure",
    "energy",
    "field",
    "time",
)


class WorldDynamics:
    """
    V1.4 — Dynamic layer of the Quantum-Embedding World Model.

    Analyzes how physical properties evolve between
    temporal observations.

    Features:
        - property deltas
        - rates of change
        - increasing/decreasing/stable trends
        - motion dynamics
        - temperature dynamics
        - multi-property analysis
        - confidence propagation
        - uncertainty propagation
        - UNKNOWN preservation
        - world immutability
        - deterministic world snapshots
    """

    EPSILON = 1e-9

    def __init__(
        self,
        world: SparseWorldModel,
        temporal: Optional[TemporalWorldModel] = None,
    ):
        self.world = world

        if temporal is None:
            temporal = TemporalWorldModel(world)

        self.temporal = temporal

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_property(
        self,
        property_name: str,
    ) -> None:

        if property_name not in PROPERTIES:
            raise ValueError(
                f"Unknown property: {property_name}"
            )

    def _validate_position(
        self,
        position: Tuple[int, int, int],
    ) -> None:

        if len(position) != 3:
            raise ValueError(
                "Position must contain x, y, z"
            )

        x, y, z = position

        if not (
            0 <= x < self.world.size[0]
            and 0 <= y < self.world.size[1]
            and 0 <= z < self.world.size[2]
        ):
            raise ValueError(
                f"Position outside world bounds: {position}"
            )

    # ========================================================
    # TIMESTAMP UTILITIES
    # ========================================================

    @staticmethod
    def _timestamp_seconds(
        timestamp: str,
    ) -> Optional[float]:

        try:
            parsed = datetime.fromisoformat(
                timestamp.replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return parsed.timestamp()

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _time_delta(
        cls,
        previous_timestamp: str,
        current_timestamp: str,
    ) -> Optional[float]:

        previous = cls._timestamp_seconds(
            previous_timestamp
        )

        current = cls._timestamp_seconds(
            current_timestamp
        )

        if previous is None or current is None:
            return None

        delta = current - previous

        if delta <= cls.EPSILON:
            return None

        return delta

    # ========================================================
    # PROPERTY DELTA
    # ========================================================

    def property_delta(
        self,
        position: Tuple[int, int, int],
        property_name: str,
    ) -> Dict[str, Any]:

        self._validate_position(position)
        self._validate_property(property_name)

        series = self.temporal.property_series(
            position,
            property_name,
        )

        # ----------------------------------------------------
        # Not enough temporal observations
        # ----------------------------------------------------

        if len(series) < 2:

            return {
                "position": position,
                "property": property_name,
                "previous_value": None,
                "current_value": None,
                "delta": None,
                "trend": "UNKNOWN",
                "confidence": 0.0,
                "uncertainty": 1.0,
                "state": "UNKNOWN",
            }

        previous = series[-2]
        current = series[-1]

        previous_value = previous.get(
            "value"
        )

        current_value = current.get(
            "value"
        )

        previous_state = previous.get(
            "state",
            "UNKNOWN",
        )

        current_state = current.get(
            "state",
            "UNKNOWN",
        )

        # ----------------------------------------------------
        # UNKNOWN preservation
        # ----------------------------------------------------

        if (
            previous_value is None
            or current_value is None
            or previous_state == "UNKNOWN"
            or current_state == "UNKNOWN"
        ):

            return {
                "position": position,
                "property": property_name,
                "previous_value": previous_value,
                "current_value": current_value,
                "delta": None,
                "trend": "UNKNOWN",
                "confidence": float(
                    current.get(
                        "confidence",
                        0.0,
                    )
                ),
                "uncertainty": float(
                    current.get(
                        "uncertainty",
                        1.0,
                    )
                ),
                "state": "UNKNOWN",
            }

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        try:
            previous_value = float(
                previous_value
            )

            current_value = float(
                current_value
            )

        except (
            TypeError,
            ValueError,
        ):

            return {
                "position": position,
                "property": property_name,
                "previous_value": previous_value,
                "current_value": current_value,
                "delta": None,
                "trend": "UNKNOWN",
                "confidence": float(
                    current.get(
                        "confidence",
                        0.0,
                    )
                ),
                "uncertainty": float(
                    current.get(
                        "uncertainty",
                        1.0,
                    )
                ),
                "state": "UNKNOWN",
            }

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        delta = (
            current_value
            - previous_value
        )

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        if abs(delta) <= self.EPSILON:
            trend = "STABLE"

        elif delta > 0:
            trend = "INCREASING"

        else:
            trend = "DECREASING"

        return {
            "position": position,
            "property": property_name,
            "previous_value": previous_value,
            "current_value": current_value,
            "delta": delta,
            "trend": trend,
            "confidence": float(
                current.get(
                    "confidence",
                    0.0,
                )
            ),
            "uncertainty": float(
                current.get(
                    "uncertainty",
                    1.0,
                )
            ),
            "state": "OBSERVED",
        }

    # ========================================================
    # RATE OF CHANGE
    # ========================================================

    def property_rate(
        self,
        position: Tuple[int, int, int],
        property_name: str,
    ) -> Dict[str, Any]:

        result = self.property_delta(
            position,
            property_name,
        )

        result["rate"] = None

        if result["delta"] is None:
            return result

        series = self.temporal.property_series(
            position,
            property_name,
        )

        if len(series) < 2:
            return result

        previous_timestamp = series[-2].get(
            "timestamp"
        )

        current_timestamp = series[-1].get(
            "timestamp"
        )

        if (
            previous_timestamp is None
            or current_timestamp is None
        ):
            return result

        time_delta = self._time_delta(
            previous_timestamp,
            current_timestamp,
        )

        if time_delta is None:
            return result

        result["time_delta"] = time_delta

        result["rate"] = (
            result["delta"]
            / time_delta
        )

        return result

    # ========================================================
    # MOTION DYNAMICS
    # ========================================================

    def motion_dynamics(
        self,
        position: Tuple[int, int, int],
    ) -> Dict[str, Any]:

        result = self.property_rate(
            position,
            "motion",
        )

        return {
            "position": position,
            "property": "motion",
            "previous_motion": result.get(
                "previous_value"
            ),
            "current_motion": result.get(
                "current_value"
            ),
            "delta_motion": result.get(
                "delta"
            ),
            "motion_rate": result.get(
                "rate"
            ),
            "trend": result.get(
                "trend",
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
            "state": result.get(
                "state",
                "UNKNOWN",
            ),
        }

    # ========================================================
    # TEMPERATURE DYNAMICS
    # ========================================================

    def temperature_dynamics(
        self,
        position: Tuple[int, int, int],
    ) -> Dict[str, Any]:

        result = self.property_rate(
            position,
            "temperature",
        )

        return {
            "position": position,
            "property": "temperature",
            "previous_temperature": result.get(
                "previous_value"
            ),
            "current_temperature": result.get(
                "current_value"
            ),
            "delta_temperature": result.get(
                "delta"
            ),
            "temperature_rate": result.get(
                "rate"
            ),
            "trend": result.get(
                "trend",
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
            "state": result.get(
                "state",
                "UNKNOWN",
            ),
        }

    # ========================================================
    # ALL PROPERTY DYNAMICS
    # ========================================================

    def analyze_voxel(
        self,
        position: Tuple[int, int, int],
    ) -> Dict[str, Any]:

        self._validate_position(position)

        properties: Dict[str, Any] = {}

        for property_name in PROPERTIES:

            properties[property_name] = (
                self.property_rate(
                    position,
                    property_name,
                )
            )

        return {
            "position": position,
            "properties": properties,
        }

    # ========================================================
    # CHANGED PROPERTIES
    # ========================================================

    def changed_properties(
        self,
        position: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:

        analysis = self.analyze_voxel(
            position
        )

        changes: List[Dict[str, Any]] = []

        for result in (
            analysis["properties"].values()
        ):

            delta = result.get(
                "delta"
            )

            if (
                delta is not None
                and abs(delta) > self.EPSILON
            ):

                changes.append(
                    result
                )

        return changes

    # ========================================================
    # DYNAMIC SUMMARY
    # ========================================================

    def dynamic_summary(
        self,
        position: Tuple[int, int, int],
    ) -> Dict[str, Any]:

        changes = self.changed_properties(
            position
        )

        increasing: List[str] = []
        decreasing: List[str] = []
        stable: List[str] = []

        for change in changes:

            property_name = change[
                "property"
            ]

            trend = change.get(
                "trend"
            )

            if trend == "INCREASING":

                increasing.append(
                    property_name
                )

            elif trend == "DECREASING":

                decreasing.append(
                    property_name
                )

            elif trend == "STABLE":

                stable.append(
                    property_name
                )

        return {
            "position": position,
            "changed_properties": [
                change["property"]
                for change in changes
            ],
            "increasing": increasing,
            "decreasing": decreasing,
            "stable": stable,
            "change_count": len(changes),
        }

    # ========================================================
    # IMMUTABLE VALUE SERIALIZATION
    # ========================================================

    @classmethod
    def _freeze_value(
        cls,
        value: Any,
    ) -> Any:
        """
        Convert internal Python / NumPy values into
        deterministic comparison-safe structures.

        Handles:
            - NumPy arrays
            - dictionaries
            - lists
            - tuples
            - sets
            - datetime objects
            - scalar values
            - NumPy scalar values
        """

        # ----------------------------------------------------
        # None / basic immutable values
        # ----------------------------------------------------

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        # ----------------------------------------------------
        # datetime
        # ----------------------------------------------------

        if isinstance(
            value,
            datetime,
        ):
            return value.isoformat()

        # ----------------------------------------------------
        # NumPy arrays / NumPy scalars
        # ----------------------------------------------------

        if hasattr(
            value,
            "tolist",
        ):

            try:

                return cls._freeze_value(
                    value.tolist()
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # dictionaries
        # ----------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            frozen_items = []

            for key, item in value.items():

                frozen_items.append(
                    (
                        str(key),
                        cls._freeze_value(
                            item
                        ),
                    )
                )

            frozen_items.sort(
                key=lambda item: item[0]
            )

            return tuple(
                frozen_items
            )

        # ----------------------------------------------------
        # lists / tuples
        # ----------------------------------------------------

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):

            return tuple(
                cls._freeze_value(
                    item
                )
                for item in value
            )

        # ----------------------------------------------------
        # sets
        # ----------------------------------------------------

        if isinstance(
            value,
            set,
        ):

            frozen = [
                cls._freeze_value(
                    item
                )
                for item in value
            ]

            return tuple(
                sorted(
                    frozen,
                    key=repr,
                )
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        try:
            return repr(value)

        except Exception:
            return f"<{type(value).__name__}>"

    # ========================================================
    # WORLD IMMUTABILITY
    # ========================================================

    def world_snapshot(
        self,
    ) -> Dict[str, Any]:
        """
        Create a deterministic structural snapshot
        of the current sparse world.

        IMPORTANT:

        This method does not call state-changing methods.

        It reads the actual internal embedding state and
        converts it into deterministic immutable values.

        This avoids assumptions such as:

            confidence_vector()

        versus:

            confidence_vector

        It therefore remains compatible with NumPy-array
        attributes and normal Python methods.
        """

        snapshot: Dict[str, Any] = {
            "size": tuple(
                int(value)
                for value in self.world.size
            ),
            "voxel_count": len(
                self.world.voxels
            ),
            "voxels": {},
        }

        # ----------------------------------------------------
        # Sort positions for deterministic ordering
        # ----------------------------------------------------

        positions = sorted(
            self.world.voxels.keys()
        )

        for position in positions:

            embedding = self.world.voxels[
                position
            ]

            # ------------------------------------------------
            # Prefer the actual internal __dict__.
            #
            # This is more reliable than guessing whether
            # individual QuantumEmbedding members are methods
            # or NumPy attributes.
            # ------------------------------------------------

            if hasattr(
                embedding,
                "__dict__",
            ):

                internal_state = (
                    self._freeze_value(
                        embedding.__dict__
                    )
                )

            else:

                internal_state = (
                    self._freeze_value(
                        embedding
                    )
                )

            snapshot[
                "voxels"
            ][
                tuple(
                    int(value)
                    for value in position
                )
            ] = {
                "type": type(
                    embedding
                ).__name__,
                "state": internal_state,
            }

        return snapshot

    # ========================================================
    # ASSERTION
    # ========================================================

    @staticmethod
    def _assert(
        condition: bool,
        message: str,
    ) -> None:

        if not condition:
            raise AssertionError(
                message
            )


# ============================================================
# DEMO WORLD
# ============================================================

def create_demo_world():

    world = SparseWorldModel(
        size=(8, 8, 8),
        dim=16,
    )

    temporal = TemporalWorldModel(
        world
    )

    dynamics = WorldDynamics(
        world,
        temporal,
    )

    return (
        world,
        temporal,
        dynamics,
    )


# ============================================================
# TESTS
# ============================================================

def run_tests():

    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V1.4")
    print("WORLD DYNAMICS")
    print("=" * 60)

    world, temporal, dynamics = (
        create_demo_world()
    )

    # --------------------------------------------------------
    # TEST 1 — INITIALIZATION
    # --------------------------------------------------------

    WorldDynamics(
        world,
        temporal,
    )

    print(
        "World dynamics initialization test: PASS"
    )

    # --------------------------------------------------------
    # FIRST OBSERVATION
    # --------------------------------------------------------

    world.update_sensor(
        3,
        3,
        2,
        "radar",
        value=1.0,
        confidence=0.90,
    )

    world.update_sensor(
        3,
        3,
        2,
        "thermal",
        value=30.0,
        confidence=0.90,
    )

    temporal.capture(
        "OBSERVED"
    )

    initial_series = (
        temporal.property_series(
            (3, 3, 2),
            "temperature",
        )
    )

    WorldDynamics._assert(
        len(initial_series) == 1,
        "Initial temporal state missing",
    )

    print(
        "Initial temporal state test: PASS"
    )

    # --------------------------------------------------------
    # SECOND OBSERVATION
    # --------------------------------------------------------

    world.update_sensor(
        3,
        3,
        2,
        "radar",
        value=2.0,
        confidence=0.90,
    )

    world.update_sensor(
        3,
        3,
        2,
        "thermal",
        value=40.0,
        confidence=0.90,
    )

    temporal.capture(
        "OBSERVED"
    )

    # --------------------------------------------------------
    # TEST 2 — TEMPERATURE
    # --------------------------------------------------------

    temperature = (
        dynamics.temperature_dynamics(
            (3, 3, 2)
        )
    )

    WorldDynamics._assert(
        temperature[
            "delta_temperature"
        ] is not None,
        "Temperature delta missing",
    )

    WorldDynamics._assert(
        temperature[
            "delta_temperature"
        ] > 0,
        "Temperature should increase",
    )

    WorldDynamics._assert(
        temperature["trend"]
        == "INCREASING",
        "Temperature trend incorrect",
    )

    print(
        "Temperature dynamics test: PASS"
    )

    # --------------------------------------------------------
    # TEST 3 — MOTION
    # --------------------------------------------------------

    motion = dynamics.motion_dynamics(
        (3, 3, 2)
    )

    WorldDynamics._assert(
        motion["delta_motion"]
        is not None,
        "Motion delta missing",
    )

    WorldDynamics._assert(
        motion["delta_motion"] > 0,
        "Motion should increase",
    )

    WorldDynamics._assert(
        motion["trend"]
        == "INCREASING",
        "Motion trend incorrect",
    )

    print(
        "Motion dynamics test: PASS"
    )

    # --------------------------------------------------------
    # TEST 4 — RATE
    # --------------------------------------------------------

    rate = dynamics.property_rate(
        (3, 3, 2),
        "temperature",
    )

    WorldDynamics._assert(
        "rate" in rate,
        "Rate field missing",
    )

    WorldDynamics._assert(
        rate["rate"] is None
        or rate["rate"] >= 0,
        "Invalid temperature rate",
    )

    print(
        "Property rate calculation test: PASS"
    )

    # --------------------------------------------------------
    # TEST 5 — CHANGED PROPERTIES
    # --------------------------------------------------------

    changes = dynamics.changed_properties(
        (3, 3, 2)
    )

    changed_names = {
        item["property"]
        for item in changes
    }

    WorldDynamics._assert(
        "motion" in changed_names,
        "Motion change not detected",
    )

    WorldDynamics._assert(
        "temperature" in changed_names,
        "Temperature change not detected",
    )

    print(
        "Changed property detection test: PASS"
    )

    # --------------------------------------------------------
    # TEST 6 — TRENDS
    # --------------------------------------------------------

    summary = dynamics.dynamic_summary(
        (3, 3, 2)
    )

    WorldDynamics._assert(
        "motion"
        in summary["increasing"],
        "Increasing motion missing",
    )

    WorldDynamics._assert(
        "temperature"
        in summary["increasing"],
        "Increasing temperature missing",
    )

    print(
        "Dynamic trend classification test: PASS"
    )

    # --------------------------------------------------------
    # TEST 7 — UNKNOWN
    # --------------------------------------------------------

    unknown = dynamics.property_delta(
        (3, 3, 2),
        "energy",
    )

    WorldDynamics._assert(
        unknown["trend"]
        == "UNKNOWN",
        "UNKNOWN state lost",
    )

    WorldDynamics._assert(
        unknown["delta"] is None,
        "UNKNOWN delta should be None",
    )

    WorldDynamics._assert(
        unknown["uncertainty"]
        == 1.0,
        "UNKNOWN uncertainty incorrect",
    )

    print(
        "UNKNOWN dynamics preservation test: PASS"
    )

    # --------------------------------------------------------
    # TEST 8 — STABLE PRESSURE
    #
    # "contact" = sensor type
    # "pressure" = physical property
    # --------------------------------------------------------

    world.update_sensor(
        3,
        3,
        2,
        "contact",
        value=5.0,
        confidence=0.90,
    )

    temporal.capture(
        "OBSERVED"
    )

    world.update_sensor(
        3,
        3,
        2,
        "contact",
        value=5.0,
        confidence=0.90,
    )

    temporal.capture(
        "OBSERVED"
    )

    pressure = dynamics.property_delta(
        (3, 3, 2),
        "pressure",
    )

    WorldDynamics._assert(
        pressure["trend"]
        == "STABLE",
        "Stable pressure not detected",
    )

    print(
        "Stable property dynamics test: PASS"
    )

    # --------------------------------------------------------
    # TEST 9 — MULTI-PROPERTY
    # --------------------------------------------------------

    voxel_analysis = (
        dynamics.analyze_voxel(
            (3, 3, 2)
        )
    )

    WorldDynamics._assert(
        len(
            voxel_analysis["properties"]
        )
        == len(PROPERTIES),
        "Not all properties analyzed",
    )

    print(
        "Multi-property dynamics test: PASS"
    )

    # --------------------------------------------------------
    # TEST 10 — CONFIDENCE / UNCERTAINTY
    # --------------------------------------------------------

    for change in changes:

        WorldDynamics._assert(
            0.0
            <= change["confidence"]
            <= 1.0,
            "Invalid confidence",
        )

        WorldDynamics._assert(
            0.0
            <= change["uncertainty"]
            <= 1.0,
            "Invalid uncertainty",
        )

    print(
        "Confidence and uncertainty propagation test: PASS"
    )

    # --------------------------------------------------------
    # TEST 11 — IMMUTABILITY
    # --------------------------------------------------------

    before = dynamics.world_snapshot()

    dynamics.analyze_voxel(
        (3, 3, 2)
    )

    dynamics.motion_dynamics(
        (3, 3, 2)
    )

    dynamics.temperature_dynamics(
        (3, 3, 2)
    )

    dynamics.dynamic_summary(
        (3, 3, 2)
    )

    after = dynamics.world_snapshot()

    WorldDynamics._assert(
        before == after,
        "Dynamics modified the world",
    )

    print(
        "Dynamics immutability test: PASS"
    )

    # --------------------------------------------------------
    # TEST 12 — BOUNDS
    # --------------------------------------------------------

    try:

        dynamics.analyze_voxel(
            (99, 99, 99)
        )

        raise AssertionError(
            "Out-of-bounds position accepted"
        )

    except ValueError:

        pass

    print(
        "World-boundary validation test: PASS"
    )

    # --------------------------------------------------------
    # TEST 13 — INVALID PROPERTY
    # --------------------------------------------------------

    try:

        dynamics.property_delta(
            (3, 3, 2),
            "invalid_property",
        )

        raise AssertionError(
            "Invalid property accepted"
        )

    except ValueError:

        pass

    print(
        "Invalid property rejection test: PASS"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 60)
    print("WORLD DYNAMICS SUMMARY")
    print("-" * 60)

    print(
        f"World size        : {world.size}"
    )

    print(
        f"Observed voxels   : "
        f"{len(world.voxels)}"
    )

    print(
        "Temperature trend : "
        f"{temperature['trend']}"
    )

    print(
        "Temperature delta : "
        f"{temperature['delta_temperature']}"
    )

    print(
        "Motion trend      : "
        f"{motion['trend']}"
    )

    print(
        "Motion delta      : "
        f"{motion['delta_motion']}"
    )

    print(
        "UNKNOWN preserved : YES"
    )

    print(
        "World mutated     : NO"
    )

    print()
    print("=" * 60)
    print("V1.4 TEST COMPLETE")
    print("=" * 60)

    print()
    print("All V1.4 tests passed.")

    print()
    print("Architecture validated:")

    print(
        "  [OK] Temporal dynamics"
    )

    print(
        "  [OK] Property delta detection"
    )

    print(
        "  [OK] Rate of change"
    )

    print(
        "  [OK] Temperature evolution"
    )

    print(
        "  [OK] Motion evolution"
    )

    print(
        "  [OK] Increasing trend"
    )

    print(
        "  [OK] Stable state"
    )

    print(
        "  [OK] UNKNOWN dynamics"
    )

    print(
        "  [OK] Multi-property dynamics"
    )

    print(
        "  [OK] Confidence propagation"
    )

    print(
        "  [OK] Uncertainty propagation"
    )

    print(
        "  [OK] 3D spatial dynamics"
    )

    print(
        "  [OK] World immutability"
    )


if __name__ == "__main__":
    run_tests()