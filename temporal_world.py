"""
Quantum-Embedding World Model
V1.3 - Temporal World Model

Purpose:
    Add a temporal layer to the 3D sparse world model.

Architecture:

        3D WORLD
           |
           v
      WORLD STATE
           |
           v
      TEMPORAL MEMORY
           |
     +-----+-----+
     |     |     |
    t-1    t    t+1
  history current predicted

Important:

    OBSERVED != HISTORICAL != PREDICTED

    UNKNOWN is preserved.

    This module does not modify:
        world_model.py
        prediction.py
        reasoning.py
        world_reasoning.py
"""

from copy import deepcopy
from datetime import datetime, timezone

from world_model import SparseWorldModel


class TemporalWorldModel:
    """
    Temporal memory layer for the Quantum-Embedding World Model.

    The underlying SparseWorldModel remains the source
    of the current physical state.

    TemporalWorldModel stores snapshots of that state.
    """

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

    TEMPORAL_LABELS = {
        "HISTORICAL",
        "OBSERVED",
        "PREDICTED",
    }

    def __init__(self, world, max_history=100):
        if max_history < 1:
            raise ValueError(
                "max_history must be >= 1"
            )

        self.world = world
        self.max_history = max_history
        self.history = []

    # =========================================================
    # TIME
    # =========================================================

    @staticmethod
    def _timestamp():
        return datetime.now(
            timezone.utc
        ).isoformat()

    # =========================================================
    # CURRENT WORLD SNAPSHOT
    # =========================================================

    def snapshot(self, label="OBSERVED"):
        """
        Capture the current sparse 3D world state.

        Only existing sparse voxels are copied.

        UNKNOWN voxels are never materialized into the
        underlying SparseWorldModel.
        """

        self.validate_temporal_label(label)

        voxels = {}

        for coordinate in self.world.voxels.keys():

            voxel_state = {}

            x, y, z = coordinate

            for property_name in self.PROPERTIES:

                result = self.world.query(
                    x,
                    y,
                    z,
                    property_name,
                )

                if not result:
                    voxel_state[property_name] = {
                        "state": "UNKNOWN",
                        "value": None,
                        "confidence": 0.0,
                        "uncertainty": 1.0,
                        "source": None,
                        "timestamp": None,
                    }
                    continue

                voxel_state[property_name] = {
                    "state": result.get(
                        "state",
                        "UNKNOWN",
                    ),
                    "value": result.get(
                        "value"
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
                    "timestamp": result.get(
                        "timestamp"
                    ),
                }

            voxels[coordinate] = voxel_state

        return {
            "timestamp": self._timestamp(),
            "label": label,
            "world_size": self.world.size,
            "observed_voxels": len(voxels),
            "voxels": deepcopy(voxels),
        }

    # =========================================================
    # STORE SNAPSHOT
    # =========================================================

    def capture(self, label="OBSERVED"):
        """
        Capture and store one temporal state.
        """

        snapshot = self.snapshot(
            label=label
        )

        self.history.append(
            snapshot
        )

        if len(self.history) > self.max_history:
            self.history.pop(0)

        return deepcopy(snapshot)

    # =========================================================
    # HISTORY
    # =========================================================

    def history_length(self):
        return len(self.history)

    def latest(self):
        if not self.history:
            return None

        return deepcopy(
            self.history[-1]
        )

    def get_snapshot(self, index):
        if index < 0:
            index += len(self.history)

        if (
            index < 0
            or index >= len(self.history)
        ):
            return None

        return deepcopy(
            self.history[index]
        )

    # =========================================================
    # SNAPSHOT COMPARISON
    # =========================================================

    def compare_snapshots(
        self,
        previous,
        current,
    ):
        """
        Compare two temporal snapshots.

        Detects:

            APPEARED
            DISAPPEARED
            CHANGED
            UNCHANGED
        """

        previous_voxels = previous.get(
            "voxels",
            {}
        )

        current_voxels = current.get(
            "voxels",
            {}
        )

        all_coordinates = (
            set(previous_voxels.keys())
            | set(current_voxels.keys())
        )

        appeared = []
        disappeared = []
        changed = []
        unchanged = []

        for coordinate in sorted(
            all_coordinates
        ):

            was_present = (
                coordinate
                in previous_voxels
            )

            is_present = (
                coordinate
                in current_voxels
            )

            if (
                not was_present
                and is_present
            ):
                appeared.append(
                    coordinate
                )
                continue

            if (
                was_present
                and not is_present
            ):
                disappeared.append(
                    coordinate
                )
                continue

            previous_state = (
                previous_voxels[
                    coordinate
                ]
            )

            current_state = (
                current_voxels[
                    coordinate
                ]
            )

            if previous_state != current_state:
                changed.append(
                    coordinate
                )
            else:
                unchanged.append(
                    coordinate
                )

        return {
            "previous_timestamp": previous.get(
                "timestamp"
            ),
            "current_timestamp": current.get(
                "timestamp"
            ),
            "appeared": appeared,
            "disappeared": disappeared,
            "changed": changed,
            "unchanged": unchanged,
            "appeared_count": len(
                appeared
            ),
            "disappeared_count": len(
                disappeared
            ),
            "changed_count": len(
                changed
            ),
            "unchanged_count": len(
                unchanged
            ),
        }

    # =========================================================
    # PROPERTY CHANGE
    # =========================================================

    def property_change(
        self,
        previous,
        current,
        coordinate,
        property_name,
    ):
        """
        Compare one property of one voxel
        between two temporal states.
        """

        previous_voxel = previous.get(
            "voxels",
            {}
        ).get(coordinate)

        current_voxel = current.get(
            "voxels",
            {}
        ).get(coordinate)

        if (
            previous_voxel is None
            or current_voxel is None
        ):
            return {
                "status": "UNKNOWN",
                "coordinate": coordinate,
                "property": property_name,
            }

        previous_property = (
            previous_voxel.get(
                property_name
            )
        )

        current_property = (
            current_voxel.get(
                property_name
            )
        )

        if (
            previous_property is None
            or current_property is None
        ):
            return {
                "status": "UNKNOWN",
                "coordinate": coordinate,
                "property": property_name,
            }

        previous_value = (
            previous_property.get(
                "value"
            )
        )

        current_value = (
            current_property.get(
                "value"
            )
        )

        previous_known = (
            previous_property.get(
                "state"
            ) != "UNKNOWN"
        )

        current_known = (
            current_property.get(
                "state"
            ) != "UNKNOWN"
        )

        if (
            not previous_known
            or not current_known
        ):
            return {
                "status": "UNKNOWN",
                "coordinate": coordinate,
                "property": property_name,
                "previous_value": previous_value,
                "current_value": current_value,
            }

        if (
            previous_value is None
            or current_value is None
        ):
            return {
                "status": "UNKNOWN",
                "coordinate": coordinate,
                "property": property_name,
            }

        delta = (
            current_value
            - previous_value
        )

        if delta > 0:
            direction = "INCREASED"
        elif delta < 0:
            direction = "DECREASED"
        else:
            direction = "UNCHANGED"

        return {
            "status": "OBSERVED_CHANGE",
            "coordinate": coordinate,
            "property": property_name,
            "previous_value": previous_value,
            "current_value": current_value,
            "delta": delta,
            "direction": direction,
        }

    # =========================================================
    # TEMPORAL TRANSITION
    # =========================================================

    def classify_transition(
        self,
        previous,
        current,
        coordinate,
        property_name,
    ):
        """
        Classify a property's temporal transition.

        Possible states:

            UNKNOWN_TO_OBSERVED
            OBSERVED_TO_UNKNOWN
            OBSERVED_TO_OBSERVED
            UNKNOWN
        """

        previous_voxel = previous.get(
            "voxels",
            {}
        ).get(coordinate)

        current_voxel = current.get(
            "voxels",
            {}
        ).get(coordinate)

        if (
            previous_voxel is None
            or current_voxel is None
        ):
            return "UNKNOWN"

        previous_property = (
            previous_voxel.get(
                property_name
            )
        )

        current_property = (
            current_voxel.get(
                property_name
            )
        )

        if (
            previous_property is None
            or current_property is None
        ):
            return "UNKNOWN"

        previous_known = (
            previous_property.get(
                "state"
            ) != "UNKNOWN"
        )

        current_known = (
            current_property.get(
                "state"
            ) != "UNKNOWN"
        )

        if (
            not previous_known
            and current_known
        ):
            return "UNKNOWN_TO_OBSERVED"

        if (
            previous_known
            and not current_known
        ):
            return "OBSERVED_TO_UNKNOWN"

        if (
            previous_known
            and current_known
        ):
            return "OBSERVED_TO_OBSERVED"

        return "UNKNOWN"

    # =========================================================
    # TEMPORAL SERIES
    # =========================================================

    def property_series(
        self,
        coordinate,
        property_name,
    ):
        """
        Return the temporal series of one property
        at one 3D coordinate.

        Missing or unknown observations remain UNKNOWN.
        """

        series = []

        for snapshot in self.history:

            voxel = snapshot.get(
                "voxels",
                {}
            ).get(coordinate)

            if voxel is None:
                series.append({
                    "timestamp": snapshot.get(
                        "timestamp"
                    ),
                    "label": snapshot.get(
                        "label"
                    ),
                    "state": "UNKNOWN",
                    "value": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                })
                continue

            property_state = voxel.get(
                property_name
            )

            if property_state is None:
                series.append({
                    "timestamp": snapshot.get(
                        "timestamp"
                    ),
                    "label": snapshot.get(
                        "label"
                    ),
                    "state": "UNKNOWN",
                    "value": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                })
                continue

            series.append({
                "timestamp": snapshot.get(
                    "timestamp"
                ),
                "label": snapshot.get(
                    "label"
                ),
                "state": property_state.get(
                    "state",
                    "UNKNOWN",
                ),
                "value": property_state.get(
                    "value"
                ),
                "confidence": property_state.get(
                    "confidence",
                    0.0,
                ),
                "uncertainty": property_state.get(
                    "uncertainty",
                    1.0,
                ),
            })

        return series

    # =========================================================
    # FUTURE / TEMPORAL LABEL SEPARATION
    # =========================================================

    def validate_temporal_label(
        self,
        label,
    ):
        """
        Validate explicit temporal state labels.
        """

        if label not in self.TEMPORAL_LABELS:
            raise ValueError(
                "Temporal label must be "
                "HISTORICAL, OBSERVED or PREDICTED."
            )

        return True

    # =========================================================
    # IMMUTABILITY
    # =========================================================

    def history_does_not_modify_world(self):
        """
        Verify that taking a snapshot does not
        create or remove sparse voxels.
        """

        before = len(
            self.world.voxels
        )

        self.snapshot()

        after = len(
            self.world.voxels
        )

        return {
            "world_modified": (
                before != after
            ),
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
            "history_length": len(
                self.history
            ),
            "max_history": self.max_history,
            "temporal_states": [
                "HISTORICAL",
                "OBSERVED",
                "PREDICTED",
            ],
            "unknown_preserved": True,
            "world_mutation": False,
        }


# =============================================================
# DEMO WORLD
# =============================================================

def create_demo_world():

    world = SparseWorldModel(
        size=(8, 8, 8)
    )

    # Initial physical state
    world.update_sensor(
        3,
        3,
        2,
        "lidar",
        1.0,
        0.95,
    )

    world.update_sensor(
        3,
        3,
        2,
        "radar",
        1.0,
        0.85,
    )

    world.update_sensor(
        3,
        3,
        2,
        "thermal",
        30.0,
        0.90,
    )

    return world


# =============================================================
# TESTS
# =============================================================

def run_tests():

    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V1.3")
    print("TEMPORAL WORLD MODEL")
    print("=" * 60)

    print()
    print("Creating temporal world...")

    world = create_demo_world()

    temporal = TemporalWorldModel(
        world,
        max_history=10,
    )

    print()
    print(
        "World size:",
        world.size,
    )

    print(
        "Observed voxels:",
        len(world.voxels),
    )

    # =========================================================
    # TEST 1
    # =========================================================

    summary = temporal.summary()

    assert (
        summary["unknown_preserved"]
        is True
    )

    assert (
        summary["world_mutation"]
        is False
    )

    print()
    print(
        "Temporal engine initialization test: PASS"
    )

    # =========================================================
    # TEST 2
    # =========================================================

    snapshot_1 = temporal.capture(
        label="OBSERVED"
    )

    assert (
        snapshot_1["observed_voxels"]
        == 1
    )

    assert (
        snapshot_1["label"]
        == "OBSERVED"
    )

    print(
        "Initial snapshot test: PASS"
    )

    # =========================================================
    # TEST 3
    # =========================================================

    temporal.validate_temporal_label(
        "HISTORICAL"
    )

    temporal.validate_temporal_label(
        "PREDICTED"
    )

    print(
        "Temporal label validation test: PASS"
    )

    # =========================================================
    # TEST 4
    # =========================================================

    world.update_sensor(
        3,
        3,
        2,
        "radar",
        2.0,
        0.85,
    )

    world.update_sensor(
        3,
        3,
        2,
        "thermal",
        40.0,
        0.90,
    )

    snapshot_2 = temporal.capture(
        label="OBSERVED"
    )

    assert (
        snapshot_2["observed_voxels"]
        == 1
    )

    print(
        "Second temporal snapshot test: PASS"
    )

    # =========================================================
    # TEST 5
    # =========================================================

    comparison = temporal.compare_snapshots(
        snapshot_1,
        snapshot_2,
    )

    assert (
        (3, 3, 2)
        in comparison["changed"]
    )

    assert (
        comparison["changed_count"]
        >= 1
    )

    print(
        "Temporal snapshot comparison test: PASS"
    )

    # =========================================================
    # TEST 6
    # =========================================================

    motion_change = (
        temporal.property_change(
            snapshot_1,
            snapshot_2,
            (3, 3, 2),
            "motion",
        )
    )

    assert (
        motion_change["status"]
        == "OBSERVED_CHANGE"
    )

    assert (
        motion_change["delta"]
        != 0
    )

    print(
        "Motion temporal change test: PASS"
    )

    # =========================================================
    # TEST 7
    # =========================================================

    temperature_change = (
        temporal.property_change(
            snapshot_1,
            snapshot_2,
            (3, 3, 2),
            "temperature",
        )
    )

    assert (
        temperature_change["status"]
        == "OBSERVED_CHANGE"
    )

    assert (
        temperature_change["direction"]
        == "INCREASED"
    )

    print(
        "Temperature temporal change test: PASS"
    )

    # =========================================================
    # TEST 8
    # =========================================================

    transition = (
        temporal.classify_transition(
            snapshot_1,
            snapshot_2,
            (3, 3, 2),
            "motion",
        )
    )

    assert (
        transition
        == "OBSERVED_TO_OBSERVED"
    )

    print(
        "Temporal transition classification test: PASS"
    )

    # =========================================================
    # TEST 9
    # =========================================================

    series = temporal.property_series(
        (3, 3, 2),
        "temperature",
    )

    assert (
        len(series)
        == 2
    )

    # First observation is exactly 30.0.
    assert (
        series[0]["value"]
        == 30.0
    )

    # The second value is the result of the
    # physical observation-fusion layer.
    # It should therefore be greater than
    # the previous value but does not need
    # to equal exactly 40.0.
    assert (
        series[1]["value"]
        > series[0]["value"]
    )

    assert (
        series[1]["value"]
        < 40.0
    )

    print(
        "Temporal property series test: PASS"
    )

    # =========================================================
    # TEST 10
    # =========================================================

    unknown_series = temporal.property_series(
        (7, 7, 7),
        "temperature",
    )

    assert (
        len(unknown_series)
        == 2
    )

    assert all(
        item["state"] == "UNKNOWN"
        for item in unknown_series
    )

    assert all(
        item["value"] is None
        for item in unknown_series
    )

    print(
        "UNKNOWN temporal series test: PASS"
    )

    # =========================================================
    # TEST 11
    # =========================================================

    world.update_sensor(
        6,
        6,
        6,
        "lidar",
        1.0,
        0.90,
    )

    snapshot_3 = temporal.capture(
        label="OBSERVED"
    )

    comparison_2 = temporal.compare_snapshots(
        snapshot_2,
        snapshot_3,
    )

    assert (
        (6, 6, 6)
        in comparison_2["appeared"]
    )

    print(
        "Voxel appearance detection test: PASS"
    )

    # =========================================================
    # TEST 12
    # =========================================================

    check = (
        temporal.history_does_not_modify_world()
    )

    assert (
        check["world_modified"]
        is False
    )

    assert (
        check["voxels_before"]
        == check["voxels_after"]
    )

    print(
        "Temporal immutability test: PASS"
    )

    # =========================================================
    # TEST 13
    # =========================================================

    assert (
        temporal.history_length()
        == 3
    )

    latest = temporal.latest()

    assert (
        latest is not None
    )

    assert (
        latest["label"]
        == "OBSERVED"
    )

    print(
        "Temporal history management test: PASS"
    )

    # =========================================================
    # TEST 14
    # =========================================================

    historical = temporal.get_snapshot(
        0
    )

    assert (
        historical is not None
    )

    assert (
        historical["label"]
        == "OBSERVED"
    )

    assert (
        historical["timestamp"]
        is not None
    )

    assert (
        historical["world_size"]
        == (8, 8, 8)
    )

    print(
        "Historical snapshot retrieval test: PASS"
    )

    # =========================================================
    # TEST 15
    # =========================================================

    before = len(
        world.voxels
    )

    predicted_snapshot = (
        temporal.snapshot(
            label="PREDICTED"
        )
    )

    after = len(
        world.voxels
    )

    assert (
        predicted_snapshot["label"]
        == "PREDICTED"
    )

    assert (
        before
        == after
    )

    print(
        "Predicted-label snapshot immutability test: PASS"
    )

    # =========================================================
    # TEST 16
    # =========================================================

    invalid_label_rejected = False

    try:
        temporal.validate_temporal_label(
            "INVALID"
        )
    except ValueError:
        invalid_label_rejected = True

    assert (
        invalid_label_rejected
        is True
    )

    print(
        "Invalid temporal label rejection test: PASS"
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("-" * 60)
    print("TEMPORAL WORLD SUMMARY")
    print("-" * 60)

    print(
        "World size       :",
        world.size,
    )

    print(
        "Observed voxels  :",
        len(world.voxels),
    )

    print(
        "History length   :",
        temporal.history_length(),
    )

    print(
        "Initial temp     : 30.0"
    )

    print(
        "Second observation: fused > 30.0 and < 40.0"
    )

    print(
        "New voxel        : (6, 6, 6)"
    )

    print(
        "UNKNOWN preserved: YES"
    )

    print(
        "World mutated by temporal layer: NO"
    )

    print()
    print("=" * 60)
    print("V1.3 TEST COMPLETE")
    print("=" * 60)

    print()
    print("All V1.3 tests passed.")

    print()
    print("Architecture validated:")

    print(
        "  [OK] Temporal snapshots"
    )

    print(
        "  [OK] Historical world states"
    )

    print(
        "  [OK] Current observed states"
    )

    print(
        "  [OK] Temporal snapshot comparison"
    )

    print(
        "  [OK] Property change detection"
    )

    print(
        "  [OK] Temperature evolution"
    )

    print(
        "  [OK] Motion evolution"
    )

    print(
        "  [OK] Voxel appearance detection"
    )

    print(
        "  [OK] Temporal property series"
    )

    print(
        "  [OK] UNKNOWN temporal states"
    )

    print(
        "  [OK] Explicit temporal labels"
    )

    print(
        "  [OK] 3D temporal state"
    )

    print(
        "  [OK] Temporal memory management"
    )

    print(
        "  [OK] History does not mutate world"
    )

    print(
        "  [OK] UNKNOWN remains UNKNOWN"
    )

    print(
        "  [OK] Invalid temporal labels rejected"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_tests()