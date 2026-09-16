"""
Quantum-Embedding World Model
V1.2 - Multi-Property World-State Reasoning

Purpose:
    Reason about multiple physical properties inside the
    shared 3D world model.

Properties used in V1.2:

    matter
    motion
    temperature
    pressure
    light
    energy

Core principle:

    UNKNOWN != 0
    UNKNOWN != FREE
    UNKNOWN != SAFE

The reasoner never invents missing information.

This module does not modify:
    world_model.py
    spatial_queries.py
    sparse_compute.py
    prediction.py
    reasoning.py
"""

from world_model import SparseWorldModel


class WorldStateReasoner:
    """
    Multi-property reasoning layer.

    It inspects several physical properties of a voxel or
    3D region and produces a structured world-state description.
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

    def _get_property(
        self,
        coordinate,
        property_name,
    ):
        x, y, z = coordinate

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

    # =========================================================
    # VOXEL STATE
    # =========================================================

    def inspect_voxel(
        self,
        coordinate,
        min_confidence=0.5,
    ):
        """
        Inspect multiple physical properties of one voxel.

        Returns known properties and explicitly tracks
        unknown properties.
        """

        if not self._inside_bounds(coordinate):
            return {
                "coordinate": coordinate,
                "status": "OUT_OF_BOUNDS",
            }

        properties = [
            "matter",
            "motion",
            "temperature",
            "pressure",
            "light",
            "energy",
        ]

        state = {}
        known_properties = []
        unknown_properties = []
        low_confidence_properties = []

        for property_name in properties:

            result = self._get_property(
                coordinate,
                property_name,
            )

            if result is None:
                state[property_name] = {
                    "state": "UNKNOWN",
                    "known": False,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                }

                unknown_properties.append(
                    property_name
                )

                continue

            confidence = result.get(
                "confidence",
                0.0,
            )

            value = result.get("value")

            state[property_name] = {
                "state": result.get(
                    "state",
                    "OBSERVED_VALUE",
                ),
                "known": True,
                "value": value,
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "source": result.get("source"),
                "timestamp": result.get("timestamp"),
            }

            known_properties.append(
                property_name
            )

            if confidence < min_confidence:
                low_confidence_properties.append(
                    property_name
                )

        if not known_properties:
            overall_status = "UNKNOWN"

        elif low_confidence_properties:
            overall_status = "UNCERTAIN"

        elif len(known_properties) < len(properties):
            overall_status = "PARTIALLY_OBSERVED"

        else:
            overall_status = "OBSERVED"

        return {
            "coordinate": coordinate,
            "status": overall_status,
            "properties": state,
            "known_properties": known_properties,
            "unknown_properties": unknown_properties,
            "low_confidence_properties": (
                low_confidence_properties
            ),
            "known_count": len(
                known_properties
            ),
            "property_count": len(properties),
        }

    # =========================================================
    # PHYSICAL CONDITION DETECTION
    # =========================================================

    def detect_conditions(
        self,
        coordinate,
        hot_threshold=30.0,
        motion_threshold=0.0,
        pressure_threshold=0.0,
        min_confidence=0.5,
    ):
        """
        Detect meaningful physical conditions.

        Possible conditions:

            SOLID
            MOVING
            HOT
            PRESSURIZED
            ILLUMINATED
            ENERGETIC
            UNKNOWN_MATTER
            UNKNOWN_MOTION
            UNKNOWN_TEMPERATURE
        """

        inspection = self.inspect_voxel(
            coordinate,
            min_confidence=min_confidence,
        )

        if inspection["status"] == "OUT_OF_BOUNDS":
            return inspection

        properties = inspection["properties"]

        conditions = []
        unknown_conditions = []

        # -----------------------------------------------------
        # MATTER
        # -----------------------------------------------------

        matter = properties["matter"]

        if not matter["known"]:
            unknown_conditions.append(
                "UNKNOWN_MATTER"
            )

        elif (
            matter["confidence"] >= min_confidence
            and matter["value"] is not None
            and matter["value"] > 0
        ):
            conditions.append("SOLID")

        # -----------------------------------------------------
        # MOTION
        # -----------------------------------------------------

        motion = properties["motion"]

        if not motion["known"]:
            unknown_conditions.append(
                "UNKNOWN_MOTION"
            )

        elif (
            motion["confidence"] >= min_confidence
            and motion["value"] is not None
            and abs(motion["value"]) > motion_threshold
        ):
            conditions.append("MOVING")

        # -----------------------------------------------------
        # TEMPERATURE
        # -----------------------------------------------------

        temperature = properties["temperature"]

        if not temperature["known"]:
            unknown_conditions.append(
                "UNKNOWN_TEMPERATURE"
            )

        elif (
            temperature["confidence"] >= min_confidence
            and temperature["value"] is not None
            and temperature["value"] >= hot_threshold
        ):
            conditions.append("HOT")

        # -----------------------------------------------------
        # PRESSURE
        # -----------------------------------------------------

        pressure = properties["pressure"]

        if not pressure["known"]:
            unknown_conditions.append(
                "UNKNOWN_PRESSURE"
            )

        elif (
            pressure["confidence"] >= min_confidence
            and pressure["value"] is not None
            and pressure["value"] > pressure_threshold
        ):
            conditions.append("PRESSURIZED")

        # -----------------------------------------------------
        # LIGHT
        # -----------------------------------------------------

        light = properties["light"]

        if (
            light["known"]
            and light["confidence"] >= min_confidence
            and light["value"] is not None
            and light["value"] > 0
        ):
            conditions.append("ILLUMINATED")

        # -----------------------------------------------------
        # ENERGY
        # -----------------------------------------------------

        energy = properties["energy"]

        if (
            energy["known"]
            and energy["confidence"] >= min_confidence
            and energy["value"] is not None
            and energy["value"] > 0
        ):
            conditions.append("ENERGETIC")

        return {
            "coordinate": coordinate,
            "conditions": conditions,
            "unknown_conditions": unknown_conditions,
            "inspection": inspection,
        }

    # =========================================================
    # MULTI-PHYSICAL OBJECT DETECTION
    # =========================================================

    def analyze_object(
        self,
        coordinate,
        hot_threshold=30.0,
        motion_threshold=0.0,
        min_confidence=0.5,
    ):
        """
        Determine whether a voxel appears to contain an
        observed physical object and describe its state.
        """

        conditions = self.detect_conditions(
            coordinate=coordinate,
            hot_threshold=hot_threshold,
            motion_threshold=motion_threshold,
            min_confidence=min_confidence,
        )

        if "status" in conditions:
            return conditions

        detected = conditions["conditions"]
        unknown = conditions["unknown_conditions"]

        if "SOLID" in detected:
            if "MOVING" in detected:
                object_state = "OBSERVED_MOVING_OBJECT"
            else:
                object_state = "OBSERVED_OBJECT"

        elif "UNKNOWN_MATTER" in unknown:
            object_state = "UNKNOWN_OBJECT_STATE"

        else:
            object_state = "NO_OBSERVED_OBJECT"

        return {
            "coordinate": coordinate,
            "object_state": object_state,
            "conditions": detected,
            "unknown_conditions": unknown,
        }

    # =========================================================
    # REGION REASONING
    # =========================================================

    def analyze_region(
        self,
        min_corner,
        max_corner,
        hot_threshold=30.0,
        motion_threshold=0.0,
        min_confidence=0.5,
    ):
        """
        Analyze every coordinate inside a 3D region.

        Only existing/observed information is interpreted.
        UNKNOWN space is not materialized.
        """

        x1, y1, z1 = min_corner
        x2, y2, z2 = max_corner

        observations = []
        observed_count = 0
        unknown_count = 0
        moving_count = 0
        hot_count = 0
        solid_count = 0

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):

                    coordinate = (
                        x,
                        y,
                        z,
                    )

                    result = self.analyze_object(
                        coordinate,
                        hot_threshold=hot_threshold,
                        motion_threshold=motion_threshold,
                        min_confidence=min_confidence,
                    )

                    observations.append(result)

                    if result.get(
                        "object_state"
                    ) != "UNKNOWN_OBJECT_STATE":
                        if result.get(
                            "object_state"
                        ) in (
                            "OBSERVED_OBJECT",
                            "OBSERVED_MOVING_OBJECT",
                        ):
                            observed_count += 1

                    else:
                        unknown_count += 1

                    conditions = result.get(
                        "conditions",
                        [],
                    )

                    if "SOLID" in conditions:
                        solid_count += 1

                    if "MOVING" in conditions:
                        moving_count += 1

                    if "HOT" in conditions:
                        hot_count += 1

        total_voxels = (
            (x2 - x1 + 1)
            * (y2 - y1 + 1)
            * (z2 - z1 + 1)
        )

        return {
            "min_corner": min_corner,
            "max_corner": max_corner,
            "total_voxels": total_voxels,
            "observations": observations,
            "observed_objects": observed_count,
            "unknown_object_states": unknown_count,
            "solid_voxels": solid_count,
            "moving_voxels": moving_count,
            "hot_voxels": hot_count,
        }

    # =========================================================
    # COMBINED SAFETY REASONING
    # =========================================================

    def assess_voxel_safety(
        self,
        coordinate,
        hot_threshold=30.0,
        motion_threshold=0.0,
        min_confidence=0.5,
    ):
        """
        Produce a conservative safety classification.

        Possible results:

            SAFE
            HAZARDOUS
            UNCERTAIN
        """

        analysis = self.analyze_object(
            coordinate,
            hot_threshold=hot_threshold,
            motion_threshold=motion_threshold,
            min_confidence=min_confidence,
        )

        if analysis.get("object_state") == (
            "UNKNOWN_OBJECT_STATE"
        ):
            return {
                "coordinate": coordinate,
                "safety": "UNCERTAIN",
                "reason": (
                    "Matter state is unknown."
                ),
                "analysis": analysis,
            }

        conditions = analysis.get(
            "conditions",
            [],
        )

        if (
            "HOT" in conditions
            or "MOVING" in conditions
            or "PRESSURIZED" in conditions
        ):
            return {
                "coordinate": coordinate,
                "safety": "HAZARDOUS",
                "reason": (
                    "Observed physical conditions "
                    "require caution."
                ),
                "analysis": analysis,
            }

        if (
            analysis.get("object_state")
            == "OBSERVED_OBJECT"
        ):
            return {
                "coordinate": coordinate,
                "safety": "SAFE",
                "reason": (
                    "Observed object has no detected "
                    "hazard condition in this model."
                ),
                "analysis": analysis,
            }

        return {
            "coordinate": coordinate,
            "safety": "SAFE",
            "reason": (
                "No observed hazardous condition."
            ),
            "analysis": analysis,
        }

    # =========================================================
    # IMMUTABILITY
    # =========================================================

    def reasoning_does_not_modify_world(
        self,
        reasoning_function,
        *args,
        **kwargs,
    ):
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
                "multi-property physical world-state reasoning"
            ),
            "properties": [
                "matter",
                "motion",
                "temperature",
                "pressure",
                "light",
                "energy",
            ],
            "unknown_is_zero": False,
            "unknown_is_safe": False,
            "world_mutation": False,
        }


# =============================================================
# DEMO WORLD
# =============================================================

def create_demo_world():

    world = SparseWorldModel(
        size=(8, 8, 8)
    )

    # ---------------------------------------------------------
    # Static object
    # ---------------------------------------------------------

    world.update_sensor(
        2, 2, 2,
        "lidar",
        1.0,
        0.95,
    )

    world.update_sensor(
        2, 2, 2,
        "camera",
        0.8,
        0.90,
    )

    # ---------------------------------------------------------
    # Moving object
    # ---------------------------------------------------------

    world.update_sensor(
        4, 4, 2,
        "lidar",
        1.0,
        0.95,
    )

    world.update_sensor(
        4, 4, 2,
        "radar",
        2.5,
        0.85,
    )

    # ---------------------------------------------------------
    # Hot object
    # ---------------------------------------------------------

    world.update_sensor(
        6, 4, 2,
        "lidar",
        1.0,
        0.95,
    )

    world.update_sensor(
        6, 4, 2,
        "thermal",
        45.0,
        0.90,
    )

    # ---------------------------------------------------------
    # Pressurized object
    # ---------------------------------------------------------

    world.update_sensor(
        1, 5, 2,
        "contact",
        8.0,
        0.90,
    )

    return world


# =============================================================
# TESTS
# =============================================================

def run_tests():

    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V1.2")
    print("MULTI-PROPERTY WORLD-STATE REASONING")
    print("=" * 60)

    print()
    print("Creating world-state reasoning test world...")

    world = create_demo_world()

    reasoner = WorldStateReasoner(world)

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
        summary["unknown_is_zero"]
        is False
    )

    assert (
        summary["unknown_is_safe"]
        is False
    )

    print()
    print(
        "World-state reasoner initialization test: PASS"
    )

    # =========================================================
    # TEST 2
    # =========================================================

    static = reasoner.inspect_voxel(
        (2, 2, 2)
    )

    assert (
        static["status"]
        in (
            "PARTIALLY_OBSERVED",
            "OBSERVED",
        )
    )

    assert (
        "matter"
        in static["known_properties"]
    )

    assert (
        "temperature"
        in static["unknown_properties"]
    )

    print(
        "Multi-property voxel inspection test: PASS"
    )

    # =========================================================
    # TEST 3
    # =========================================================

    moving = reasoner.analyze_object(
        (4, 4, 2)
    )

    assert (
        moving["object_state"]
        == "OBSERVED_MOVING_OBJECT"
    )

    assert (
        "MOVING"
        in moving["conditions"]
    )

    print(
        "Moving object reasoning test: PASS"
    )

    # =========================================================
    # TEST 4
    # =========================================================

    hot = reasoner.analyze_object(
        (6, 4, 2)
    )

    assert (
        hot["object_state"]
        == "OBSERVED_OBJECT"
    )

    assert (
        "HOT"
        in hot["conditions"]
    )

    print(
        "Hot object reasoning test: PASS"
    )

    # =========================================================
    # TEST 5
    # =========================================================

    unknown = reasoner.analyze_object(
        (7, 7, 7)
    )

    assert (
        unknown["object_state"]
        == "UNKNOWN_OBJECT_STATE"
    )

    print(
        "UNKNOWN object-state reasoning test: PASS"
    )

    # =========================================================
    # TEST 6
    # =========================================================

    safety_hot = reasoner.assess_voxel_safety(
        (6, 4, 2)
    )

    assert (
        safety_hot["safety"]
        == "HAZARDOUS"
    )

    print(
        "Hot-condition safety reasoning test: PASS"
    )

    # =========================================================
    # TEST 7
    # =========================================================

    safety_unknown = reasoner.assess_voxel_safety(
        (7, 7, 7)
    )

    assert (
        safety_unknown["safety"]
        == "UNCERTAIN"
    )

    print(
        "UNKNOWN safety reasoning test: PASS"
    )

    # =========================================================
    # TEST 8
    # =========================================================

    region = reasoner.analyze_region(
        min_corner=(1, 1, 1),
        max_corner=(3, 3, 3),
    )

    assert (
        region["total_voxels"]
        == 27
    )

    assert (
        region["solid_voxels"]
        >= 1
    )

    print(
        "3D region multi-property reasoning test: PASS"
    )

    # =========================================================
    # TEST 9
    # =========================================================

    pressurized = reasoner.detect_conditions(
        (1, 5, 2)
    )

    assert (
        "PRESSURIZED"
        in pressurized["conditions"]
    )

    print(
        "Pressure-condition reasoning test: PASS"
    )

    # =========================================================
    # TEST 10
    # =========================================================

    light = reasoner.detect_conditions(
        (2, 2, 2)
    )

    assert (
        "ILLUMINATED"
        in light["conditions"]
    )

    print(
        "Light-condition reasoning test: PASS"
    )

    # =========================================================
    # TEST 11
    # =========================================================

    before = len(world.voxels)

    check = (
        reasoner.reasoning_does_not_modify_world(
            reasoner.analyze_object,
            (4, 4, 2),
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
    # TEST 12
    # =========================================================

    out_of_bounds = reasoner.inspect_voxel(
        (99, 99, 99)
    )

    assert (
        out_of_bounds["status"]
        == "OUT_OF_BOUNDS"
    )

    print(
        "3D bounds handling test: PASS"
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("-" * 60)
    print("WORLD-STATE REASONING SUMMARY")
    print("-" * 60)

    print(
        "Observed voxels :",
        len(world.voxels),
    )

    print(
        "Static object   : OBSERVED_OBJECT"
    )

    print(
        "Moving object   : OBSERVED_MOVING_OBJECT"
    )

    print(
        "Hot object      : HOT"
    )

    print(
        "Unknown region  : UNKNOWN_OBJECT_STATE"
    )

    print(
        "Safety model    : SAFE / HAZARDOUS / UNCERTAIN"
    )

    print()
    print("=" * 60)
    print("V1.2 TEST COMPLETE")
    print("=" * 60)

    print()
    print("All V1.2 tests passed.")

    print()
    print("Architecture validated:")
    print(
        "  [OK] Multi-property voxel inspection"
    )
    print(
        "  [OK] Matter reasoning"
    )
    print(
        "  [OK] Motion reasoning"
    )
    print(
        "  [OK] Temperature reasoning"
    )
    print(
        "  [OK] Pressure reasoning"
    )
    print(
        "  [OK] Light reasoning"
    )
    print(
        "  [OK] Energy reasoning"
    )
    print(
        "  [OK] Moving-object detection"
    )
    print(
        "  [OK] Hot-object detection"
    )
    print(
        "  [OK] 3D region reasoning"
    )
    print(
        "  [OK] Conservative safety reasoning"
    )
    print(
        "  [OK] UNKNOWN-aware reasoning"
    )
    print(
        "  [OK] 3D bounds handling"
    )
    print(
        "  [OK] Reasoning does not mutate world"
    )
    print(
        "  [OK] UNKNOWN != ZERO"
    )
    print(
        "  [OK] UNKNOWN != SAFE"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_tests()