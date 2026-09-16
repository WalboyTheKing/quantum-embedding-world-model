"""
QUANTUM-EMBEDDING WORLD MODEL
V0.7 - MULTI-SENSOR FUSION

Shared sparse 3D world.

Sensors:
    CAMERA  -> matter + light
    LIDAR   -> matter
    RADAR   -> motion
    THERMAL -> temperature
    CONTACT -> matter + pressure

Core principles:
    - shared spatial world
    - sparse voxel storage
    - UNKNOWN != 0
    - per-property confidence
    - per-property uncertainty
    - 16D embedding
    - multi-sensor observations in the same voxel
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Tuple

from world_model import SparseWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

WORLD_SIZE = (8, 8, 8)

PROPERTY_NAMES = [
    "matter",
    "light",
    "temperature",
    "motion",
    "pressure",
    "energy",
    "field",
    "time",
]


# ============================================================
# SENSOR OBSERVATION
# ============================================================

@dataclass
class SensorObservation:
    sensor_type: str
    x: int
    y: int
    z: int
    property_name: str
    value: float
    confidence: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.sensor_type,
            "voxel": (self.x, self.y, self.z),
            "property": self.property_name,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


# ============================================================
# MULTI SENSOR FUSION
# ============================================================

class MultiSensorFusion:

    def __init__(
        self,
        world_size: Tuple[int, int, int] = WORLD_SIZE,
    ):
        self.world = SparseWorldModel(
            size=world_size
        )

        self.observations: List[
            SensorObservation
        ] = []

        print(
            "Shared multi-sensor world initialized."
        )
        print(
            f"World size: {world_size}"
        )
        print(
            f"Virtual voxel count: "
            f"{self.virtual_voxel_count()}"
        )

    # ========================================================
    # WORLD STATISTICS
    # ========================================================

    def virtual_voxel_count(self) -> int:
        x, y, z = self.world.size

        return x * y * z

    def observed_voxel_count(self) -> int:
        """
        world_model.py exposes the sparse voxel dictionary
        through `voxels`.
        """

        return len(self.world.voxels)

    def unknown_voxel_count(self) -> int:
        return (
            self.virtual_voxel_count()
            - self.observed_voxel_count()
        )

    def storage_ratio(self) -> float:
        total = self.virtual_voxel_count()

        if total == 0:
            return 0.0

        return (
            self.observed_voxel_count()
            / total
            * 100.0
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_coordinates(
        self,
        x: int,
        y: int,
        z: int,
    ) -> None:

        sx, sy, sz = self.world.size

        if not (
            0 <= x < sx
            and
            0 <= y < sy
            and
            0 <= z < sz
        ):
            raise ValueError(
                f"Voxel ({x},{y},{z}) "
                f"is outside world {self.world.size}"
            )

    def validate_confidence(
        self,
        confidence: float,
    ) -> None:

        if not (
            0.0 < confidence <= 1.0
        ):
            raise ValueError(
                "Confidence must be "
                "greater than 0 and <= 1."
            )

    # ========================================================
    # GENERIC SENSOR OBSERVATION
    # ========================================================

    def observe(
        self,
        sensor_type: str,
        x: int,
        y: int,
        z: int,
        property_name: str,
        value: float,
        confidence: float,
    ) -> None:

        self.validate_coordinates(
            x,
            y,
            z,
        )

        self.validate_confidence(
            confidence
        )

        timestamp = (
            datetime.now(UTC).isoformat()
        )

        # IMPORTANT:
        #
        # Actual world_model.py signature:
        #
        # update_sensor(
        #     self,
        #     x,
        #     y,
        #     z,
        #     sensor_type,
        #     value,
        #     confidence=0.9
        # )
        #
        self.world.update_sensor(
            x,
            y,
            z,
            sensor_type,
            value,
            confidence,
        )

        self.observations.append(
            SensorObservation(
                sensor_type=sensor_type,
                x=x,
                y=y,
                z=z,
                property_name=property_name,
                value=value,
                confidence=confidence,
                timestamp=timestamp,
            )
        )

        print(
            f"[FUSION] "
            f"{sensor_type.upper():8s} "
            f"voxel=({x},{y},{z}) "
            f"{property_name}={value} "
            f"confidence={confidence:.2f}"
        )

    # ========================================================
    # CAMERA
    # ========================================================

    def camera(
        self,
        x: int,
        y: int,
        z: int,
        light: float,
        confidence: float = 0.95,
    ) -> None:

        self.observe(
            sensor_type="camera",
            x=x,
            y=y,
            z=z,
            property_name="light",
            value=light,
            confidence=confidence,
        )

    # ========================================================
    # THERMAL
    # ========================================================

    def thermal(
        self,
        x: int,
        y: int,
        z: int,
        temperature: float,
        confidence: float = 0.85,
    ) -> None:

        self.observe(
            sensor_type="thermal",
            x=x,
            y=y,
            z=z,
            property_name="temperature",
            value=temperature,
            confidence=confidence,
        )

    # ========================================================
    # RADAR
    # ========================================================

    def radar(
        self,
        x: int,
        y: int,
        z: int,
        motion: float,
        confidence: float = 0.70,
    ) -> None:

        self.observe(
            sensor_type="radar",
            x=x,
            y=y,
            z=z,
            property_name="motion",
            value=motion,
            confidence=confidence,
        )

    # ========================================================
    # CONTACT
    # ========================================================

    def contact(
        self,
        x: int,
        y: int,
        z: int,
        pressure: float,
        confidence: float = 0.95,
    ) -> None:

        self.observe(
            sensor_type="contact",
            x=x,
            y=y,
            z=z,
            property_name="pressure",
            value=pressure,
            confidence=confidence,
        )

    # ========================================================
    # LIDAR
    # ========================================================

    def lidar(
        self,
        x: int,
        y: int,
        z: int,
        matter: float = 1.0,
        confidence: float = 0.95,
    ) -> None:

        self.observe(
            sensor_type="lidar",
            x=x,
            y=y,
            z=z,
            property_name="matter",
            value=matter,
            confidence=confidence,
        )

    # ========================================================
    # QUERY
    # ========================================================

    def query_property(
        self,
        x: int,
        y: int,
        z: int,
        property_name: str,
    ) -> Dict[str, Any]:

        return self.world.query(
            x,
            y,
            z,
            property_name,
        )

    # ========================================================
    # VOXEL
    # ========================================================

    def get_voxel(
        self,
        x: int,
        y: int,
        z: int,
    ):

        return self.world.get(
            x,
            y,
            z,
        )

    # ========================================================
    # PROPERTY HELPERS
    # ========================================================

    @staticmethod
    def property_is_known(
        result: Any,
    ) -> bool:

        if not isinstance(
            result,
            dict,
        ):
            return False

        state = result.get(
            "state"
        )

        if state in (
            "OBSERVED_VALUE",
            "OBSERVED_ZERO",
        ):
            return True

        return (
            result.get("value")
            is not None
            and
            result.get(
                "confidence",
                0.0,
            ) > 0.0
        )

    @staticmethod
    def property_value(
        result: Any,
    ):

        if not isinstance(
            result,
            dict,
        ):
            return None

        return result.get(
            "value"
        )

    @staticmethod
    def property_confidence(
        result: Any,
    ) -> float:

        if not isinstance(
            result,
            dict,
        ):
            return 0.0

        return float(
            result.get(
                "confidence",
                0.0,
            )
        )

    # ========================================================
    # ANALYZE VOXEL
    # ========================================================

    def analyze_voxel(
        self,
        x: int,
        y: int,
        z: int,
    ) -> Dict[str, Any]:

        self.validate_coordinates(
            x,
            y,
            z,
        )

        properties = {}

        for property_name in PROPERTY_NAMES:

            properties[
                property_name
            ] = self.query_property(
                x,
                y,
                z,
                property_name,
            )

        known_count = sum(
            1
            for result in properties.values()
            if self.property_is_known(
                result
            )
        )

        return {
            "voxel": (x, y, z),
            "known_properties": known_count,
            "total_properties": len(
                PROPERTY_NAMES
            ),
            "properties": properties,
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    def fusion_summary(
        self,
    ) -> Dict[str, Any]:

        virtual_voxels = (
            self.virtual_voxel_count()
        )

        observed_voxels = (
            self.observed_voxel_count()
        )

        return {
            "virtual_voxels":
                virtual_voxels,

            "observed_voxels":
                observed_voxels,

            "unknown_voxels":
                virtual_voxels
                - observed_voxels,

            "observations":
                len(
                    self.observations
                ),

            "storage_ratio":
                self.storage_ratio(),
        }

    # ========================================================
    # HISTORY
    # ========================================================

    def print_observation_history(
        self,
    ) -> None:

        print()
        print(
            "--- SENSOR OBSERVATION HISTORY ---"
        )

        for index, observation in enumerate(
            self.observations,
            start=1,
        ):

            print(
                f"{index:02d}. "
                f"{observation.sensor_type.upper():8s} "
                f"voxel="
                f"({observation.x},"
                f"{observation.y},"
                f"{observation.z}) "
                f"{observation.property_name}="
                f"{observation.value} "
                f"confidence="
                f"{observation.confidence:.2f}"
            )


# ============================================================
# DEMO WORLD
# ============================================================

def create_demo_world():

    print()
    print(
        "Creating shared multi-sensor world..."
    )
    print()

    fusion = MultiSensorFusion(
        world_size=WORLD_SIZE
    )

    # ========================================================
    # CENTRAL OBJECT
    # ========================================================

    fusion.camera(
        4,
        4,
        2,
        light=0.8,
        confidence=0.95,
    )

    fusion.radar(
        4,
        4,
        2,
        motion=1.2,
        confidence=0.70,
    )

    fusion.contact(
        4,
        4,
        2,
        pressure=1.0,
        confidence=0.95,
    )

    fusion.lidar(
        4,
        4,
        2,
        matter=1.0,
        confidence=0.98,
    )

    # ========================================================
    # HOT AREA
    # ========================================================

    fusion.thermal(
        6,
        4,
        2,
        temperature=42.0,
        confidence=0.85,
    )

    # ========================================================
    # SECOND OBJECT
    # ========================================================

    fusion.camera(
        2,
        5,
        1,
        light=0.45,
        confidence=0.90,
    )

    fusion.lidar(
        2,
        5,
        1,
        matter=1.0,
        confidence=0.92,
    )

    # ========================================================
    # MOVING OBJECT
    # ========================================================

    fusion.radar(
        5,
        2,
        1,
        motion=2.4,
        confidence=0.80,
    )

    return fusion


# ============================================================
# TEST 1 - STATISTICS
# ============================================================

def test_statistics(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- WORLD STATISTICS ---"
    )

    summary = (
        fusion.fusion_summary()
    )

    print(
        f"Virtual voxels : "
        f"{summary['virtual_voxels']}"
    )

    print(
        f"Observed voxels: "
        f"{summary['observed_voxels']}"
    )

    print(
        f"Unknown voxels : "
        f"{summary['unknown_voxels']}"
    )

    print(
        f"Observations   : "
        f"{summary['observations']}"
    )

    print(
        f"Storage ratio  : "
        f"{summary['storage_ratio']:.2f}%"
    )

    assert (
        summary["virtual_voxels"]
        == 512
    )

    assert (
        summary["observed_voxels"]
        == 4
    )

    assert (
        summary["unknown_voxels"]
        == 508
    )

    assert (
        summary["observations"]
        == 8
    )

    print(
        "Statistics test: PASS"
    )


# ============================================================
# TEST 2 - SHARED VOXEL
# ============================================================

def test_shared_voxel(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- SHARED VOXEL TEST ---"
    )

    x, y, z = 4, 4, 2

    central = fusion.get_voxel(
        x,
        y,
        z,
    )

    assert central is not None

    print()
    print(
        "Central voxel inspection:"
    )
    print()

    # inspect() prints directly.
    # It returns None in the current
    # world_model.py.

    central.inspect(
        x,
        y,
        z,
    )

    # ========================================================
    # QUERY MATTER
    # ========================================================

    matter = fusion.query_property(
        x,
        y,
        z,
        "matter",
    )

    assert isinstance(
        matter,
        dict,
    )

    assert fusion.property_is_known(
        matter
    )

    assert (
        matter["value"]
        is not None
    )

    assert abs(
        float(
            matter["value"]
        ) - 1.0
    ) < 0.000001

    # ========================================================
    # QUERY LIGHT
    # ========================================================

    light = fusion.query_property(
        x,
        y,
        z,
        "light",
    )

    assert isinstance(
        light,
        dict,
    )

    assert fusion.property_is_known(
        light
    )

    assert abs(
        float(
            light["value"]
        ) - 0.8
    ) < 0.000001

    # ========================================================
    # QUERY TEMPERATURE
    # ========================================================

    temperature = fusion.query_property(
        x,
        y,
        z,
        "temperature",
    )

    assert isinstance(
        temperature,
        dict,
    )

    assert not fusion.property_is_known(
        temperature
    )

    assert (
        temperature["value"]
        is None
    )

    assert (
        temperature["confidence"]
        == 0.0
    )

    # ========================================================
    # QUERY MOTION
    # ========================================================

    motion = fusion.query_property(
        x,
        y,
        z,
        "motion",
    )

    assert isinstance(
        motion,
        dict,
    )

    assert fusion.property_is_known(
        motion
    )

    assert abs(
        float(
            motion["value"]
        ) - 1.2
    ) < 0.000001

    # ========================================================
    # QUERY PRESSURE
    # ========================================================

    pressure = fusion.query_property(
        x,
        y,
        z,
        "pressure",
    )

    assert isinstance(
        pressure,
        dict,
    )

    assert fusion.property_is_known(
        pressure
    )

    assert abs(
        float(
            pressure["value"]
        ) - 1.0
    ) < 0.000001

    print()
    print(
        "Shared voxel test: PASS"
    )


# ============================================================
# TEST 3 - UNKNOWN != ZERO
# ============================================================

def test_unknown_is_not_zero(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- UNKNOWN != ZERO TEST ---"
    )

    result = fusion.query_property(
        0,
        0,
        0,
        "temperature",
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["value"]
        is None
    )

    assert (
        result["confidence"]
        == 0.0
    )

    assert (
        result["uncertainty"]
        == 1.0
    )

    assert (
        result["state"]
        == "UNKNOWN"
    )

    print(
        "Unobserved temperature: UNKNOWN"
    )

    print(
        "UNKNOWN != 0: PASS"
    )


# ============================================================
# TEST 4 - OBSERVED ZERO
# ============================================================

def test_observed_zero(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- OBSERVED ZERO TEST ---"
    )

    fusion.thermal(
        1,
        1,
        1,
        temperature=0.0,
        confidence=0.90,
    )

    result = fusion.query_property(
        1,
        1,
        1,
        "temperature",
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["value"]
        == 0.0
    )

    assert (
        result["confidence"]
        > 0.0
    )

    assert (
        result["state"]
        == "OBSERVED_ZERO"
    )

    assert (
        result["uncertainty"]
        < 1.0
    )

    print(
        "Temperature value: 0.0"
    )

    print(
        "Temperature state: OBSERVED_ZERO"
    )

    print(
        "Observed zero test: PASS"
    )


# ============================================================
# TEST 5 - OBSERVATION MASK
# ============================================================

def test_observation_mask(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- OBSERVATION MASK TEST ---"
    )

    central = fusion.get_voxel(
        4,
        4,
        2,
    )

    assert central is not None

    mask = (
        central.observation_mask()
    )

    expected = [
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
    ]

    print(
        f"Observation mask: "
        f"{mask.tolist()}"
    )

    assert (
        mask.tolist()
        == expected
    )

    print(
        "Observation mask test: PASS"
    )


# ============================================================
# TEST 6 - EMBEDDING
# ============================================================

def test_embedding(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- EMBEDDING TEST ---"
    )

    central = fusion.get_voxel(
        4,
        4,
        2,
    )

    assert central is not None

    state = (
        central.state_vector()
    )

    print(
        f"Embedding shape: "
        f"{state.shape}"
    )

    print(
        f"Embedding dimension: "
        f"{len(state)}"
    )

    assert len(state) == 16

    print(
        "16D embedding test: PASS"
    )


# ============================================================
# TEST 7 - THERMAL ZONE
# ============================================================

def test_temperature_zone(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- THERMAL SENSOR TEST ---"
    )

    result = fusion.query_property(
        6,
        4,
        2,
        "temperature",
    )

    assert isinstance(
        result,
        dict,
    )

    assert fusion.property_is_known(
        result
    )

    assert abs(
        float(
            result["value"]
        ) - 42.0
    ) < 0.000001

    assert (
        result["confidence"]
        > 0.0
    )

    print(
        f"Temperature: "
        f"{result['value']} °C"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.3f}"
    )

    print(
        "Thermal sensor test: PASS"
    )


# ============================================================
# TEST 8 - MULTI SENSOR SAME VOXEL
# ============================================================

def test_sensor_fusion_logic(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- MULTI-SENSOR FUSION TEST ---"
    )

    data = fusion.analyze_voxel(
        4,
        4,
        2,
    )

    print()
    print(
        f"Voxel: {data['voxel']}"
    )

    print(
        f"Known properties: "
        f"{data['known_properties']}/"
        f"{data['total_properties']}"
    )

    for property_name in PROPERTY_NAMES:

        result = data[
            "properties"
        ][property_name]

        if fusion.property_is_known(
            result
        ):

            print(
                f"  {property_name:12s} "
                f"KNOWN "
                f"value="
                f"{result.get('value')} "
                f"confidence="
                f"{result.get('confidence', 0):.3f}"
            )

        else:

            print(
                f"  {property_name:12s} "
                f"UNKNOWN"
            )

    assert (
        data["known_properties"]
        == 4
    )

    print()
    print(
        "Multi-sensor shared-state test: PASS"
    )


# ============================================================
# TEST 9 - SPARSITY
# ============================================================

def test_sparsity(
    fusion: MultiSensorFusion,
) -> None:

    print()
    print(
        "--- SPARSITY TEST ---"
    )

    virtual = (
        fusion.virtual_voxel_count()
    )

    observed = (
        fusion.observed_voxel_count()
    )

    ratio = (
        fusion.storage_ratio()
    )

    print(
        f"Virtual voxels : {virtual}"
    )

    print(
        f"Observed voxels: {observed}"
    )

    print(
        f"Storage ratio  : {ratio:.2f}%"
    )

    assert (
        observed < virtual
    )

    assert (
        ratio < 10.0
    )

    print(
        "Sparse representation test: PASS"
    )


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_tests():

    print("=" * 60)
    print(
        "QUANTUM-EMBEDDING WORLD MODEL V0.7"
    )
    print(
        "MULTI-SENSOR FUSION"
    )
    print("=" * 60)

    fusion = create_demo_world()

    # --------------------------------------------------------
    # TESTS
    # --------------------------------------------------------

    test_statistics(
        fusion
    )

    test_shared_voxel(
        fusion
    )

    test_unknown_is_not_zero(
        fusion
    )

    test_observation_mask(
        fusion
    )

    test_embedding(
        fusion
    )

    test_temperature_zone(
        fusion
    )

    test_sensor_fusion_logic(
        fusion
    )

    test_sparsity(
        fusion
    )

    # --------------------------------------------------------
    # OBSERVED ZERO
    # --------------------------------------------------------

    test_observed_zero(
        fusion
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    fusion.print_observation_history()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "V0.7 TEST COMPLETE"
    )
    print("=" * 60)

    summary = (
        fusion.fusion_summary()
    )

    print(
        f"Virtual voxels : "
        f"{summary['virtual_voxels']}"
    )

    print(
        f"Observed voxels: "
        f"{summary['observed_voxels']}"
    )

    print(
        f"Unknown voxels : "
        f"{summary['unknown_voxels']}"
    )

    print(
        f"Observations   : "
        f"{summary['observations']}"
    )

    print(
        f"Storage ratio  : "
        f"{summary['storage_ratio']:.2f}%"
    )

    print()
    print(
        "All V0.7 tests passed."
    )

    print()
    print(
        "Architecture validated:"
    )

    print(
        "  [OK] Shared sparse 3D world"
    )

    print(
        "  [OK] Camera -> light + matter"
    )

    print(
        "  [OK] LiDAR -> matter"
    )

    print(
        "  [OK] Radar -> motion"
    )

    print(
        "  [OK] Thermal -> temperature"
    )

    print(
        "  [OK] Contact -> matter + pressure"
    )

    print(
        "  [OK] Per-property confidence"
    )

    print(
        "  [OK] Per-property uncertainty"
    )

    print(
        "  [OK] UNKNOWN != OBSERVED_ZERO"
    )

    print(
        "  [OK] 16D embedding"
    )

    print(
        "  [OK] Observation mask"
    )

    print(
        "  [OK] Sparse storage"
    )

    print(
        "  [OK] Multi-sensor shared state"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_tests()