import numpy as np
from datetime import datetime, UTC


# ============================================================
# QUANTUM-EMBEDDING WORLD MODEL
# V0.5 + LIDAR SUPPORT
# ============================================================

PROPERTIES = [
    "matter",
    "light",
    "temperature",
    "motion",
    "pressure",
    "energy",
    "field",
    "time",
]


class PhysicalProperty:
    """
    Etat d'une propriété physique.

    UNKNOWN:
        value=None
        confidence=0

    OBSERVED_ZERO:
        value=0.0
        confidence>0

    OBSERVED_VALUE:
        value!=0
        confidence>0
    """

    def __init__(self):
        self.value = None
        self.confidence = 0.0
        self.source = None
        self.timestamp = None

    @property
    def is_known(self):
        return self.confidence > 0.0

    @property
    def is_unknown(self):
        return not self.is_known

    @property
    def is_observed_zero(self):
        return self.is_known and self.value == 0.0

    @property
    def uncertainty(self):
        return 1.0 - self.confidence

    @property
    def state_type(self):
        if self.is_unknown:
            return "UNKNOWN"

        if self.is_observed_zero:
            return "OBSERVED_ZERO"

        return "OBSERVED_VALUE"

    def update(self, value, confidence, source):

        if not 0.0 < confidence <= 1.0:
            raise ValueError(
                "confidence doit être > 0 et <= 1."
            )

        now = datetime.now(UTC).isoformat()

        # Première observation
        if self.is_unknown:

            self.value = float(value)
            self.confidence = float(confidence)
            self.source = source
            self.timestamp = now

            return

        # Fusion pondérée
        old_weight = self.confidence
        new_weight = confidence

        self.value = (
            self.value * old_weight
            + value * new_weight
        ) / (old_weight + new_weight)

        # Fusion des confiances
        self.confidence = 1.0 - (
            (1.0 - self.confidence)
            * (1.0 - confidence)
        )

        # Fusion des sources
        if self.source == source:
            pass

        elif self.source is None:
            self.source = source

        else:
            sources = set(self.source.split("+"))
            sources.add(source)
            self.source = "+".join(sorted(sources))

        self.timestamp = now


class QuantumEmbedding:
    """
    Etat complet d'un voxel.

    Representation :

        [values 8D | confidence 8D]

    Le known_mask permet de distinguer
    UNKNOWN d'une vraie valeur zéro.
    """

    def __init__(self, dim=16):

        if dim != 16:
            raise ValueError(
                "V0.5 utilise une dimension de 16."
            )

        self.dim = dim

        self.properties = {
            name: PhysicalProperty()
            for name in PROPERTIES
        }

        self.state = np.zeros(
            self.dim,
            dtype=np.float64,
        )

        self.known_mask = np.zeros(
            8,
            dtype=np.float64,
        )

        self.confidence_vector = np.zeros(
            8,
            dtype=np.float64,
        )

    def update(
        self,
        property_name,
        value,
        confidence,
        source,
    ):

        if property_name not in self.properties:
            raise ValueError(
                f"Propriété inconnue : {property_name}"
            )

        self.properties[property_name].update(
            value=value,
            confidence=confidence,
            source=source,
        )

        self._rebuild_embedding()

    def _rebuild_embedding(self):

        for index, name in enumerate(PROPERTIES):

            prop = self.properties[name]

            if prop.is_unknown:

                self.state[index] = 0.0
                self.known_mask[index] = 0.0
                self.confidence_vector[index] = 0.0

            else:

                self.state[index] = prop.value
                self.known_mask[index] = 1.0
                self.confidence_vector[index] = (
                    prop.confidence
                )

            self.state[index + 8] = (
                self.confidence_vector[index]
            )

    def get_property(self, property_name):

        if property_name not in self.properties:
            raise ValueError(
                f"Propriété inconnue : {property_name}"
            )

        prop = self.properties[property_name]

        return {
            "value": prop.value,
            "state": prop.state_type,
            "known": prop.is_known,
            "confidence": prop.confidence,
            "uncertainty": prop.uncertainty,
            "source": prop.source,
            "timestamp": prop.timestamp,
        }

    def known_properties(self):

        return [
            name
            for name, prop in self.properties.items()
            if prop.is_known
        ]

    def known_count(self):

        return len(self.known_properties())

    def average_uncertainty(self):

        return float(
            np.mean(
                [
                    prop.uncertainty
                    for prop in self.properties.values()
                ]
            )
        )

    def state_vector(self):

        return self.state.copy()

    def observation_mask(self):

        return self.known_mask.copy()

    def inspect(self, x, y, z):

        print()
        print("=" * 70)
        print(f"VOXEL ({x}, {y}, {z})")
        print("=" * 70)

        for name in PROPERTIES:

            prop = self.properties[name]

            if prop.is_unknown:

                print(
                    f"{name:12s} "
                    f"UNKNOWN "
                    f"confidence=0.000 "
                    f"uncertainty=1.000"
                )

            else:

                print(
                    f"{name:12s} "
                    f"value={prop.value:.3f} "
                    f"state={prop.state_type:14s} "
                    f"confidence={prop.confidence:.3f} "
                    f"uncertainty={prop.uncertainty:.3f}"
                )

        print("-" * 70)

        print(
            f"Known properties    : "
            f"{self.known_count()}/8"
        )

        print(
            f"Average uncertainty : "
            f"{self.average_uncertainty():.3f}"
        )

        print(
            f"Embedding dimension : "
            f"{self.dim}"
        )

        print(
            f"Observation mask    : "
            f"{self.known_mask.astype(int).tolist()}"
        )

        print(
            f"State vector        : "
            f"{np.round(self.state, 3).tolist()}"
        )

        print("=" * 70)


class SparseWorldModel:
    """
    Monde 3D sparse.

    Le volume virtuel existe entièrement,
    mais seuls les voxels observés sont stockés.
    """

    def __init__(
        self,
        size=(8, 8, 8),
        dim=16,
    ):

        self.size = size
        self.dim = dim
        self.voxels = {}

        total = (
            size[0]
            * size[1]
            * size[2]
        )

        print(
            f"Sparse World Model created: "
            f"{size[0]}x{size[1]}x{size[2]} "
            f"virtual space"
        )

        print(
            f"Virtual voxel count: {total}"
        )

    def _validate_coordinates(
        self,
        x,
        y,
        z,
    ):

        sx, sy, sz = self.size

        if not (
            0 <= x < sx
            and 0 <= y < sy
            and 0 <= z < sz
        ):

            raise ValueError(
                f"Coordinates ({x},{y},{z}) "
                f"outside world bounds {self.size}"
            )

    def get(self, x, y, z):

        self._validate_coordinates(
            x,
            y,
            z,
        )

        return self.voxels.get(
            (x, y, z)
        )

    def get_or_create(
        self,
        x,
        y,
        z,
    ):

        self._validate_coordinates(
            x,
            y,
            z,
        )

        key = (x, y, z)

        if key not in self.voxels:

            self.voxels[key] = (
                QuantumEmbedding(self.dim)
            )

        return self.voxels[key]

    def update_sensor(
        self,
        x,
        y,
        z,
        sensor_type,
        value,
        confidence=0.9,
    ):

        voxel = self.get_or_create(
            x,
            y,
            z,
        )

        if sensor_type == "camera":

            voxel.update(
                "matter",
                1.0,
                confidence,
                "camera",
            )

            voxel.update(
                "light",
                value,
                confidence,
                "camera",
            )

            print(
                f"[CAMERA] "
                f"voxel=({x},{y},{z}) "
                f"light={value}"
            )

        elif sensor_type == "thermal":

            voxel.update(
                "temperature",
                value,
                confidence,
                "thermal",
            )

            print(
                f"[THERMAL] "
                f"voxel=({x},{y},{z}) "
                f"temperature={value}"
            )

        elif sensor_type == "radar":

            voxel.update(
                "motion",
                value,
                confidence,
                "radar",
            )

            print(
                f"[RADAR] "
                f"voxel=({x},{y},{z}) "
                f"motion={value}"
            )

        elif sensor_type == "contact":

            voxel.update(
                "matter",
                1.0,
                confidence,
                "contact",
            )

            voxel.update(
                "pressure",
                value,
                confidence,
                "contact",
            )

            print(
                f"[CONTACT] "
                f"voxel=({x},{y},{z}) "
                f"pressure={value}"
            )

        elif sensor_type == "lidar":

            voxel.update(
                "matter",
                value,
                confidence,
                "lidar",
            )

            print(
                f"[LIDAR] "
                f"voxel=({x},{y},{z}) "
                f"matter={value}"
            )

        else:

            raise ValueError(
                f"Unknown sensor type: {sensor_type}"
            )

    def query(
        self,
        x,
        y,
        z,
        property_name,
    ):

        voxel = self.get(
            x,
            y,
            z,
        )

        if voxel is None:

            return {
                "value": None,
                "state": "UNKNOWN",
                "known": False,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "source": None,
                "timestamp": None,
                "voxel_observed": False,
            }

        result = voxel.get_property(
            property_name
        )

        result["voxel_observed"] = True

        return result

    def observed_voxel_count(self):

        return len(self.voxels)

    def total_voxel_count(self):

        x, y, z = self.size

        return x * y * z

    def unknown_voxel_count(self):

        return (
            self.total_voxel_count()
            - self.observed_voxel_count()
        )

    def sparsity_ratio(self):

        total = self.total_voxel_count()

        if total == 0:
            return 0.0

        return (
            self.observed_voxel_count()
            / total
        )

    def stats(self):

        print()
        print("=" * 60)
        print("SPARSE WORLD MODEL STATISTICS")
        print("=" * 60)

        print(
            f"Virtual voxels : "
            f"{self.total_voxel_count()}"
        )

        print(
            f"Observed voxels: "
            f"{self.observed_voxel_count()}"
        )

        print(
            f"Unknown voxels : "
            f"{self.unknown_voxel_count()}"
        )

        print(
            f"Storage ratio  : "
            f"{self.sparsity_ratio() * 100:.2f}%"
        )

        print("=" * 60)

    def print_slice(self, z=0):

        if not 0 <= z < self.size[2]:

            raise ValueError(
                f"z={z} outside world bounds"
            )

        print()
        print(
            f"=== Sparse Slice z={z} ==="
        )

        for y in range(self.size[1]):

            row = []

            for x in range(self.size[0]):

                voxel = self.get(
                    x,
                    y,
                    z,
                )

                if voxel is None:

                    row.append("·")

                elif voxel.known_count() > 0:

                    row.append("█")

                else:

                    row.append("·")

            print(" ".join(row))


# ============================================================
# V0.5 REGRESSION TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "     QUANTUM-EMBEDDING WORLD MODEL V0.5"
    )
    print(
        "       EXPLICIT STATE REPRESENTATION"
    )
    print("=" * 60)

    world = SparseWorldModel(
        size=(8, 8, 8),
        dim=16,
    )

    print()
    print("--- MULTI-SENSOR TEST ---")

    world.update_sensor(
        4,
        4,
        2,
        "camera",
        value=0.8,
        confidence=0.95,
    )

    world.update_sensor(
        4,
        4,
        2,
        "radar",
        value=1.2,
        confidence=0.70,
    )

    world.update_sensor(
        4,
        4,
        2,
        "contact",
        value=1.0,
        confidence=0.95,
    )

    world.update_sensor(
        6,
        4,
        2,
        "thermal",
        value=42.0,
        confidence=0.85,
    )

    world.print_slice(z=2)

    world.stats()

    print()
    print("--- QUERY TEST ---")

    central_temperature = world.query(
        4,
        4,
        2,
        "temperature",
    )

    central_matter = world.query(
        4,
        4,
        2,
        "matter",
    )

    unknown_temperature = world.query(
        0,
        0,
        0,
        "temperature",
    )

    print()
    print("Central temperature:")
    print(central_temperature)

    print()
    print("Central matter:")
    print(central_matter)

    print()
    print("Unknown voxel temperature:")
    print(unknown_temperature)

    central = world.get(
        4,
        4,
        2,
    )

    print()
    print("--- CENTRAL OBJECT ---")

    central.inspect(
        4,
        4,
        2,
    )

    print()
    print("--- UNKNOWN != ZERO TEST ---")

    zero_voxel = world.get_or_create(
        2,
        2,
        2,
    )

    zero_voxel.update(
        "temperature",
        0.0,
        0.90,
        "thermal",
    )

    observed_zero = world.query(
        2,
        2,
        2,
        "temperature",
    )

    unknown = world.query(
        0,
        0,
        0,
        "temperature",
    )

    print()
    print("Observed physical zero:")
    print(observed_zero)

    print()
    print("Unknown temperature:")
    print(unknown)

    assert (
        observed_zero["state"]
        == "OBSERVED_ZERO"
    )

    assert (
        observed_zero["known"]
        is True
    )

    assert (
        observed_zero["confidence"]
        > 0
    )

    assert (
        unknown["state"]
        == "UNKNOWN"
    )

    assert (
        unknown["known"]
        is False
    )

    print()
    print("--- OBSERVATION MASK TEST ---")

    mask = central.observation_mask()

    print("Properties:")
    print(PROPERTIES)

    print("Mask:")
    print(
        mask.astype(int).tolist()
    )

    expected_mask = [
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
    ]

    assert (
        mask.astype(int).tolist()
        == expected_mask
    )

    print(
        "PASS: observation mask is correct."
    )

    print()
    print("--- EMBEDDING TEST ---")

    vector = central.state_vector()

    print(
        f"Embedding shape: {vector.shape}"
    )

    print(
        f"Embedding: "
        f"{np.round(vector, 3).tolist()}"
    )

    assert vector.shape == (16,)

    assert vector[2] == 0.0
    assert vector[10] == 0.0

    assert vector[1] == 0.8
    assert vector[9] == 0.95

    print(
        "PASS: embedding contains values "
        "and confidence information."
    )

    print()
    print("=" * 60)
    print("V0.5 TEST COMPLETE")
    print("=" * 60)