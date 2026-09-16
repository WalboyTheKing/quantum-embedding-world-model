
import matplotlib.pyplot as plt

from world_model import SparseWorldModel


# ============================================================
# QUANTUM-EMBEDDING WORLD MODEL
# 3D SPARSE WORLD VISUALIZER V1.3
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


def get_property_state(world, x, y, z, property_name):
    """
    Retourne l'état d'une propriété d'un voxel observé.

    IMPORTANT :
    Cette fonction ne crée jamais de nouveau voxel.
    """

    voxel = world.get(x, y, z)

    if voxel is None:
        return None

    return world.query(
        x,
        y,
        z,
        property_name,
    )


def get_dominant_property(world, x, y, z):
    """
    Détermine une propriété utile à afficher pour un voxel.

    Priorité :
        temperature
        motion
        pressure
        matter
        light
        energy
        field
        time
    """

    priority = [
        "temperature",
        "motion",
        "pressure",
        "matter",
        "light",
        "energy",
        "field",
        "time",
    ]

    for property_name in priority:

        result = get_property_state(
            world,
            x,
            y,
            z,
            property_name,
        )

        if result is None:
            continue

        if result["state"] != "UNKNOWN":

            return property_name, result

    return None, None


def build_world_visualization(world):

    fig = plt.figure(
        figsize=(12, 10)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # --------------------------------------------------------
    # UNKNOWN WORLD
    # --------------------------------------------------------

    unknown_x = []
    unknown_y = []
    unknown_z = []

    max_x = world.size[0] - 1
    max_y = world.size[1] - 1
    max_z = world.size[2] - 1

    for x in range(world.size[0]):
        for y in range(world.size[1]):
            for z in range(world.size[2]):

                voxel = world.get(
                    x,
                    y,
                    z,
                )

                if voxel is None:

                    unknown_x.append(x)
                    unknown_y.append(y)
                    unknown_z.append(z)

    # UNKNOWN is visualized as a very faint background
    ax.scatter(
        unknown_x,
        unknown_y,
        unknown_z,
        s=12,
        alpha=0.035,
        marker="s",
    )

    # --------------------------------------------------------
    # OBSERVED VOXELS
    # --------------------------------------------------------

    observed_x = []
    observed_y = []
    observed_z = []

    for (x, y, z) in world.voxels.keys():

        observed_x.append(x)
        observed_y.append(y)
        observed_z.append(z)

    if observed_x:

        ax.scatter(
            observed_x,
            observed_y,
            observed_z,
            s=280,
            alpha=0.85,
            marker="s",
            edgecolors="black",
            linewidths=0.8,
        )

    # --------------------------------------------------------
    # PROPERTY LABELS
    # --------------------------------------------------------

    for (x, y, z) in world.voxels.keys():

        property_name, result = get_dominant_property(
            world,
            x,
            y,
            z,
        )

        if result is None:
            continue

        value = result["value"]
        confidence = result["confidence"]

        if property_name == "temperature":

            label = (
                f"T={value:.1f}°"
                f"\nC={confidence:.2f}"
            )

        elif property_name == "motion":

            label = (
                f"M={value:.2f}"
                f"\nC={confidence:.2f}"
            )

        elif property_name == "pressure":

            label = (
                f"P={value:.2f}"
                f"\nC={confidence:.2f}"
            )

        elif property_name == "matter":

            label = (
                f"MAT={value:.2f}"
                f"\nC={confidence:.2f}"
            )

        elif property_name == "light":

            label = (
                f"L={value:.2f}"
                f"\nC={confidence:.2f}"
            )

        else:

            label = (
                f"{property_name}"
                f"\n{value:.2f}"
            )

        ax.text(
            x,
            y,
            z + 0.35,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    # --------------------------------------------------------
    # WORLD FRAME
    # --------------------------------------------------------

    corners = [
        (0, 0, 0),
        (max_x, 0, 0),
        (0, max_y, 0),
        (max_x, max_y, 0),
        (0, 0, max_z),
        (max_x, 0, max_z),
        (0, max_y, max_z),
        (max_x, max_y, max_z),
    ]

    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    for start, end in edges:

        x1, y1, z1 = corners[start]
        x2, y2, z2 = corners[end]

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            linewidth=1,
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlabel(
        "X — largeur"
    )

    ax.set_ylabel(
        "Y — profondeur"
    )

    ax.set_zlabel(
        "Z — hauteur"
    )

    ax.set_xlim(
        -0.5,
        max_x + 0.5,
    )

    ax.set_ylim(
        -0.5,
        max_y + 0.5,
    )

    ax.set_zlim(
        -0.5,
        max_z + 0.5,
    )

    ax.set_xticks(
        range(world.size[0])
    )

    ax.set_yticks(
        range(world.size[1])
    )

    ax.set_zticks(
        range(world.size[2])
    )

    ax.set_box_aspect(
        (
            world.size[0],
            world.size[1],
            world.size[2],
        )
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    total_voxels = (
        world.size[0]
        * world.size[1]
        * world.size[2]
    )

    observed_voxels = len(
        world.voxels
    )

    unknown_voxels = (
        total_voxels
        - observed_voxels
    )

    ax.set_title(
        "QUANTUM-EMBEDDING WORLD MODEL V1.3\n"
        f"3D Sparse World — "
        f"Observed: {observed_voxels}/{total_voxels} — "
        f"Unknown: {unknown_voxels}"
    )

    plt.tight_layout()

    return fig, ax


# ============================================================
# DEMO WORLD
# ============================================================

def create_demo_world():

    world = SparseWorldModel(
        size=(8, 8, 8),
        dim=16,
    )

    print()
    print("----------------------------------------------")
    print("SENSOR SIMULATION")
    print("----------------------------------------------")

    # --------------------------------------------------------
    # CENTRAL OBJECT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # HOT ZONE
    # --------------------------------------------------------

    world.update_sensor(
        6,
        4,
        2,
        "thermal",
        value=42.0,
        confidence=0.90,
    )

    # --------------------------------------------------------
    # SECOND OBJECT
    # --------------------------------------------------------

    world.update_sensor(
        2,
        2,
        4,
        "lidar",
        value=1.0,
        confidence=0.90,
    )

    world.update_sensor(
        2,
        2,
        4,
        "radar",
        value=0.6,
        confidence=0.80,
    )

    return world


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("QUANTUM-EMBEDDING WORLD MODEL V1.3")
    print("3D SPARSE WORLD VISUALIZATION")
    print("=" * 60)

    print()
    print("Creating Sparse World Model...")

    world = create_demo_world()

    print()
    print("----------------------------------------------")
    print("WORLD INFORMATION")
    print("----------------------------------------------")

    print(
        f"World size      : {world.size}"
    )

    total = (
        world.size[0]
        * world.size[1]
        * world.size[2]
    )

    observed = len(
        world.voxels
    )

    unknown = (
        total
        - observed
    )

    print(
        f"Virtual voxels  : {total}"
    )

    print(
        f"Observed voxels : {observed}"
    )

    print(
        f"Unknown voxels  : {unknown}"
    )

    print(
        f"Storage ratio   : "
        f"{observed / total * 100:.2f}%"
    )

    print()
    print("Launching 3D visualization...")

    build_world_visualization(
        world
    )

    plt.show()
