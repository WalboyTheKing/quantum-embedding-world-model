import matplotlib.pyplot as plt
from world_model import WorldModel


def visualize_world(world):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    known_x = []
    known_y = []
    known_z = []

    unknown_x = []
    unknown_y = []
    unknown_z = []

    # Parcourir tous les voxels
    for x in range(world.size[0]):
        for y in range(world.size[1]):
            for z in range(world.size[2]):

                voxel = world.get(x, y, z)

                if voxel.uncertainty < 1.0:
                    known_x.append(x)
                    known_y.append(y)
                    known_z.append(z)
                else:
                    unknown_x.append(x)
                    unknown_y.append(y)
                    unknown_z.append(z)

    # Monde inconnu
    ax.scatter(
        unknown_x,
        unknown_y,
        unknown_z,
        s=25,
        alpha=0.08,
        marker="s"
    )

    # Zones observées
    if known_x:
        ax.scatter(
            known_x,
            known_y,
            known_z,
            s=250,
            alpha=0.9,
            marker="s",
            edgecolors="black",
            linewidths=0.8
        )

    # Dimensions
    max_x = world.size[0] - 1
    max_y = world.size[1] - 1
    max_z = world.size[2] - 1

    # Cadre du monde
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
            linewidth=1
        )

    # Axes
    ax.set_xlabel("X — largeur")
    ax.set_ylabel("Y — profondeur")
    ax.set_zlabel("Z — hauteur")

    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(-0.5, max_y + 0.5)
    ax.set_zlim(-0.5, max_z + 0.5)

    ax.set_xticks(range(world.size[0]))
    ax.set_yticks(range(world.size[1]))
    ax.set_zticks(range(world.size[2]))

    ax.set_box_aspect(
        (
            world.size[0],
            world.size[1],
            world.size[2]
        )
    )

    total = (
        world.size[0]
        * world.size[1]
        * world.size[2]
    )

    known = len(known_x)

    ax.set_title(
        "QUANTUM-EMBEDDING WORLD MODEL V0.2\n"
        f"8×8×8 Voxels — Known: {known}/{total}"
    )

    plt.tight_layout()

    print()
    print("==============================================")
    print("       3D WORLD MODEL")
    print("==============================================")
    print(f"Total voxels : {total}")
    print(f"Known voxels : {known}")
    print(f"Unknown      : {total - known}")
    print()
    print("Opening 3D window...")

    # IMPORTANT
    plt.show()


if __name__ == "__main__":

    print("Creating World Model...")

    world = WorldModel(
        size=(8, 8, 8),
        dim=16
    )

    print()
    print("--- Sensor simulation ---")

    # Objet central
    world.update_sensor(
        4, 4, 2,
        "camera",
        value=0.8,
        confidence=0.95
    )

    world.update_sensor(
        4, 4, 2,
        "radar",
        value=1.2,
        confidence=0.70
    )

    world.update_sensor(
        4, 4, 2,
        "contact",
        value=1.0,
        confidence=0.95
    )

    # Zone chaude
    world.update_sensor(
        6, 4, 2,
        "thermal",
        value=42.0,
        confidence=0.90
    )

    print()
    print("Launching 3D World Model...")

    visualize_world(world)