import numpy as np

class QuantumEmbedding:
    """
    Représentation inspirée des états quantiques pour un point de l'espace.
    UNKNOWN ≠ 0
    """
    def __init__(self, dim=16):
        self.dim = dim
        self.state = np.zeros(dim)              # Vecteur principal (embedding)
        self.confidence = np.zeros(dim)        # 0 = unknown, 1 = certain
        
        # Propriétés physiques lisibles (pour le debug)
        self.properties = {
            "material": 0.0,
            "light": 0.0,
            "temperature": 0.0,
            "motion": 0.0,
            "pressure": 0.0,
            "energy": 0.0,
            "field": 0.0
        }
        self.uncertainty = 1.0                 # 1.0 = complètement unknown

    def update(self, property_name, value, confidence=0.9):
        """Met à jour une propriété avec une certaine confiance"""
        if property_name in self.properties:
            # Fusion simple (plus tard on fera mieux)
            old_conf = self.confidence[0]  # simplifié pour l'instant
            self.properties[property_name] = value
            self.uncertainty = max(0.0, 1.0 - confidence)
            print(f"Updated {property_name} = {value} (confidence: {confidence})")

    def __repr__(self):
        return f"QE(uncertainty={self.uncertainty:.2f}, temp={self.properties['temperature']:.1f}, light={self.properties['light']:.1f})"


class WorldModel:
    def __init__(self, size=(8, 8, 8), dim=16):
        self.size = size
        self.dim = dim
        self.grid = np.empty(size, dtype=object)
        
        # Initialiser tous les voxels
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    self.grid[x, y, z] = QuantumEmbedding(dim)
        
        print(f"World Model created: {size[0]}x{size[1]}x{size[2]} voxels")

    def get(self, x, y, z):
        return self.grid[x, y, z]

    def update_sensor(self, x, y, z, sensor_type, value, confidence=0.9):
        """Simule un capteur qui écrit dans le monde"""
        voxel = self.get(x, y, z)
        
        if sensor_type == "camera":
            voxel.update("light", value, confidence)
            voxel.update("material", 1.0, confidence)  # occupancy
        elif sensor_type == "thermal":
            voxel.update("temperature", value, confidence)
        elif sensor_type == "radar":
            voxel.update("motion", value, confidence)
        else:
            print(f"Unknown sensor type: {sensor_type}")

    def print_slice(self, z=0):
        """Affiche une coupe 2D pour visualiser"""
        print(f"\n=== Slice z={z} ===")
        for y in range(self.size[1]):
            row = []
            for x in range(self.size[0]):
                v = self.get(x, y, z)
                if v.uncertainty < 0.5:
                    row.append("█")
                else:
                    row.append("·")
            print(" ".join(row))


# ====================== TEST ======================
if __name__ == "__main__":
    # Création du monde
    world = WorldModel(size=(8, 8, 8))

    print("\n--- Simulation de capteurs ---")
    
    # Caméra voit un objet au centre
    world.update_sensor(4, 4, 2, "camera", value=0.8, confidence=0.95)
    
    # Capteur thermique détecte de la chaleur
    world.update_sensor(5, 4, 2, "thermal", value=42.0, confidence=0.85)
    
    # Radar détecte un mouvement
    world.update_sensor(3, 4, 2, "radar", value=1.2, confidence=0.7)

    # Affichage
    world.print_slice(z=2)

    print("\nÉtat du voxel central (4,4,2):")
    print(world.get(4, 4, 2))
    
    print("\nÉtat d'un voxel non observé (0,0,0):")
    print(world.get(0, 0, 0))
