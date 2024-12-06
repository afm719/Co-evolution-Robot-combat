import random
import math
import matplotlib.pyplot as plt

# Definición de la clase Node para el árbol de expresión
class Node:
    def __init__(self, operation=None, left=None, right=None, value=None):
        self.operation = operation  # Operación que este nodo representa (ej. '+', '-', 'distancia')
        self.left = left  # Hijo izquierdo
        self.right = right  # Hijo derecho
        self.value = value  # Si es un nodo hoja, tiene un valor (por ejemplo, posición o salud)

    def evaluate(self, robot):
        if self.operation:
            left_value = self.left.evaluate(robot) if self.left else 0
            right_value = self.right.evaluate(robot) if self.right else 0
            
            # Operaciones básicas con control
            if self.operation == 'add':
                return min(left_value + right_value, 2)  # Limita el valor sumado a 5
            elif self.operation == 'subtract':
                return max(left_value - right_value, -2)  # Limita el valor restado a -5
            elif self.operation == 'multiply':
                return min(left_value * right_value, 2)  # Limita el valor multiplicado a 5
            elif self.operation == 'divide':
                # Evita divisiones por cero, y limita el valor a 5
                return min(left_value / right_value if right_value != 0 else 2, 2)
            elif self.operation == 'distance':
                if isinstance(left_value, tuple) and isinstance(right_value, tuple):
                    return min(((left_value[0] - right_value[0])**2 + (left_value[1] - right_value[1])**2)**0.5, 5)  # Limita la distancia
                return 0
        else:
            # Devuelve valores según el nodo hoja
            if self.value == 'position':
                return robot.position[0]  # O ajusta según tu modelo
            elif self.value == 'health':
                return robot.health
            elif self.value == 'enemy_position':
                return robot.enemy_position[0]
        return 0

   
    def calculate_distance(self, position1, position2):
        if isinstance(position1, tuple) and len(position1) == 2 and \
        isinstance(position2, tuple) and len(position2) == 2:
            #print(f"Calculating distance between {position1} and {position2}")  # Debug
            return ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
        #print("Invalid positions for distance calculation.")  # Debug
        return 0

# Clase Robot
class Robot:
    def __init__(self, position, health, enemy_position):
        self.position = position  # Example: (10, 20)
        self.health = health
        self.enemy_position = enemy_position  # Example: (30, 40)
        self.tree = None  # Expression tree
        self.fitness = 0  # Initialize fitness to 0

    def normalize_fitness(self, value, min_value=0, max_value=100):
        #print(f"normalize_fitness called with: value={value}, min_value={min_value}, max_value={max_value}")
        #print(f"Types: value={type(value)}, min_value={type(min_value)}, max_value={type(max_value)}")
        first_selection = max(value, min_value)
        second_selection = min(first_selection, max_value)

        return second_selection


    
    def evaluate_fitness(self):
        if self.tree is not None:
            #print("entra aqui")
            raw_fitness = self.tree.evaluate(self)
            #print(raw_fitness)
            normalized_fitness = self.normalize_fitness(raw_fitness)
            #print(f"Fitness normalizado: {normalized_fitness}")
            #print(f"Evaluating fitness: {raw_fitness} -> Normalized: {normalized_fitness}")
            self.fitness = normalized_fitness
            #print("Retorna este fitness")
            return self.fitness
        self.fitness = 0
        return self.fitness

# Funciones del algoritmo genético
def generate_random_tree():
    operations = ['add', 'subtract', 'multiply', 'distance']
    value_choices = ['position', 'health', 'enemy_position']
    
    # Árbol simple con nodos de operación
    if random.random() < 0.5:  # Crear un nodo hoja
        value = random.choice(value_choices)
        #print(f"Leaf node created with value: {value}")  # Debug

        return Node(value=value)
    else:  # Crear un nodo de operación
        operation = random.choice(operations)
        left = generate_random_tree()
        right = generate_random_tree()
        #print(f"Operation node created with operation: {operation}")  # Debug
        return Node(operation=operation, left=left, right=right)
    
def crossover(parent1, parent2):
    # Realizar cruce entre dos árboles
    if random.random() > 0.5:
        return parent1
    else:
        return parent2

def mutate(tree):
    # Mutar el árbol
    if random.random() < 0.1:
        return generate_random_tree()  # Reemplazar por un nuevo árbol
    return tree

def genetic_algorithm():
    population_size = 10
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.07
    best_fitness = 0  # Inicializa el mejor fitness
    best_fitness_current = []
    
    # Inicializar la población de robots
    population = []
    for _ in range(population_size):
        position = (random.randint(0, 100), random.randint(0, 100))
        health = random.randint(1, 100)
        enemy_position = (random.randint(0, 100), random.randint(0, 100))
        robot = Robot(position=position, health=health, enemy_position=enemy_position)
        robot.tree = generate_random_tree()  # Asignar un árbol de operaciones aleatorio
        population.append(robot)
    
    # Evolución de la población
    for generation in range(generations):
        print(f"Generación {generation + 1}:")
        
        # Evaluar la aptitud de todos los robots
        generation_best_fitness = 0  # El mejor fitness de la generación actual
        
        for robot in population:
            robot_fitness = robot.evaluate_fitness()
            #print(f"Fitness del robot: {robot_fitness}")
            
            # Compara y actualiza el mejor fitness de la generación
            if robot_fitness > generation_best_fitness:
                generation_best_fitness = robot_fitness
                # También actualiza el mejor fitness global
                if generation_best_fitness > best_fitness:
                    best_fitness = generation_best_fitness
            best_fitness_current.append(best_fitness)
        
        # Mostrar el mejor fitness de la generación
        print(f"Generación {generation + 1}: Mejor Fitness = {generation_best_fitness}")
        
        # Selección de los mejores robots para la próxima generación
        population.sort(key=lambda robot: robot.fitness, reverse=True)
        best_robots = population[:int(population_size * 0.2)]  # Los mejores robots (20%)

        # Generar nueva población mediante cruce y mutación
        new_population = []
        while len(new_population) < population_size:
            parent1 = random.choice(best_robots)
            parent2 = random.choice(best_robots)

            if random.random() < crossover_rate:
                child_tree = crossover(parent1.tree, parent2.tree)
            else:
                child_tree = parent1.tree  # No cruza, elige uno de los padres

            # Aplicar mutación al árbol del hijo
            child_tree = mutate(child_tree)
            child_position = (random.randint(0, 100), random.randint(0, 100))
            child_health = random.randint(1, 100)
            child_enemy_position = (random.randint(0, 100), random.randint(0, 100))
            
            child_robot = Robot(position=child_position, health=child_health, enemy_position=child_enemy_position)
            child_robot.tree = child_tree
            new_population.append(child_robot)

        population = new_population  # Actualiza la población para la siguiente generación

        # Detener si alcanzamos un fitness suficientemente alto
        if best_fitness >= 99.99:
            print("¡Se alcanzó el fitness máximo!")
            break

    plt.figure(figsize=(12, 6))

    plt.plot(best_fitness_current, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', markersize=6, markerfacecolor='green', markeredgewidth=2)

    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generation", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')

    plt.gca().set_facecolor('#f0f0f0')  # Fondo gris claro
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)

    plt.legend(loc='upper left', fontsize=12, frameon=False, title="Fitness Metrics", title_fontsize=13)

    plt.tight_layout()  
    plt.show()

    print(f"Mejor fitness final: {best_fitness}")

# Llamar a la función para ejecutar el algoritmo genético
genetic_algorithm()
