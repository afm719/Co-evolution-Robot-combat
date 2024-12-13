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
                return min(left_value + right_value, 5)  # Limita el valor sumado a 5
            elif self.operation == 'subtract':
                return max(left_value - right_value, -5)  # Limita el valor restado a -5
            elif self.operation == 'multiply':
                return min(left_value * right_value, 5)  # Limita el valor multiplicado a 5
            elif self.operation == 'divide':
                # Evita divisiones por cero, y limita el valor a 5
                return min(left_value / right_value if right_value != 0 else 2, 5)
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

# Clase Robot
class Robot:
    def __init__(self, position, health, enemy_position):
        self.position = position  # Example: (10, 20)
        self.health = health
        self.enemy_position = enemy_position  # Example: (30, 40)
        self.tree = None  # Expression tree
        self.fitness = 0  # Initialize fitness to 0

    def normalize_fitness(self, value, min_value=0, max_value=100):
        first_selection = max(value, min_value)
        second_selection = min(first_selection, max_value)
        return second_selection
    
    def evaluate_fitness(self):
        if self.tree is not None:
            raw_fitness = self.tree.evaluate(self)
            normalized_fitness = self.normalize_fitness(raw_fitness)
            self.fitness = normalized_fitness
            return self.fitness
        self.fitness = 0
        return self.fitness

# Función de suavizado de fitness (promedio móvil)
def smooth_fitness(fitness_list, smoothing_factor=0.05):
    if len(fitness_list) < 2:
        return fitness_list[-1] if fitness_list else 0
    return fitness_list[-1] * smoothing_factor + fitness_list[-2] * (1 - smoothing_factor)

# Función para generar un árbol de operaciones aleatorio
def generate_random_tree():
    operations = ['add', 'subtract', 'multiply', 'distance']
    value_choices = ['position', 'health', 'enemy_position']
    
    # Árbol simple con nodos de operación
    if random.random() < 0.5:  # Crear un nodo hoja
        value = random.choice(value_choices)
        return Node(value=value)
    else:  # Crear un nodo de operación
        operation = random.choice(operations)
        left = generate_random_tree()
        right = generate_random_tree()
        return Node(operation=operation, left=left, right=right)
    
# Función de cruce
def crossover(parent1, parent2):
    if random.random() > 0.5:
        return parent1
    else:
        return parent2

# Función de mutación
def mutate(tree):
    if random.random() < 0.05:  # Reducción de la tasa de mutación
        return generate_random_tree()  # Reemplazar por un nuevo árbol
    return tree

# Función de selección por torneo
def tournament_selection(population, tournament_size=3):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda robot: robot.fitness, reverse=True)
    return selected[0]  # El mejor del torneo

def genetic_algorithm():
    population_size = 10
    generations = 100
    mutation_rate = 0.05  # Tasa de mutación más baja para un cambio gradual
    crossover_rate = 0.7  # Tasa de cruce adecuada
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
        print(f"Generación {generation}:")
        
        generation_best_fitness = 0  # El mejor fitness de la generación actual
        
        for robot in population:
            robot_fitness = robot.evaluate_fitness()
            # Aplicar suavizado del fitness para evitar fluctuaciones grandes
            if robot_fitness > generation_best_fitness:
                generation_best_fitness = robot_fitness
                if generation_best_fitness > best_fitness:
                    best_fitness = generation_best_fitness
        
        # Suavizar la evolución del fitness
        best_fitness_current.append(smooth_fitness(best_fitness_current))
        
        print(f"Generación {generation}: Mejor Fitness = {generation_best_fitness}")
        
        # Selección de los mejores robots para la próxima generación mediante torneo
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < crossover_rate:
                child_tree = crossover(parent1.tree, parent2.tree)
            else:
                child_tree = parent1.tree

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

    # Graficar el progreso del mejor fitness
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
