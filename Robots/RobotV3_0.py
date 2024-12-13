import random
import matplotlib.pyplot as plt

# Definición de la clase Node para el árbol de expresión
class Node:
    def __init__(self, operation=None, left=None, right=None, value=None):
        self.operation = operation
        self.left = left
        self.right = right
        self.value = value

    def evaluate(self, robot):
        if self.operation:
            left_value = self.left.evaluate(robot) if self.left else 0
            right_value = self.right.evaluate(robot) if self.right else 0
            if self.operation == 'add':
                return left_value + right_value
            elif self.operation == 'subtract':
                return left_value - right_value
            elif self.operation == 'multiply':
                return left_value * right_value
            elif self.operation == 'divide':
                return left_value / right_value if right_value != 0 else 0
        else:
            if self.value == 'position':
                return robot.position[0]
            elif self.value == 'health':
                return robot.health
            elif self.value == 'enemy_position':
                return robot.enemy_position[0]
        return 0

# Clase Robot
class Robot:
    def __init__(self, position, health, enemy_position):
        self.position = position
        self.health = health
        self.enemy_position = enemy_position
        self.tree = None
        self.fitness = 0

    def normalize_fitness(self, value, min_value=0.5, max_value=1):
        return max(min((value - min_value) / (max_value - min_value), 1), 0)

    def evaluate_fitness(self):
        if self.tree:
            # Evaluamos el árbol para el robot
            raw_fitness = self.tree.evaluate(self)
            
            # Penalizaciones: Fitness inicial más bajo
            distance_to_goal = abs(self.position[0] - 0.5) + abs(self.position[1] - 0.5)  # Queremos que esté cerca de (0.5, 0.5)
            health_penalty = 1 - self.health  # Penaliza si la salud es baja
            distance_to_enemy = abs(self.enemy_position[0] - self.position[0]) + abs(self.enemy_position[1] - self.position[1])
            
            # Fitness total ponderado (valores altos son mejores)
            raw_fitness = 1 - (0.4 * distance_to_goal + 0.4 * health_penalty + 0.2 * distance_to_enemy)
            self.fitness = self.normalize_fitness(raw_fitness, 0, 1)
            return self.fitness
        self.fitness = 0
        return self.fitness
    
OPERATORS = ['+', '-', '*', '/']
# Generar un árbol de operaciones aleatorio
def generate_random_tree(depth=0, max_depth=5):
    # Establecemos un límite de profundidad para evitar la recursión infinita
    if depth > max_depth:
        return Node(value=random.randint(1, 10))  # Nodo hoja con un valor numérico
    
    # Elegir si el nodo será un operador o un valor numérico
    if random.random() < 0.5:
        # Nodo operador
        operator = random.choice(OPERATORS)
        left = generate_random_tree(depth + 1, max_depth)
        right = generate_random_tree(depth + 1, max_depth)
        return Node(value=operator, left=left, right=right)
    else:
        # Nodo hoja con valor numérico
        return Node(value=random.randint(1, 10))

# Aumentar la tasa de mutación
def mutate_tree_node(node):
    """Realiza una mutación en un nodo del árbol cambiando su operación o su valor."""
    if random.random() < 0.5:  # 50% de probabilidad de cambiar el nodo
        # Cambiar la operación (si tiene alguna)
        if node.operation:
            new_operation = random.choice(['add', 'subtract', 'multiply', 'divide'])
            return Node(operation=new_operation, left=node.left, right=node.right)
        else:
            # Cambiar el valor (si es una hoja, con 'position', 'health', o 'enemy_position')
            new_value = random.choice(['position', 'health', 'enemy_position'])
            return Node(value=new_value)
    else:
        # Cambiar el subárbol izquierdo o derecho de manera recursiva
        if node.left:
            node.left = mutate_tree_node(node.left)
        if node.right:
            node.right = mutate_tree_node(node.right)
        return node

# Inicialización de la población
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        robot = Robot(position=[random.uniform(0, 1), random.uniform(0, 1)], 
                      health=random.uniform(0, 1), 
                      enemy_position=[random.uniform(0, 1), random.uniform(0, 1)])
        robot.tree = generate_random_tree()  # Asigna un árbol aleatorio
        population.append(robot)
    return population

# Evaluar el fitness de toda la población
def evaluate_population_fitness(population):
    fitness_scores = [robot.evaluate_fitness() for robot in population]
    return fitness_scores

# Selección por torneo
def tournament_selection(population, fitness_scores, tournament_size=5):
    tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]

# Cruce de dos árboles
def crossover_trees(tree1, tree2):
    """Cruce de dos árboles de operaciones (nodos)."""
    # Esto es solo un ejemplo, puedes agregar tu propia lógica de cruce para árboles
    return tree1, tree2

def crossover(parent1, parent2):
    # Aplicamos el cruce a los árboles de los robots
    child1_tree, child2_tree = crossover_trees(parent1.tree, parent2.tree)
    
    # Crear robots hijos con los árboles cruzados
    child1 = Robot(position=parent1.position, health=parent1.health, enemy_position=parent1.enemy_position)
    child1.tree = child1_tree
    child2 = Robot(position=parent2.position, health=parent2.health, enemy_position=parent2.enemy_position)
    child2.tree = child2_tree

    return child1, child2

# Mutación de un robot
def mutate(robot):
    # Aplica la mutación a un árbol de operaciones
    robot.tree = mutate_tree_node(robot.tree)

# Diversificación de la población
def diversify_population(population, fitness_scores):
    diversified_population = []
    for i in range(len(population)):
        if fitness_scores[i] < 0.5 or random.random() < 0.8:  # Aumentamos el porcentaje de reemplazo y usamos fitness bajo
            diversified_population.append(generate_random_tree())  # Genera un nuevo árbol completamente aleatorio
        else:
            diversified_population.append(population[i])  # Mantén el individuo sin cambios
    return diversified_population

# Graficar la evolución del mejor fitness
def plot_fitness(fitness_history):
    plt.figure(figsize=(12, 6))
    plt.plot(fitness_history, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', 
             markersize=6, markerfacecolor='green', markeredgewidth=2)
    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generation", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')
    plt.gca().set_facecolor('#f0f0f0')  # Fondo gris claro
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12, frameon=False, title="Fitness Metrics", title_fontsize=13)
    plt.tight_layout()
    plt.show()

# Algoritmo genético
def genetic_algorithm():
    population = initialize_population()
    best_fitness_acumulado = -float("inf")  # Mejor fitness acumulado inicializado con el menor valor posible
    prev_best_fitness = -float("inf")  # Mejor fitness de la generación anterior
    stagnation_count = 0  # Contador de generaciones sin mejora
    fitness_history = []  # Lista para almacenar el mejor fitness por generación

    for generation in range(MAX_GENERATIONS):
        # Evaluar el fitness de toda la población
        fitness_scores = evaluate_population_fitness(population)

        # Encuentra el mejor fitness de esta generación
        best_fitness_generacion = max(fitness_scores)

        # Si el fitness de la generación es el mismo que el mejor hasta ahora, incrementa el contador de estancamiento
        if best_fitness_generacion == prev_best_fitness:
            stagnation_count += 1
        else:
            stagnation_count = 0  # Reinicia el contador si hubo mejora

        # Actualiza el mejor fitness acumulado si el fitness de esta generación es mejor
        if best_fitness_generacion > best_fitness_acumulado:
            best_fitness_acumulado = best_fitness_generacion

        # Imprime el progreso
        print(f"Generación {generation}: Mejor Fitness Acumulado = {best_fitness_acumulado:.3f} (Estancamiento: {stagnation_count})")

        # Termina si se alcanza el fitness óptimo
        if best_fitness_acumulado >= TARGET_FITNESS:
            print("¡Se alcanzó el fitness óptimo!")
            break
        # Almacena el mejor fitness de esta generación en la lista
        fitness_history.append(best_fitness_acumulado)

        # Si el algoritmo está estancado durante varias generaciones, modificamos algunos individuos
        if stagnation_count >= STAGNATION_THRESHOLD:
            print(f"Estancamiento detectado. Diversificando la población...")
            population = diversify_population(population, fitness_scores)  # Función que diversifica la población
            stagnation_count = 0  # Reinicia el contador de estancamiento

        # Selección, cruce y mutación para generar la nueva población
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        population = new_population
        prev_best_fitness = best_fitness_generacion  # Guardamos el fitness de la generación anterior

    print(f"Mejor fitness alcanzado: {best_fitness_acumulado:.3f}")

    # Al final, graficar la evolución del mejor fitness
    plot_fitness(fitness_history)

# Parámetros del algoritmo genético
POPULATION_SIZE = 300
GENE_LENGTH = 10
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.1
TARGET_FITNESS = 0.98  # El fitness óptimo
STAGNATION_THRESHOLD = 5  # Número de generaciones sin mejora para detectar estancamiento

# Ejecuta el algoritmo genético
genetic_algorithm()
