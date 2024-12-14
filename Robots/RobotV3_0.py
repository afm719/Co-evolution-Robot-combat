import random
import matplotlib.pyplot as plt
import copy
import numpy as np

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

    def normalize_fitness(self, value, min_value=0, max_value=1):
        return max(min((value - min_value) / (max_value - min_value), 1), 0)

    def evaluate_fitness(self):
        if self.tree:
            raw_fitness = self.tree.evaluate(self)

            distance_to_goal = abs(self.position[0] - 0.5) + abs(self.position[1] - 0.5)
            health_penalty = max(0, 1 - self.health)
            distance_to_enemy = max(0, 1 - (abs(self.enemy_position[0] - self.position[0]) + abs(self.enemy_position[1] - self.position[1])) / 2)

            # Penalización por complejidad del árbol
            complexity_penalty = len(tree_to_string(self.tree)) / 100.0

            raw_fitness = 1 - (0.4 * distance_to_goal + 0.4 * health_penalty + 0.2 * distance_to_enemy + complexity_penalty)
            self.fitness = self.normalize_fitness(raw_fitness, 0, 1)
            return self.fitness
        self.fitness = 0
        return self.fitness


OPERATORS = ['add', 'subtract', 'multiply', 'divide']

def generate_random_tree(depth=0, max_depth=5):
    if depth > max_depth:
        return Node(value=random.choice(['position', 'health', 'enemy_position', random.randint(1, 10)]))
    
    if random.random() < 0.5:
        operator = random.choice(OPERATORS)
        left = generate_random_tree(depth + 1, max_depth)
        right = generate_random_tree(depth + 1, max_depth)
        return Node(operation=operator, left=left, right=right)
    else:
        return Node(value=random.choice(['position', 'health', 'enemy_position', random.randint(1, 10)]))

def mutate_tree_node(node):
    if random.random() < 0.5:
        if node.operation:
            new_operation = random.choice(OPERATORS)
            return Node(operation=new_operation, left=node.left, right=node.right)
        else:
            new_value = random.choice(['position', 'health', 'enemy_position', random.randint(1, 10)])
            return Node(value=new_value)
    else:
        if random.random() < 0.3:
            return generate_random_tree(max_depth=random.randint(2, 4))
        if node.left:
            node.left = mutate_tree_node(node.left)
        if node.right:
            node.right = mutate_tree_node(node.right)
        return node

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        robot = Robot(
            position=[random.uniform(0, 1), random.uniform(0, 1)], 
            health=random.uniform(0, 1), 
            enemy_position=[random.uniform(0, 1), random.uniform(0, 1)]
        )
        robot.tree = generate_random_tree()
        population.append(robot)
    return population

def evaluate_population_fitness(population):
    fitness_scores = [robot.evaluate_fitness() for robot in population]
    return fitness_scores

def tournament_selection(population, fitness_scores, diversity_factor=0.3, tournament_size=5):
    tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
    # Favorece el fitness, pero también selecciona individuos genéticamente diversos
    winner = max(tournament, key=lambda x: x[1] * (1 - diversity_factor) + calculate_genetic_diversity([x[0]]) * diversity_factor)
    return winner[0]

def crossover_trees_simple(tree1, tree2):
    if not isinstance(tree1, Node) or not isinstance(tree2, Node):
        raise ValueError("Los árboles para cruzar deben ser instancias de la clase Node.")
    
    if random.random() < 0.7:
        new_tree1 = Node(
            operation=tree1.operation, 
            left=tree2.left,
            right=tree1.right,
            value=tree1.value
        )
        new_tree2 = Node(
            operation=tree2.operation, 
            left=tree1.left,
            right=tree2.right,
            value=tree2.value
        )
    else:
        new_tree1 = Node(
            operation=tree1.operation, 
            left=tree1.left, 
            right=tree2.right,
            value=tree1.value
        )
        new_tree2 = Node(
            operation=tree2.operation, 
            left=tree2.left, 
            right=tree1.right,
            value=tree2.value
        )
    return new_tree1, new_tree2

def crossover_trees(tree1, tree2):
    # Cruce multipunto
    if random.random() < 0.7:
        new_tree1 = copy.deepcopy(tree1)
        new_tree2 = copy.deepcopy(tree2)

        # Intercambiar subárboles aleatorios
        if new_tree1.left and new_tree2.left:
            new_tree1.left, new_tree2.left = new_tree2.left, new_tree1.left
        if new_tree1.right and new_tree2.right:
            new_tree1.right, new_tree2.right = new_tree2.right, new_tree1.right

        return new_tree1, new_tree2
    else:
        # Cruce simple (como antes)
        return crossover_trees_simple(tree1, tree2)

def crossover(parent1, parent2):
    if not isinstance(parent1, Robot) or not isinstance(parent2, Robot):
        raise ValueError("Los padres deben ser instancias de la clase Robot.")
    if not parent1.tree or not parent2.tree:
        raise ValueError("Los padres deben tener un árbol de decisiones asignado.")

    child1_tree, child2_tree = crossover_trees(parent1.tree, parent2.tree)

    child1 = Robot(position=parent1.position, health=parent1.health, enemy_position=parent1.enemy_position)
    child1.tree = child1_tree
    child2 = Robot(position=parent2.position, health=parent2.health, enemy_position=parent2.enemy_position)
    child2.tree = child2_tree

    return child1, child2

def mutate(robot, mutation_rate):
    if random.random() < mutation_rate:
        # Mutar de forma agresiva (reemplazar el árbol completo ocasionalmente)
        if random.random() < 0.5:
            robot.tree = generate_random_tree(max_depth=random.randint(3, 5))
        else:
            robot.tree = mutate_tree_node(robot.tree)
            
def diversify_population(population, fitness_scores):
    diversified_population = []
    for i in range(len(population)):
        if random.random() < 0.5:
            # Crear un nuevo robot aleatorio
            new_robot = Robot(
                position=[random.uniform(0, 1), random.uniform(0, 1)],
                health=random.uniform(0, 1),
                enemy_position=[random.uniform(0, 1), random.uniform(0, 1)]
            )
            new_robot.tree = generate_random_tree(max_depth=random.randint(5, 7))
            diversified_population.append(new_robot)
        else:
            # Realizar una mutación agresiva en un clon
            clone = copy.deepcopy(population[i])
            mutate(clone, mutation_rate=random.uniform(0.3, 0.5))  # Mutación más alta
            diversified_population.append(clone)
    return diversified_population


# Función para convertir el árbol de expresión en una representación de texto
def tree_to_string(tree):
    if tree is None:
        return ""
    if tree.operation:
        left_str = tree_to_string(tree.left)
        right_str = tree_to_string(tree.right)
        return f"({left_str} {tree.operation} {right_str})"
    return str(tree.value)

# Función para calcular la distancia de Levenshtein entre dos cadenas de texto
def levenshtein_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    matrix = np.zeros((len_s1 + 1, len_s2 + 1))

    for i in range(len_s1 + 1):
        matrix[i][0] = i
    for j in range(len_s2 + 1):
        matrix[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,
                               matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + cost)

    return matrix[len_s1][len_s2]

# Función para calcular la diversidad genética de la población usando la distancia de Levenshtein
def calculate_genetic_diversity(population):
    trees = [robot.tree for robot in population]
    tree_strings = [tree_to_string(tree) for tree in trees]

    pairwise_distances = 0
    num_comparisons = 0

    for i in range(len(tree_strings)):
        for j in range(i + 1, len(tree_strings)):
            dist = levenshtein_distance(tree_strings[i], tree_strings[j])
            pairwise_distances += dist
            num_comparisons += 1

    diversity = pairwise_distances / num_comparisons if num_comparisons > 0 else 0
    return diversity

def plot_fitness(fitness_history):
    plt.figure(figsize=(12, 6))
    plt.plot(fitness_history, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', 
             markersize=6, markerfacecolor='green', markeredgewidth=2)
    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generation", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12, frameon=False, title="Fitness Metrics", title_fontsize=13)
    plt.tight_layout()
    plt.show()

def genetic_algorithm():
    population = initialize_population()
    best_fitness_acumulado = -float("inf")
    prev_best_fitness = -float("inf")
    stagnation_count = 0
    fitness_history = []

    mutation_rate = MUTATION_RATE

    for generation in range(MAX_GENERATIONS):
        fitness_scores = evaluate_population_fitness(population)
        best_fitness_generacion = max(fitness_scores)

        if best_fitness_generacion == prev_best_fitness:
            stagnation_count += 1
        else:
            stagnation_count = 0

        if best_fitness_generacion > best_fitness_acumulado:
            best_fitness_acumulado = best_fitness_generacion

        print(f"Generación {generation}: Mejor Fitness Acumulado = {best_fitness_acumulado:.3f} (Estancamiento: {stagnation_count})")

        diversity = calculate_genetic_diversity(population)
        print(f"Generación {generation}: Diversidad Genética = {diversity:.3f}")

        if best_fitness_acumulado >= TARGET_FITNESS:
            print("¡Se alcanzó el fitness óptimo!")
            break

        fitness_history.append(best_fitness_acumulado)

        # Adaptación dinámica al estancamiento
        if stagnation_count >= STAGNATION_THRESHOLD:
            print("Estancamiento detectado. Incrementando mutaciones y diversificando...")
            mutation_rate = min(0.5, mutation_rate * 1.5)  # Incrementar tasa de mutación
            population = diversify_population(population, fitness_scores)
            stagnation_count = 0
        else:
            mutation_rate = MUTATION_RATE  # Restablecer tasa de mutación

        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1, child2 = crossover(parent1, parent2)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population
        prev_best_fitness = best_fitness_generacion

    print(f"Mejor fitness alcanzado: {best_fitness_acumulado:.3f}")
    plot_fitness(fitness_history)


# Parámetros del algoritmo genético
POPULATION_SIZE = 60
MAX_GENERATIONS = 150
MUTATION_RATE = 0.1  # Increased mutation rate
TARGET_FITNESS = 0.95
STAGNATION_THRESHOLD = 2  # Increased threshold to allow more generations before diversification

# Ejecuta el algoritmo genético
genetic_algorithm()
