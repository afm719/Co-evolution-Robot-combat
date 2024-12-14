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
            # Evaluar el árbol de decisión
            raw_fitness = self.tree.evaluate(self)

            # Componentes de la evaluación: Distancia al objetivo, salud, distancia al enemigo
            distance_to_goal = abs(self.position[0] - 0.5) + abs(self.position[1] - 0.5)
            health_penalty = max(0, 1 - self.health)  # Penalización por baja salud
            distance_to_enemy = max(0, 1 - (abs(self.enemy_position[0] - self.position[0]) + abs(self.enemy_position[1] - self.position[1])) / 2)

            # Penalización por complejidad del árbol
            complexity_penalty = len(tree_to_string(self.tree)) / 100.0

            # Calcular el fitness total con ponderaciones balanceadas
            raw_fitness = 1 - (0.4 * distance_to_goal + 0.4 * health_penalty + 0.2 * distance_to_enemy + complexity_penalty)
            
            # Normalizar fitness entre 0 y 1
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

def elitism_selection(population, fitness_scores, elite_fraction=0.1):
    num_elites = max(1, int(len(population) * elite_fraction))
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elites = [ind for ind, _ in sorted_population[:num_elites]]
    return elites


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

def partial_reinitialization(population, fitness_scores, reinit_fraction=0.2):
    num_to_reinit = int(len(population) * reinit_fraction)
    # Seleccionamos aleatoriamente individuos para reemplazar
    indices_to_reinit = random.sample(range(len(population)), num_to_reinit)
    for idx in indices_to_reinit:
        population[idx] = Robot(
            position=[random.uniform(0, 1), random.uniform(0, 1)],
            health=random.uniform(0, 1),
            enemy_position=[random.uniform(0, 1), random.uniform(0, 1)]
        )
        population[idx].tree = generate_random_tree()
    return population

def calculate_genetic_diversity(population):
    # Usando la distancia de Levenshtein para calcular la diversidad de los árboles
    # como un proxy de diversidad genética
    def levenshtein(a, b):
        if len(a) < len(b):
            return levenshtein(b, a)
        if len(b) == 0:
            return len(a)
        a, b = list(a), list(b)
        costs = range(len(b) + 1)
        for i, ca in enumerate(a):
            cost = i + 1
            new_costs = [cost]
            for j, cb in enumerate(b):
                last_cost = costs[j]
                cost = min(
                    last_cost + (ca != cb),
                    new_costs[j] + 1,
                    costs[j + 1] + 1
                )
                new_costs.append(cost)
            costs = new_costs
        return costs[-1]
    
    trees = [tree_to_string(robot.tree) for robot in population]
    total_diversity = 0
    num_comparisons = 0
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            total_diversity += levenshtein(trees[i], trees[j])
            num_comparisons += 1
    if num_comparisons == 0:
        return 0
    return total_diversity / num_comparisons

def tree_to_string(node):
    if node is None:
        return ''
    if node.operation:
        return f"({tree_to_string(node.left)} {node.operation} {tree_to_string(node.right)})"
    return str(node.value)

# Parámetros de la evolución
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
STAGNATION_THRESHOLD = 2
def plot_fitness_over_time(fitness_values):
    plt.plot(fitness_values)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del fitness a lo largo de las generaciones')
    plt.show()
# Modifica la tasa de mutación para ver si hay mejoras
def run_evolution():
    # Inicializar población
    population = initialize_population()
    stagnation_counter = 0
    mutation_rate = MUTATION_RATE
    fitness_over_time = []  # Lista para almacenar el fitness de cada generación

    for generation in range(NUM_GENERATIONS):
        print(f"Generación {generation + 1}/{NUM_GENERATIONS}")

        # Evaluación de la población
        fitness_scores = evaluate_population_fitness(population)

        # Obtener el mejor fitness de esta generación
        max_fitness = max(fitness_scores)
        best_fitness = max_fitness
        best_individual = population[fitness_scores.index(max_fitness)]

        print(f"Mejor fitness de esta generación: {max_fitness:.5f}")

        # Almacenar el mejor fitness de esta generación
        fitness_over_time.append(max_fitness)

        # Estancamiento detectado, aumentar la tasa de mutación
        if stagnation_counter >= STAGNATION_THRESHOLD:
            mutation_rate = min(0.5, mutation_rate * 1.5)
            stagnation_counter = 0
            print("Aumentando tasa de mutación debido al estancamiento...")
        else:
            mutation_rate = MUTATION_RATE

        # Selección y cruzamiento
        elites = elitism_selection(population, fitness_scores)
        new_population = []

        while len(new_population) < POPULATION_SIZE - len(elites):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1, child2 = crossover(parent1, parent2)

            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)

            new_population.append(child1)
            new_population.append(child2)

        population = elites + new_population

        # Diversificación y reinicialización parcial
        population = diversify_population(population, fitness_scores)
        population = partial_reinitialization(population, fitness_scores)

    # Después del bucle, graficar la evolución del fitness
    plot_fitness_over_time(fitness_over_time)

    return best_individual
# Ejecutar la evolución
best_robot = run_evolution()

# Mostrar el mejor robot encontrado
print("Mejor robot encontrado:")
print(f"Fitness: {best_robot.fitness:.3f}")
