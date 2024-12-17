import random
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import copy

# Parámetros de la evolución
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
STAGNATION_THRESHOLD = 5

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

def tree_to_string(node):
    if node is None:
        return ''
    if node.operation:
        return f"({tree_to_string(node.left)} {node.operation} {tree_to_string(node.right)})"
    return str(node.value)

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fitness_scores = list(executor.map(lambda robot: robot.evaluate_fitness(), population))
    return fitness_scores

# Métodos de selección, cruce y mutación

def tournament_selection(population, fitness_scores, tournament_size=20):  # Aumento del tamaño del torneo
    tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]

def elitism_selection(population, fitness_scores, elite_fraction=0.1):
    num_elites = max(1, int(len(population) * elite_fraction))
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elites = [ind for ind, _ in sorted_population[:num_elites]]
    return elites

def crossover_trees_simple(tree1, tree2):
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

def crossover(parent1, parent2):
    child1_tree, child2_tree = crossover_trees_simple(parent1.tree, parent2.tree)
    child1 = Robot(position=parent1.position, health=parent1.health, enemy_position=parent1.enemy_position)
    child1.tree = child1_tree
    child2 = Robot(position=parent2.position, health=parent2.health, enemy_position=parent2.enemy_position)
    child2.tree = child2_tree

    return child1, child2

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        return random.uniform(-1, 1)  # Aumenta el rango de valores mutables
    return individual

def diversify_population(population, fitness_scores, mutation_rate=0.5):  # Aumentamos la tasa de mutación
    diversified_population = []
    for i in range(len(population)):
        if random.random() < 0.6:  # Aumentamos la probabilidad de que un individuo sea completamente nuevo
            new_robot = Robot(
                position=[random.uniform(0, 1), random.uniform(0, 1)],
                health=random.uniform(0, 1),
                enemy_position=[random.uniform(0, 1), random.uniform(0, 1)]
            )
            new_robot.tree = generate_random_tree(max_depth=random.randint(5, 7))
            diversified_population.append(new_robot)
        else:
            clone = copy.deepcopy(population[i])
            mutate(clone, mutation_rate)  # Aplicar mutación más fuerte
            diversified_population.append(clone)
    return diversified_population

def partial_reinitialization(population, fitness_scores, reinit_fraction=0.5):  # Aumentada a 50%
    num_to_reinit = int(len(population) * reinit_fraction)
    indices_to_reinit = random.sample(range(len(population)), num_to_reinit)
    for idx in indices_to_reinit:
        population[idx] = Robot(
            position=[random.uniform(0, 1), random.uniform(0, 1)],
            health=random.uniform(0, 1),
            enemy_position=[random.uniform(0, 1), random.uniform(0, 1)]
        )
        population[idx].tree = generate_random_tree()
    return population

def deeper_mutate(individual, mutation_rate=0.3):
    # Mutación más profunda en la estructura del árbol
    if random.random() < mutation_rate:
        # Seleccionar aleatoriamente un nodo para cambiar
        node_to_mutate = individual.tree
        # Asegúrese de que el nodo no sea None y sigue el recorrido en el árbol
        while node_to_mutate:
            if random.random() < 0.5:
                # Si hay un hijo izquierdo, descendemos en él
                if node_to_mutate.left:
                    node_to_mutate = node_to_mutate.left
                else:
                    break  # Si no tiene hijo izquierdo, salimos
            else:
                # Si hay un hijo derecho, descendemos en él
                if node_to_mutate.right:
                    node_to_mutate = node_to_mutate.right
                else:
                    break  # Si no tiene hijo derecho, salimos

        # Cambiar el nodo seleccionado por un subárbol completamente nuevo
        if node_to_mutate:
            node_to_mutate = generate_random_tree(max_depth=random.randint(3, 5))  # Profundidad más pequeña para evitar árboles demasiado grandes
            individual.tree = node_to_mutate  # Asignamos el nuevo subárbol al individuo


def crossover_trees_with_subtrees(tree1, tree2):
    # Intercambiar subárboles aleatorios
    if random.random() < 0.7:
        subtree1, subtree2 = random.choice([tree1, tree2]), random.choice([tree1, tree2])
        # Hacer el cruce de subárboles completos
        subtree1, subtree2 = subtree2, subtree1  # Simplemente intercambiamos los subárboles seleccionados
    return tree1, tree2

def run_evolution():
    population = initialize_population()
    stagnation_counter = 0
    mutation_rate = MUTATION_RATE
    fitness_over_time = []

    for generation in range(NUM_GENERATIONS):
        print(f"Generación {generation + 1}/{NUM_GENERATIONS}")

        fitness_scores = evaluate_population_fitness(population)

        max_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(max_fitness)]

        print(f"Mejor fitness de esta generación: {max_fitness:.5f}")

        fitness_over_time.append(max_fitness)

        # Detectar estancamiento por fitness
        if stagnation_counter >= STAGNATION_THRESHOLD:
            mutation_rate = min(0.6, mutation_rate * 1.5)  # Aumentar aún más la tasa de mutación
            stagnation_counter = 0
            print("Aumentando tasa de mutación debido al estancamiento...")

        elites = elitism_selection(population, fitness_scores)
        new_population = []

        while len(new_population) < POPULATION_SIZE - len(elites):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Usar crossover de subárboles
            child1_tree, child2_tree = crossover_trees_with_subtrees(parent1.tree, parent2.tree)
            child1 = Robot(position=parent1.position, health=parent1.health, enemy_position=parent1.enemy_position)
            child1.tree = child1_tree
            child2 = Robot(position=parent2.position, health=parent2.health, enemy_position=parent2.enemy_position)
            child2.tree = child2_tree

            # Mutar con mutación profunda
            deeper_mutate(child1, mutation_rate)
            deeper_mutate(child2, mutation_rate)

            new_population.append(child1)
            new_population.append(child2)

        population = elites + new_population

        # Diversificar población y realizar reinicialización parcial
        population = diversify_population(population, fitness_scores)
        population = partial_reinitialization(population, fitness_scores)

        # Detectar estancamiento
        if max_fitness == max(fitness_scores):
            stagnation_counter += 1
        else:
            stagnation_counter = 0

    plot_fitness(fitness_over_time)

import matplotlib.pyplot as plt

def plot_fitness(fitness_over_time):
    plt.figure(figsize=(12, 6))
    
    # Graficamos el fitness con un estilo atractivo
    plt.plot(fitness_over_time, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', markersize=6, markerfacecolor='green', markeredgewidth=2)
    
    # Títulos y etiquetas con una tipografía más destacada
    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generations", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')
    
    # Fondo de la gráfica y ajuste de la cuadrícula
    plt.gca().set_facecolor('#f0f0f0')  # Fondo gris claro
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)
    
    # Leyenda y personalización
    plt.legend(loc='upper left', fontsize=12, frameon=False, title="Métricas de Fitness", title_fontsize=13)
    
    # Aseguramos que todo se ajuste bien en la figura
    plt.tight_layout()
    
    # Mostramos la gráfica
    plt.show()


if __name__ == '__main__':
    run_evolution()
