import random
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo genético
POPULATION_SIZE = 100
GENOME_LENGTH = 10  # Esto no será usado directamente en el GP, pero ayuda a definir el tamaño de los robots.
MUTATION_RATE = 0.03
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3
MAX_GENERATIONS = 200
ELITE_COUNT = 2
NO_IMPROVEMENT_LIMIT = 10

# Parâmetros de los árboles de decisiones
TREE_DEPTH = 3  # Profundidad del árbol de decisión
NUM_ACTIONS = 4  # Número de acciones posibles: 'turn left', 'turn right', 'move forward', 'shoot'


class DecisionTree:
    """ Clase para el árbol de decisión en los robots """
    def __init__(self, depth=3):
        self.depth = depth
        self.tree = self.generate_tree(depth)

    def generate_tree(self, depth):
        """ Generar un árbol binario de decisión """
        if depth == 0:
            # Si la profundidad es 0, es una acción
            return random.choice(['turn left', 'turn right', 'move forward', 'shoot'])
        else:
            # Nodo condicional con acción en ramas
            left = self.generate_tree(depth - 1)
            right = self.generate_tree(depth - 1)
            return ('IF', random.random(), left, right)

    def evaluate(self, robot):
        """ Evaluar el comportamiento del robot según el árbol """
        return self.evaluate_node(self.tree, robot)

    def evaluate_node(self, node, robot):
        """ Evaluar un nodo del árbol """
        if isinstance(node, tuple) and node[0] == 'IF':
            condition, left, right = node[1], node[2], node[3]
            if robot.health > condition:  # Condición ejemplo
                return self.evaluate_node(left, robot)
            else:
                return self.evaluate_node(right, robot)
        else:
            return node  # Acción (p.ej., 'turn left')


class Robot:
    def __init__(self, health=100):
        self.health = health
        self.tree = DecisionTree(depth=TREE_DEPTH)  # Usamos árboles de decisión con una profundidad de 3
        self.fitness = 0

    def attack(self):
        """ Simula un ataque """
        base_damage = np.sum(self.tree.evaluate(self))  # Calculamos daño con el árbol
        random_factor = random.uniform(0.8, 1.2)
        return int(base_damage * random_factor)

    def move(self):
        """ El robot realiza un movimiento determinado por su árbol de decisión """
        return self.tree.evaluate(self)


def initialize_population():
    """ Inicializar la población con robots y sus árboles de decisión """
    return [Robot() for _ in range(POPULATION_SIZE)]


# Evaluación del fitness (modificación para mejorar la penalización)
def evaluate_fitness(robot, enemy_robot):
    robot_health = robot.health
    enemy_health = enemy_robot.health
    actions_taken = 0

    for _ in range(10):  # Número de pasos en el combate
        if robot.execute_actions(enemy_robot.position):
            enemy_health -= 10
            actions_taken += 1
        if enemy_robot.execute_actions(robot.position):
            robot_health -= 10

    # Cálculo del fitness con penalización por la salud
    robot.fitness = max(0, robot_health - enemy_health) + actions_taken
    # Limitar el valor máximo del fitness a 100
    robot.fitness = min(robot.fitness, 100)
    return robot.fitness


def tournament_selection(population, tournament_size):
    """ Selección por torneo para elegir a un robot """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


def crossover(parent1, parent2):
    """ Cruzamiento de árboles de decisión de dos robots """
    crossover_point = random.randint(1, TREE_DEPTH - 1)
    child_tree = parent1.tree if random.random() < CROSSOVER_RATE else parent2.tree
    return Robot(health=parent1.health), child_tree


def mutate(robot, mutation_rate):
    """ Aplicar mutaciones a un robot """
    if random.random() < mutation_rate:
        robot.tree = DecisionTree(depth=TREE_DEPTH)  # Reemplazar árbol de decisión
        robot.health = random.uniform(50, 100)  # Modificar salud


def calculate_genetic_diversity(population):
    """ Calcular la diversidad genética de la población mediante la similitud estructural de los árboles """
    pairwise_distances = 0
    comparisons = 0

    # Comparamos los árboles de decisión de todos los robots en la población
    for i, robot1 in enumerate(population):
        for j, robot2 in enumerate(population):
            if i < j:
                # Comparamos los árboles de decisión de robot1 y robot2 en base a sus evaluaciones
                # En lugar de calcular la distancia de los árboles, comparamos las acciones que toman
                actions1 = simulate_robot_actions(robot1)
                actions2 = simulate_robot_actions(robot2)

                # Contamos cuántas veces coinciden las acciones tomadas en la simulación
                similarity = sum(1 for a1, a2 in zip(actions1, actions2) if a1 == a2)
                pairwise_distances += similarity
                comparisons += 1

    # La diversidad genética es la media de las similitudes entre todos los pares de robots
    return 1 - (pairwise_distances / comparisons) if comparisons > 0 else 0

def simulate_robot_actions(robot, steps=10):
    """ Simula las acciones de un robot durante 'steps' pasos para obtener un conjunto de decisiones """
    actions = []
    for _ in range(steps):
        action = robot.tree.evaluate(robot)  # Ejecuta el árbol de decisiones del robot
        actions.append(action)
    return actions


def genetic_algorithm():
    """ Algoritmo Genético para evolucionar robots con árboles de decisión """
    population = initialize_population()
    best_fitness = []
    average_fitness = []
    worst_fitness = []
    diversity = []

    for generation in range(MAX_GENERATIONS):
        # Evaluar el fitness de cada robot
        for robot in population:
            evaluate_fitness(robot)

        # Ordenar robots por fitness (de mayor a menor)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Registrar métricas
        best_fitness_current = population[0].fitness
        best_fitness.append(best_fitness_current)
        average_fitness_current = np.mean([robot.fitness for robot in population])
        average_fitness.append(average_fitness_current)
        diversity_current = calculate_genetic_diversity(population)
        diversity.append(diversity_current)
        worst_fitness_current = min(robot.fitness for robot in population)
        worst_fitness.append(worst_fitness_current)

        # Mostrar progreso
        print(f"Generación {generation}: Mejor Fitness = {best_fitness_current:.2f}")

        # Detener si alcanzamos un fitness óptimo
        if best_fitness_current >= 99.99:
            print(f"Solución óptima encontrada en la generación {generation}: {best_fitness_current:.2f}")
            break

        # Generar nueva población (elitismo + nueva generación)
        new_population = population[:ELITE_COUNT]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_RATE:
                child = crossover(parent1, parent2)
            else:
                child = Robot(parent1.health)

            mutate(child, MUTATION_RATE)
            new_population.append(child)

        population = new_population

    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(best_fitness, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', markersize=6, markerfacecolor='green', markeredgewidth=2)
    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generación", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)
    plt.tight_layout()  
    plt.show()

    # Devolver los mejores robots
    best_robots = population[:2]  # Los robots con el mejor fitness
    worst_robots = population[-2:]  # Los robots con el peor fitness
    return best_robots, worst_robots


if __name__ == "__main__":
    genetic_algorithm()
