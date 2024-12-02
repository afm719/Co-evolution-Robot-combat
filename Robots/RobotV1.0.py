import random
import numpy as np
import matplotlib.pyplot as plt

# Clase del Nodo del Árbol
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value  # Función o terminal
        self.left = left    # Nodo izquierdo
        self.right = right  # Nodo derecho

    def evaluate(self, state):
        """Evalúa el árbol según el estado."""
        if callable(self.value):  # Si el nodo es una función
            if self.left is None or self.right is None:  # Validación de hijos
                raise ValueError("Nodo interno sin hijos válidos")
            return self.value(state, self.left, self.right)
        return self.value  # Si es terminal, retorna su valor


# Funciones internas del árbol
def if_enemy_near(state, left, right):
    if state["enemy_distance"] < 5:
        return left.evaluate(state)
   
    return right.evaluate(state)

def if_low_energy(state, left, right):
    if state["energy"] < 50:
        return left.evaluate(state)
    return right.evaluate(state)

def if_wall_close(state, left, right):
    if state["wall_distance"] < 2:
        return left.evaluate(state)
    return right.evaluate(state)


# Terminales (acciones)
def fire_action(state, _, __):
    state["energy"] -= 5
    return "fire"

def move_action(state, _, __):
    state["position"] += np.array([np.cos(np.radians(state["direction"])),
                                   np.sin(np.radians(state["direction"]))]) * 0.5
    return "move"

def turn_action(state, _, __):
    state["direction"] = (state["direction"] + random.uniform(-45, 45)) % 360
    return "turn"


# Funciones y terminales disponibles
FUNCTIONS = [if_enemy_near, if_low_energy, if_wall_close]
TERMINALS = [fire_action, move_action, turn_action]


# Generador de árboles aleatorios
def generate_random_tree(depth):
    """Genera un árbol aleatorio asegurando que los nodos internos tengan dos hijos válidos."""
    if depth == 0 or random.random() < 0.3:  # Terminal
        return TreeNode(random.choice(TERMINALS))
    func = random.choice(FUNCTIONS)
    left_child = generate_random_tree(depth - 1)
    right_child = generate_random_tree(depth - 1)
    # Verificar que ambos hijos estén presentes
    if left_child is None or right_child is None:
        raise ValueError("Error al generar el árbol: nodo interno sin hijos válidos")
    return TreeNode(func, left=left_child, right=right_child)


# Validador de árboles
def validate_tree(tree):
    """Valida que un árbol tenga nodos completos."""
    if tree is None:
        return False
    if callable(tree.value):  # Si es un nodo interno
        if not validate_tree(tree.left) or not validate_tree(tree.right):
            print(f"Árbol inválido detectado: {tree.value}")
            return False
    return True  # Si es un terminal, es válido


# Función para corregir árboles inválidos
def fix_tree(tree, depth=3):
    """Corrige un árbol inválido reemplazando nodos incompletos."""
    if tree is None or depth == 0:
        return generate_random_tree(1)  # Reemplaza con un nodo terminal válido
    if callable(tree.value):  # Si es un nodo interno
        # Asegura que los nodos internos tengan dos hijos válidos
        if tree.left is None:
            tree.left = generate_random_tree(depth - 1)
        if tree.right is None:
            tree.right = generate_random_tree(depth - 1)
        
        tree.left = fix_tree(tree.left, depth - 1)
        tree.right = fix_tree(tree.right, depth - 1)
    return tree


# Clase del Robot
class Robot:
    def __init__(self, program=None):
        self.program = program or generate_random_tree(3)
        self.fitness = 0

    def execute(self, state):
        """Ejecuta el árbol de decisiones."""
        if not validate_tree(self.program):  # Validar antes de ejecutar
            print("Corrigiendo árbol inválido durante evaluación")
            self.program = fix_tree(self.program)
        return self.program.evaluate(state)


# Algoritmo Genético
class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=20, elitism=2):
        self.population = [Robot() for _ in range(population_size)]
        self.generations = generations
        self.elitism = elitism
        self.best_fitness_per_generation = []
        self.avg_fitness_per_generation = []
        self.worst_fitness_per_generation = []

    def evaluate_fitness(self, robot):
        """Evalúa el fitness del robot."""
        total_fitness = 0
        unique_actions = set()
        num_simulations = 20
        
        # Verificar y corregir árboles inválidos
        if not validate_tree(robot.program):
            print("Corrigiendo árbol inválido durante evaluación")
            robot.program = fix_tree(robot.program)

        for _ in range(num_simulations):
            state = {
                "enemy_distance": random.uniform(0, 10),
                "wall_distance": random.uniform(0, 10),
                "energy": 100,
                "position": np.array([random.uniform(1, 9), random.uniform(1, 9)]),
                "direction": random.uniform(0, 360),
            }
            action = robot.execute(state)
            unique_actions.add(action)
            if action == "fire" and state["enemy_distance"] < 5:
                total_fitness += 50
            elif action == "move":
                total_fitness += 10
            elif action == "turn":
                total_fitness += 5

        diversity_bonus = len(unique_actions) * 5
        robot.fitness = total_fitness / num_simulations + diversity_bonus

    def crossover(self, parent1, parent2):
        """Cruza dos robots preservando subárboles válidos."""
        def copy_tree(tree):
            if not tree:
                return None
            return TreeNode(tree.value, copy_tree(tree.left), copy_tree(tree.right))

        child_program = copy_tree(parent1.program)
        subtree_parent2 = copy_tree(parent2.program)
        if child_program and subtree_parent2:
            if random.random() < 0.5 and child_program.left:
                child_program.left = subtree_parent2.left
            elif child_program.right:
                child_program.right = subtree_parent2.right

        if not validate_tree(child_program):  # Validar el árbol generado
            print("Árbol inválido después del crossover, corrigiendo...")
            child_program = fix_tree(child_program)

        return Robot(program=child_program)

    def mutate(self, robot):
        """Mutación segura: asegura subárboles válidos."""
        def mutate_subtree(tree):
            if random.random() < 0.1:  # Baja probabilidad de mutación
                return generate_random_tree(1)  # Reemplaza con un subárbol pequeño
            if tree.left:
                tree.left = mutate_subtree(tree.left)
            if tree.right:
                tree.right = mutate_subtree(tree.right)
            return tree

        mutated_tree = mutate_subtree(robot.program)
        if not validate_tree(mutated_tree):  # Validar el árbol mutado
            print("Árbol inválido después de la mutación, corrigiendo...")
            mutated_tree = fix_tree(mutated_tree)
        robot.program = mutated_tree


    def select_parents(self):
        """Selección por torneo con mayor presión evolutiva."""
        tournament_size = 4
        parents = []
        for _ in range(len(self.population) - self.elitism):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda r: r.fitness)
            parents.append(winner)
        return parents

    def evolve_population(self):
        """Evoluciona la población a través de generaciones."""
        for generation in range(self.generations):
            for robot in self.population:
                self.evaluate_fitness(robot)

            # Estadísticas de fitness
            fitness_values = [robot.fitness for robot in self.population]
            self.best_fitness_per_generation.append(max(fitness_values))
            self.avg_fitness_per_generation.append(sum(fitness_values) / len(fitness_values))
            self.worst_fitness_per_generation.append(min(fitness_values))

            print(f"Generación {generation + 1}: "
                  f"Mejor: {max(fitness_values):.2f}, "
                  f"Promedio: {self.avg_fitness_per_generation[-1]:.2f}, "
                  f"Peor: {min(fitness_values):.2f}")

            # Elitismo
            sorted_population = sorted(self.population, key=lambda r: r.fitness, reverse=True)
            new_population = sorted_population[:self.elitism]

            # Reproducción
            while len(new_population) < len(self.population):
                parent1, parent2 = random.sample(sorted_population[:10], 2)
                child = self.crossover(parent1, parent2)
                if random.random() < 0.3:
                    self.mutate(child)
                new_population.append(child)

            self.population = new_population

    def plot_fitness(self):
        """Grafica la evolución del fitness."""
        generations = range(1, self.generations + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_fitness_per_generation, label="Mejor Fitness", color="green")
        plt.plot(generations, self.avg_fitness_per_generation, label="Fitness Promedio", color="blue")
        plt.plot(generations, self.worst_fitness_per_generation, label="Peor Fitness", color="red")
        plt.title("Evolución del Fitness")
        plt.xlabel("Generaciones")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()


# Ejecutar el Algoritmo Genético
if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=50, generations=30, elitism=5)
    ga.evolve_population()
    ga.plot_fitness()
