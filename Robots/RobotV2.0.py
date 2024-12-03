# ============================================================
#                PROJECT NAME: Robot Wars Simulator
# ============================================================
# DESCRIPTION:
#   A Python simulation of a "Robot Wars" environment where two 
#   robots fight against each other. The winner is determined 
#   by survival after executing strategies driven by genetic 
#   programming (GP). The project includes:
#     - Actions: Move, shoot, etc.
#     - Sensor inputs for real-time decision-making.
#     - Co-evolution of GP individuals to control robot behaviors.
#
# AUTHOR:
#   Arahi Fernandez Monagas
#
# VERSION:
#   v2.0.0
#
# DATE CREATED:
#   2024-11-03
#
# LAST UPDATED:
#   2024-12-01
#
# LICENSE:
#   MIT License
#
# ============================================================
#                IMPORTS AND CONFIGURATION
# ============================================================

import numpy as np
import random
import matplotlib.pyplot as plt

# Global parameters
POPULATION_SIZE = 100  # Tamaño moderado para una evolución adecuada
GENES = 10  # Mantén el tamaño de los genes en 10
MAX_GENERATIONS = 410  # Asegura suficientes generaciones para la evolución
MUTATION_RATE = 0.05  # Tasa de mutación moderada
TOURNAMENT_SIZE = 7  # Selección de torneo de tamaño medio
CROSSOVER_RATE = 0.8  # Alta probabilidad de cruce
NO_IMPROVEMENT_LIMIT = 2  # Si no mejora en 5 generaciones, aumenta la mutación

# Acciones posibles para un robot
ACTIONS = ["move_forward", "move_backward", "turn_left", "turn_right", "shoot"]
 
# Clase Robot
class Robot:
    def __init__(self, genome):
        self.genome = genome  # Lista de genes que definen comportamiento
        self.fitness = 0  # Fitness inicial del robot
        self.position = np.array([random.randint(0, 10), random.randint(0, 10)])  # Posición inicial
        self.health = 100  # Salud inicial
    
    def execute_actions(self, enemy_position):
        """
        Ejecuta acciones basadas en el genoma del robot.
        """
        for gene in self.genome:
            action = ACTIONS[gene % len(ACTIONS)]
            if action == "move_forward":
                self.position[1] += 1
            elif action == "move_backward":
                self.position[1] -= 1
            elif action == "turn_left":
                self.position[0] -= 1
            elif action == "turn_right":
                self.position[0] += 1
            elif action == "shoot":
                if np.linalg.norm(self.position - enemy_position) < 3:
                    return True  # Disparo exitoso
        return False

# Función para generar un genoma aleatorio
def random_genome():
    return [random.randint(0, len(ACTIONS) - 1) for _ in range(GENES)]

# Selección por torneo
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda x: min(x.fitness, 100))
    return winner

# Cruce entre dos robots
def crossover(parent1, parent2):
    split_point = random.randint(0, GENES - 1)
    child_genome = parent1.genome[:split_point] + parent2.genome[split_point:]
    return Robot(child_genome)

# Mutación
def mutate(robot, mutation_rate):
    for i in range(len(robot.genome)):
        if random.random() < mutation_rate:
            # Realiza la mutación con un valor aleatorio
            robot.genome[i] = random.randint(0, len(ACTIONS) - 1)

    # Limitar el valor máximo del fitness a 100 después de la mutación
    robot.fitness = min(robot.fitness, 100)

# Inicialización de la población
def initialize_population():
    return [Robot(random_genome()) for _ in range(POPULATION_SIZE)]

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

# Ajustar la tasa de mutación más agresivamente si es necesario
def genetic_algorithm():
    population = initialize_population()
    best_fitness = []
    worst_fitness = []
    average_fitness = []
    diversity = []  # Para almacenar la diversidad
    stagnant_generations = 0
    last_best_fitness = 0
    global MUTATION_RATE

    for generation in range(MAX_GENERATIONS):
        # Evaluar el fitness de cada robot
        for robot in population:
            opponent = random.choice(population)
            evaluate_fitness(robot, opponent)

        # Calcular el mejor, peor y promedio de fitness
        best_fitness_current = max(robot.fitness for robot in population)
        worst_fitness_current = min(robot.fitness for robot in population)
        average_fitness_current = np.mean([robot.fitness for robot in population])

        # Calcular la diversidad (desviación estándar de los fitness)
        diversity_current = np.std([robot.fitness for robot in population])

        best_fitness.append(best_fitness_current)
        worst_fitness.append(worst_fitness_current)
        average_fitness.append(average_fitness_current)
        diversity.append(diversity_current)

        # Ajustar tasa de mutación si no hay mejora
        if best_fitness_current <= last_best_fitness:
            stagnant_generations += 1
            if stagnant_generations >= NO_IMPROVEMENT_LIMIT:
                MUTATION_RATE += 0.1  # Aumento más agresivo
                stagnant_generations = 0  # Reiniciar el contador
                print(f"Incrementando la tasa de mutación a: {MUTATION_RATE}")
        else:
            stagnant_generations = 0  # Reiniciar si hay mejora

        if best_fitness_current > last_best_fitness:
            last_best_fitness = best_fitness_current

        # Criterio de terminación
        if best_fitness_current == 100:
            print(f"Solución óptima encontrada en la generación {generation}, {best_fitness_current}")
            break

        # Selección, cruce y mutación para crear la nueva población
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)

            # Determinar si los padres se cruzan basado en el crossover_rate
            if random.random() < CROSSOVER_RATE:
                child = crossover(parent1, parent2)  # Realizar el cruce
            else:
                # Si no se realiza cruce, el hijo es una copia de uno de los padres
                child = Robot(parent1.genome[:])  # Copiar el genoma de uno de los padres

            mutate(child, MUTATION_RATE)  # Pasar la tasa de mutación correctamente
            new_population.append(child)

        population = new_population

        # Mostrar el mejor fitness de cada generación
        print(f"Generación {generation}: Mejor Fitness = {best_fitness_current}")

    # Mostrar gráficas
    # Gráfica del mejor fitness
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness)
    plt.title("Mejor Fitness a lo largo de las generaciones")
    plt.xlabel("Generación")
    plt.ylabel("Mejor Fitness")
    plt.show()



# Ejecutar el algoritmo genético
if __name__ == "__main__":
    genetic_algorithm()
