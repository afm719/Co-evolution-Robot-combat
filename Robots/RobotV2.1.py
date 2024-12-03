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
#   v2.1.0
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

import random
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo genético
POPULATION_SIZE = 100
GENOME_LENGTH = 10
MUTATION_RATE = 0.03
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3
MAX_GENERATIONS = 200
ELITE_COUNT = 2
NO_IMPROVEMENT_LIMIT = 10

# Clase Robot con genoma y fitness
class Robot:
    def __init__(self, genome=None):
        if genome is None:
            self.genome = np.random.uniform(-1, 1, GENOME_LENGTH)
        else:
            self.genome = genome
        self.fitness = 0

# Inicialización de la población
def initialize_population():
    return [Robot() for _ in range(POPULATION_SIZE)]

# Evaluación de fitness (ajustada para garantizar valores positivos)
def evaluate_fitness(robot):
    # Fitness basado en la distancia al vector óptimo (1,1,...,1)
    max_fitness_possible = GENOME_LENGTH * (2 ** 2)  # Máximo fitness posible
    fitness_raw = max_fitness_possible - np.sum((robot.genome - 1) ** 2)
    # Escalado para recompensar proporcionalmente
    robot.fitness = fitness_raw / max_fitness_possible * 100  # Fitness en rango 0-100

# Selección por torneo
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)

# Crossover (cruce)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, GENOME_LENGTH - 1)
    child_genome = np.concatenate((parent1.genome[:crossover_point], parent2.genome[crossover_point:]))
    return Robot(genome=child_genome)

# Mutación
def mutate(robot, mutation_rate):
    for i in range(len(robot.genome)):
        if random.random() < mutation_rate:
            robot.genome[i] += np.random.normal(0, 0.1)  # Mutación ligera para evitar cambios drásticos

# Calcular diversidad genética (distancia media entre genomas)
def calculate_genetic_diversity(population):
    genomes = np.array([robot.genome for robot in population])
    pairwise_distances = np.sum([np.linalg.norm(g1 - g2) for g1 in genomes for g2 in genomes])
    return pairwise_distances / (len(population) ** 2)

def genetic_algorithm():
    population = initialize_population()
    best_fitness = []
    average_fitness = []
    worst_fitness = []
    diversity = []
    stagnant_generations = 0
    last_best_fitness = -np.inf
    current_mutation_rate = MUTATION_RATE

    for generation in range(MAX_GENERATIONS):
        # Evaluar fitness
        for robot in population:
            evaluate_fitness(robot)

        # Ordenar población por fitness (descendente)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Calcular métricas
        best_fitness_current = population[0].fitness  # Mejor fitness
        average_fitness_current = np.mean([robot.fitness for robot in population])
        worst_fitness_current = min(robot.fitness for robot in population)
        diversity_current = calculate_genetic_diversity(population)

        # Registrar métricas
        best_fitness.append(best_fitness_current)
        average_fitness.append(average_fitness_current)
        diversity.append(diversity_current)
        worst_fitness.append(worst_fitness_current)

        # Incrementar tasa de mutación si no hay mejora
        if best_fitness_current <= last_best_fitness:
            stagnant_generations += 1
            if stagnant_generations >= NO_IMPROVEMENT_LIMIT:
                current_mutation_rate = min(0.3, current_mutation_rate + 0.05)
                stagnant_generations = 0
                print(f"Incrementando la tasa de mutación a: {current_mutation_rate:.2f}")
        else:
            stagnant_generations = 0

        if best_fitness_current > last_best_fitness:
            last_best_fitness = best_fitness_current

        # Detener si alcanzamos el fitness óptimo
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
                child = Robot(parent1.genome[:])

            mutate(child, current_mutation_rate)
            new_population.append(child)

        population = new_population

        # Mostrar progreso
        print(f"Generación {generation}: Mejor Fitness = {best_fitness_current:.2f}")

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label="Mejor Fitness", color='blue')
    plt.title("Mejor Fitness a lo largo de las generaciones")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(best_fitness, label="Mejor Fitness", color='blue')
    plt.plot(average_fitness, label="Fitness Promedio", color='green')
    plt.plot(worst_fitness, label="Peor Fitness", color='red')
    plt.title("Evolución del Fitness a lo largo de las generaciones", fontsize=14)
    plt.xlabel("Generación", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


    # Diversidad genética
    plt.figure(figsize=(12, 6))
    plt.plot(diversity, label="Diversidad Genética", color='purple')
    plt.title("Evolución de la Diversidad Genética", fontsize=14)
    plt.xlabel("Generación", fontsize=12)
    plt.ylabel("Diversidad", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()




# Ejecutar el algoritmo genético
if __name__ == "__main__":
    genetic_algorithm()
