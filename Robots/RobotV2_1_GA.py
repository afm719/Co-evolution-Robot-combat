# ============================================================
#                PROJECT NAME: Robot Wars Simulator
# ============================================================
# DESCRIPTION:
#   A Python simulation of a "Robot Wars" environment where two 
#   robots fight against each other. The winner is determined 
#   by survival after executing strategies driven by genetic 
#   algorithms (GA). The project includes:
#     - Actions: Move, shoot, etc.
#     - Sensor inputs for real-time decision-making.
#     - Co-evolution of GA individuals to control robot behaviors.
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
#   2024-12-03
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

# Genetic algorithm parameters
POPULATION_SIZE = 100
GENOME_LENGTH = 10
MUTATION_RATE = 0.03
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3
MAX_GENERATIONS = 200
ELITE_COUNT = 2
NO_IMPROVEMENT_LIMIT = 10


class Robot:
    def __init__(self, genome=None):
        if genome is None:
            self.genome = np.random.uniform(-1, 1, GENOME_LENGTH)
        else:
            self.genome = genome
        self.fitness = 0
        self.health = 100  

    def attack(self):
        base_damage = np.sum(self.genome)  
        random_factor = random.uniform(0.8, 1.2)  
        return int(base_damage * random_factor)

    def move(self):
        actions = ['turn left', 'turn right', 'move forward']
        return random.choice(actions)

    def battle(robot1, robot2):
        print(f"Battle started between Robot 1 and Robot 2!\n")
        round_number = 1
        while robot1.health > 0 and robot2.health > 0:
            print(f"Round {round_number}:")
            damage1 = robot1.attack()
            robot2.health -= damage1
            print(f"Robot 1 attacks with {damage1} damage. Robot 2's health: {robot2.health}")

            if robot2.health <= 0:
                print("Robot 1 wins the battle!")
                break

            damage2 = robot2.attack()
            robot1.health -= damage2
            print(f"Robot 2 attacks with {damage2} damage. Robot 1's health: {robot1.health}")

            if robot1.health <= 0:
                print("Robot 2 wins the battle!")
                break
            
            move1 = robot1.move()
            move2 = robot2.move()
            print(f"Robot 1 moves: {move1}")
            print(f"Robot 2 moves: {move2}\n")

            round_number += 1


def initialize_population():
    return [Robot() for _ in range(POPULATION_SIZE)]


def evaluate_fitness(robot):
    # Fitness based on distance to the optimal vector (1,1,...,1)
    max_fitness_possible = GENOME_LENGTH * (2 ** 2)  # Maximum possible fitness
    fitness_raw = max_fitness_possible - np.sum((robot.genome - 1) ** 2)
    # Scale to reward proportionally
    robot.fitness = fitness_raw / max_fitness_possible * 100  # Fitness in range 0-100


def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


def crossover(parent1, parent2):
    crossover_point = random.randint(1, GENOME_LENGTH - 1)
    child_genome = np.concatenate((parent1.genome[:crossover_point], parent2.genome[crossover_point:]))
    return Robot(genome=child_genome)


def mutate(robot, mutation_rate):
    for i in range(len(robot.genome)):
        if random.random() < mutation_rate:
            robot.genome[i] += np.random.normal(0, 0.1)  


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
    current_mutation_rate = MUTATION_RATE

    for generation in range(MAX_GENERATIONS):
        # Evaluate fitness
        for robot in population:
            evaluate_fitness(robot)

        # Sort population by fitness (descending)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Record metrics
        best_fitness_current = population[0].fitness
        best_fitness.append(best_fitness_current)
        average_fitness_current = np.mean([robot.fitness for robot in population])
        average_fitness.append(average_fitness_current)
        diversity_current = calculate_genetic_diversity(population)
        diversity.append(diversity_current)
        worst_fitness_current = min(robot.fitness for robot in population)
        worst_fitness.append(worst_fitness_current)

        # Show progress
        print(f"Generation {generation}: Best Fitness = {best_fitness_current:.5f}")

        # Stop if we reach optimal fitness
        if best_fitness_current >= 99.99:
            print(f"Optimal solution found in generation {generation}: {best_fitness_current:.4f}")
            break

        # Generate new population (elitism + new generation)
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

        # Show progress
        #print(f"Generation {generatcion}: Best Fitness = {best_fitness_current:.2f}")

    # Plot results
    #--------------------------------------------------------#
    plt.figure(figsize=(12, 6))

    plt.plot(best_fitness, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', markersize=6, markerfacecolor='green', markeredgewidth=2)

    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generation", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')

    plt.gca().set_facecolor('#f0f0f0')  # Fondo gris claro
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)

    plt.legend(loc='upper left', fontsize=12, frameon=False, title="Fitness Metrics", title_fontsize=13)

    plt.tight_layout()  
    plt.show()

    # #--------------------------------------------------------#

    # plt.figure(figsize=(12, 6))
    # plt.plot(best_fitness, label="Best Fitness", color='royalblue', linestyle='-', marker='o', markersize=6, markerfacecolor='yellow', markeredgewidth=2)
    # plt.plot(average_fitness, label="Average Fitness", color='forestgreen', linestyle='--', marker='s', markersize=6, markerfacecolor='lime', markeredgewidth=2)
    # plt.plot(worst_fitness, label="Worst Fitness", color='darkred', linestyle=':', linewidth=2)
    # plt.title("Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkslategray')
    # plt.xlabel("Generation", fontsize=14, fontweight='bold')
    # plt.ylabel("Fitness", fontsize=14, fontweight='bold')
    # plt.gca().set_facecolor('#f9f9f9')  # Fondo gris muy claro
    # plt.grid(True, which='both', linestyle='--', color='lightgray', alpha=0.7)
    # plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='lightblue', title="Fitness Metrics", title_fontsize=13)
    # plt.tight_layout()
    # plt.show()


    # #--------------------------------------------------------#

    # # Genetic diversity
    # plt.figure(figsize=(12, 6))

    # plt.plot(diversity, label="Genetic Diversity", color='indigo', linestyle='-', marker='D', markersize=6, markerfacecolor='yellow', markeredgewidth=2)

    # plt.title("Genetic Diversity Evolution", fontsize=16, fontweight='bold', color='darkgreen')
    # plt.xlabel("Generation", fontsize=14, fontweight='bold')
    # plt.ylabel("Diversity", fontsize=14, fontweight='bold')

    # plt.gca().set_facecolor('#f0f0f0')
    # plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)

    # plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='lightyellow', title="Diversity Metrics", title_fontsize=13)

    # plt.tight_layout()

    # plt.show()

    # At the end of all generations, return the best population or the best robot.
    best_robots = population[:2]  # The robots with the best fitness
    worst_robots = population[-2:]  # The robots with the worst fitness
    return best_robots, worst_robots


if __name__ == "__main__":
    genetic_algorithm()
