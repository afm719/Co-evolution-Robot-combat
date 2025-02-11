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
POPULATION_SIZE = 100  # Moderate size for adequate evolution
GENES = 10  # Keep the gene size at 10
MAX_GENERATIONS = 410  # Ensure enough generations for evolution
MUTATION_RATE = 0.05  # Moderate mutation rate
TOURNAMENT_SIZE = 7  # Medium-sized tournament selection
CROSSOVER_RATE = 0.8  # High crossover probability
NO_IMPROVEMENT_LIMIT = 2  # Increase mutation if no improvement in 5 generations

# Possible actions for a robot
ACTIONS = ["move_forward", "move_backward", "turn_left", "turn_right", "shoot"]
 
# Robot class
class Robot:
    def __init__(self, genome):
        self.genome = genome  # List of genes defining behavior
        self.fitness = 0  # Initial robot fitness
        self.position = np.array([random.randint(0, 10), random.randint(0, 10)])  # Initial position
        self.health = 100  # Initial health
    
    def execute_actions(self, enemy_position):
        """
        Execute actions based on the robot's genome.
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
                    return True  # Successful shot
        return False

# Function to generate a random genome
def random_genome():
    return [random.randint(0, len(ACTIONS) - 1) for _ in range(GENES)]

# Tournament selection
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda x: min(x.fitness, 100))
    return winner

# Crossover between two robots
def crossover(parent1, parent2):
    split_point = random.randint(0, GENES - 1)
    child_genome = parent1.genome[:split_point] + parent2.genome[split_point:]
    return Robot(child_genome)

# Mutation
def mutate(robot, mutation_rate):
    for i in range(len(robot.genome)):
        if random.random() < mutation_rate:
            # Perform mutation with a random value
            robot.genome[i] = random.randint(0, len(ACTIONS) - 1)

    # Limit the maximum fitness value to 100 after mutation
    robot.fitness = min(robot.fitness, 100)

# Initialize the population
def initialize_population():
    return [Robot(random_genome()) for _ in range(POPULATION_SIZE)]

# Fitness evaluation (modified to improve penalty handling)
def evaluate_fitness(robot, enemy_robot):
    robot_health = robot.health
    enemy_health = enemy_robot.health
    actions_taken = 0

    for _ in range(10):  # Number of steps in the combat
        if robot.execute_actions(enemy_robot.position):
            enemy_health -= 10
            actions_taken += 1
        if enemy_robot.execute_actions(robot.position):
            robot_health -= 10

    # Calculate fitness with penalty for health
    robot.fitness = max(0, robot_health - enemy_health) + actions_taken
    # Limit the maximum fitness value to 100
    robot.fitness = min(robot.fitness, 100)
    return robot.fitness

# Adjust mutation rate more aggressively if necessary
def genetic_algorithm():
    population = initialize_population()
    best_fitness = []
    worst_fitness = []
    average_fitness = []
    diversity = []  # To store diversity
    stagnant_generations = 0
    last_best_fitness = 0
    global MUTATION_RATE

    for generation in range(MAX_GENERATIONS):
        # Evaluate the fitness of each robot
        for robot in population:
            opponent = random.choice(population)
            evaluate_fitness(robot, opponent)

        # Calculate the best, worst, and average fitness
        best_fitness_current = max(robot.fitness for robot in population)
        worst_fitness_current = min(robot.fitness for robot in population)
        average_fitness_current = np.mean([robot.fitness for robot in population])

        # Calculate diversity (standard deviation of fitness)
        diversity_current = np.std([robot.fitness for robot in population])

        best_fitness.append(best_fitness_current)
        worst_fitness.append(worst_fitness_current)
        average_fitness.append(average_fitness_current)
        diversity.append(diversity_current)

        # Adjust mutation rate if no improvement
        if best_fitness_current <= last_best_fitness:
            stagnant_generations += 1
            if stagnant_generations >= NO_IMPROVEMENT_LIMIT:
                MUTATION_RATE += 0.1  # More aggressive increase
                stagnant_generations = 0  # Reset counter
                print(f"Increasing mutation rate to: {MUTATION_RATE}")
        else:
            stagnant_generations = 0  # Reset if improvement occurs

        if best_fitness_current > last_best_fitness:
            last_best_fitness = best_fitness_current

        # Termination criterion
        if best_fitness_current == 100:
            print(f"Optimal solution found in generation {generation}, {best_fitness_current}")
            break

        # Selection, crossover, and mutation to create the new population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)

            # Determine if parents crossover based on crossover_rate
            if random.random() < CROSSOVER_RATE:
                child = crossover(parent1, parent2)  # Perform crossover
            else:
                # If no crossover, the child is a copy of one of the parents
                child = Robot(parent1.genome[:])  # Copy genome from one parent

            mutate(child, MUTATION_RATE)  # Pass mutation rate correctly
            new_population.append(child)

        population = new_population

        # Show the best fitness of each generation
        print(f"Generation {generation}: Best Fitness = {best_fitness_current}")

    # Display graphs
    # Best fitness graph
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness)
    plt.title("Best Fitness Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()

# Run the genetic algorithm
if __name__ == "__main__":
    genetic_algorithm()
