import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class Robot:
    def __init__(self, program=None):
        self.program = program or self.random_program()
        self.energy = 100
        self.position = np.array([random.uniform(1, 9), random.uniform(1, 9)])
        self.direction = random.uniform(0, 360)  # Degrees
        self.alive = True
        self.fitness = 0  # Default fitness to avoid AttributeError

    def random_program(self):
        return {
            "actions": random.sample(["move", "turn", "fire", "scan"], k=2),
            "conditions": random.sample(["if_enemy_near", "if_wall_close", "if_low_energy"], k=2)
        }

    def execute_program(self, state):
        # Simulate the robot's actions based on its program and the state
        actions = []
        if "if_enemy_near" in self.program["conditions"] and state["enemy_distance"] < 5:
            actions.append("fire")
        if "if_low_energy" in self.program["conditions"] and self.energy < 50:
            actions.append("move")
        if "if_wall_close" in self.program["conditions"] and state["wall_distance"] < 2:
            actions.append("turn")
        return actions

    def apply_action(self, action):
        # Apply an action to the robot
        if action == "move":
            self.position += np.array([np.cos(np.radians(self.direction)),
                                       np.sin(np.radians(self.direction))]) * 0.5
        elif action == "turn":
            self.direction = (self.direction + random.uniform(-45, 45)) % 360
        elif action == "fire":
            self.energy -= 5  # Firing costs energy

        # Check for wall collision
        self.position = np.clip(self.position, 0, 10)

    def mutate(self):
        if random.random() < 0.3:  # Incrementa la probabilidad de mutación al 30%.
            if len(self.program["actions"]) < 5:
                self.program["actions"].append(random.choice(["move", "turn", "fire", "scan"]))
            if len(self.program["conditions"]) < 5:
                self.program["conditions"].append(random.choice(["if_enemy_near", "if_wall_close", "if_low_energy"]))

    def crossover(self, partner):
        child_program = {
            "actions": random.sample(self.program["actions"], len(self.program["actions"]) // 2) +
                    random.sample(partner.program["actions"], len(partner.program["actions"]) // 2),
            "conditions": random.sample(self.program["conditions"], len(self.program["conditions"]) // 2) +
                        random.sample(partner.program["conditions"], len(partner.program["conditions"]) // 2)
        }
        return Robot(program=child_program)



class GeneticAlgorithm:
    def __init__(self, population_size=50, generations=10):
        self.population = [Robot() for _ in range(population_size)]
        self.generations = generations
        # Metrics to track
        self.best_fitness_per_generation = []
        self.worst_fitness_per_generation = []
        self.avg_fitness_per_generation = []

    def evaluate_fitness(self, robot):
        state = {
        "enemy_distance": random.uniform(0, 10),  # Distancia variable al enemigo.
        "wall_distance": random.uniform(0, 10),  # Distancia variable a las paredes.
        "energy": random.randint(10, 100)        # Energía aleatoria.
    }

        actions = robot.execute_program(state)
        fitness = len(actions) * 10

        # Bonificaciones significativas
        if "fire" in actions and state["enemy_distance"] < 5:
            fitness += 100  # Cambia de 50 a 100 para permitir valores mayores.

        if "move" in actions and state["wall_distance"] < 2:
            fitness += 20
        if "scan" in actions:
            fitness += 15
        if "turn" in actions:
            fitness += 10

        # Penalización por movimiento inseguro
        if "move" in actions and state["wall_distance"] < 2 and "turn" not in actions:
            fitness -= 20

        robot.fitness = max(fitness, 0)


    def select_parents(self):
        total_fitness = sum(r.fitness for r in self.population)
        probabilities = [r.fitness / total_fitness for r in self.population]
        return random.choices(self.population, weights=probabilities, k=10)

    def evolve_population(self):
        for generation in range(self.generations):
            for robot in self.population:
                self.evaluate_fitness(robot)
            best_fitness = max(r.fitness for r in self.population)
            avg_fitness = sum(r.fitness for r in self.population) / len(self.population)
            print(f"Generation {generation + 1}: Best: {best_fitness}, Average: {avg_fitness}")
            # Collect metrics for this generation
            fitness_values = [robot.fitness for robot in self.population]
            best_fitness = max(fitness_values)
            worst_fitness = min(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            self.best_fitness_per_generation.append(best_fitness)
            self.worst_fitness_per_generation.append(worst_fitness)
            self.avg_fitness_per_generation.append(avg_fitness)

            print(f"Generation {generation + 1}: Best: {best_fitness}, Worst: {worst_fitness}, Average: {avg_fitness:.2f}")

            # Selection, crossover, and mutation
            parents = self.select_parents()
            new_population = []
            while len(new_population) < len(self.population):
                parent1, parent2 = random.sample(parents, 2)
                child = parent1.crossover(parent2)
                if random.random() < 0.1:
                    child.mutate()
                new_population.append(child)
            self.population = new_population

    def get_best_robot(self):
        return max(self.population, key=lambda r: getattr(r, "fitness", 0))

    def plot_fitness(self):
        # Plot fitness metrics
        generations = list(range(1, self.generations + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_fitness_per_generation, label='Best Fitness', color='green')
        plt.plot(generations, self.worst_fitness_per_generation, label='Worst Fitness', color='red')
        plt.plot(generations, self.avg_fitness_per_generation, label='Average Fitness', color='blue')
        plt.title('Fitness Metrics Across Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()


def simulate_battle(robot, rival):
    """
    Simulate a battle between two robots and return the result.
    Result:
        1 -> Robot wins
        0 -> Tie
       -1 -> Rival wins
    """
    robot_fitness = robot.fitness
    rival_fitness = rival.fitness

    if robot_fitness > rival_fitness:
        return 1  # Victory
    elif robot_fitness == rival_fitness:
        return 0  # Tie
    else:
        return -1  # Loss


def test_battles(population):
    """
    Perform 30 battles for each robot against all others and record results.
    """
    results = []  # List to store results for each robot
    for robot in population:
        wins, ties, losses = 0, 0, 0
        for _ in range(30):  # Each robot participates in 30 battles
            rival = random.choice(population)  # Randomly choose a rival
            if rival != robot:  # Avoid self-battle
                result = simulate_battle(robot, rival)
                if result == 1:
                    wins += 1
                elif result == 0:
                    ties += 1
                else:
                    losses += 1
        results.append((wins, ties, losses))
    return results

def plot_fitness(self):
    generations = list(range(1, self.generations + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, self.best_fitness_per_generation, label='Best Fitness', color='green')
    plt.plot(generations, self.avg_fitness_per_generation, label='Average Fitness', color='blue')
    plt.plot(generations, self.worst_fitness_per_generation, label='Worst Fitness', color='red')
    plt.title('Fitness Metrics Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fitness_distribution(self):
    fitness_values = [robot.fitness for robot in self.population]
    plt.figure(figsize=(10, 6))
    plt.hist(fitness_values, bins=10, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Fitness Distribution in Final Generation')
    plt.xlabel('Fitness')
    plt.ylabel('Number of Robots')
    plt.grid(True)
    plt.show()


def plot_battle_results(results):
    """
    Generate a stacked bar chart showing W, T, L for each robot.
    """
    num_robots = len(results)
    x = range(num_robots)
    wins = [r[0] for r in results]
    ties = [r[1] for r in results]
    losses = [r[2] for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(x, wins, label='Wins (W)', color='green')
    plt.bar(x, ties, bottom=wins, label='Ties (T)', color='blue')
    plt.bar(x, losses, bottom=[wins[i] + ties[i] for i in range(num_robots)], label='Losses (L)', color='red')

    plt.xlabel("Robots")
    plt.ylabel("Number of Battles")
    plt.title("Battle Results for Robots in Generation 15 (30 Battles Each)")
    plt.legend()
    plt.xticks(ticks=x, labels=[f"R{i+1}" for i in x], rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=50, generations=15)
    ga.evolve_population()

    # Take the robots from generation 15 (last population)
    generation_15_population = ga.population

    # Ensure fitness is calculated for all robots in the final generation
    for robot in generation_15_population:
        ga.evaluate_fitness(robot)

    # Apply the test of 30 battles to each robot
    battle_results = test_battles(generation_15_population)

    # Plot the results
    plot_battle_results(battle_results)


     # Graficar métricas de fitness
    ga.plot_fitness()

    # Graficar distribución de fitness de la última generación
    ga.plot_fitness_distribution()

