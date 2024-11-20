import random
import matplotlib.pyplot as plt
import numpy as np

class Robot:
    def __init__(self, program=None):
        self.program = program or self.random_program()
        self.energy = 100
        self.position = np.array([random.uniform(1, 9), random.uniform(1, 9)])
        self.direction = random.uniform(0, 360)  # Degrees
        self.alive = True
        self.fitness = 0
        self.memory = {"previous_state": None}  # For short-term memory

    def random_program(self):
        return {
            "actions": random.sample(["move", "turn", "fire", "scan"], k=2),
            "conditions": random.sample(["if_enemy_near", "if_wall_close", "if_low_energy"], k=2)
        }

    def execute_program(self, state):
        actions = []
        if "if_enemy_near" in self.program["conditions"] and state["enemy_distance"] < 5:
            actions.append("fire")
        if "if_low_energy" in self.program["conditions"] and self.energy < 50:
            actions.append("move")
        if "if_wall_close" in self.program["conditions"] and state["wall_distance"] < 2:
            actions.append("turn")
        return actions

    def apply_action(self, action):
        if action == "move":
            self.position += np.array([np.cos(np.radians(self.direction)),
                                       np.sin(np.radians(self.direction))]) * 0.5
        elif action == "turn":
            self.direction = (self.direction + random.uniform(-45, 45)) % 360
        elif action == "fire":
            self.energy -= 5  # Firing costs energy

        self.position = np.clip(self.position, 0, 10)

    def mutate(self):
        if random.random() < 0.3:
            if len(self.program["actions"]) < 4:
                self.program["actions"].append(random.choice(["move", "turn", "fire", "scan"]))
            if len(self.program["conditions"]) < 4:
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
    def __init__(self, population_size=50, generations=20, elitism=2):
        self.population = [Robot() for _ in range(population_size)]
        self.generations = generations
        self.elitism = elitism
        self.best_fitness_per_generation = []
        self.avg_fitness_per_generation = []
        self.worst_fitness_per_generation = []
        self.global_best_fitness = 0

    def evaluate_fitness(self, robot):
        fitness = 0
        diversity_bonus = 0.1 * len(set(robot.program["actions"]))  # Promueve diversidad
        for _ in range(10):  # Evalúa varias simulaciones
            state = {
                "enemy_distance": random.uniform(0, 10),
                "wall_distance": random.uniform(0, 10),
                "energy": robot.energy,
                "enemy_direction": random.uniform(0, 360),
            }
            actions = robot.execute_program(state)

            if "fire" in actions:
                if state["enemy_distance"] < 5:
                    fitness += 50
                else:
                    fitness -= 20  # Penalización por disparos fallidos
            if "move" in actions:
                fitness += 10
            if "turn" in actions:
                fitness += 5
            if "move" in actions and state["wall_distance"] < 2 and "turn" not in actions:
                fitness -= 30
            if robot.energy > 50:
                fitness += 10

        robot.fitness = max(fitness / 10 + diversity_bonus, 0)  # Suaviza resultados negativos

    def select_parents(self):
        tournament_size = 3
        parents = []
        for _ in range(len(self.population) - self.elitism):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda r: r.fitness)
            parents.append(winner)
        return parents

    def evolve_population(self):
        for generation in range(self.generations):
            for robot in self.population:
                self.evaluate_fitness(robot)

            fitness_values = [robot.fitness for robot in self.population]
            best_fitness = max(fitness_values)
            worst_fitness = min(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            self.global_best_fitness = max(self.global_best_fitness, best_fitness)

            self.best_fitness_per_generation.append(self.global_best_fitness)
            self.avg_fitness_per_generation.append(avg_fitness)
            self.worst_fitness_per_generation.append(worst_fitness)

            print(f"Generation {generation + 1}: Best: {self.global_best_fitness}, "
                  f"Average: {avg_fitness:.2f}, Worst: {worst_fitness}")

            sorted_population = sorted(self.population, key=lambda r: r.fitness, reverse=True)
            new_population = sorted_population[:self.elitism]

            parents = self.select_parents()

            while len(new_population) < len(self.population):
                parent1, parent2 = random.sample(parents, 2)
                child = parent1.crossover(parent2)

                if random.random() < 0.2:  # Mutación ajustada
                    child.mutate()

                new_population.append(child)

            self.population = new_population

    def plot_fitness(self):
        generations = range(1, self.generations + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_fitness_per_generation, label='Best Fitness', color='green')
        plt.plot(generations, self.avg_fitness_per_generation, label='Average Fitness', color='blue')
        plt.plot(generations, self.worst_fitness_per_generation, label='Worst Fitness', color='red')
        plt.title('Fitness Evolution Across Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=100, generations=50, elitism=5)
    ga.evolve_population()
    ga.plot_fitness()
