import random

class Robot:
    def __init__(self, program_tree=None):
        # Initialize with a random program or use an existing one
        self.program_tree = program_tree or self.random_program()
        self.fitness = 0

    def random_program(self):
        # Generate a random program
        return {
            "actions": random.sample(["move", "turn", "shoot", "scan"], k=2),
            "strength": random.choice(["low", "medium", "high"]),
            "conditions": random.sample(["if_enemy_near", "if_energy_low", "if_wall_close"], k=2)
        }

    def execute_program(self, state):
        # Simulate executing actions and generating results
        actions = []
        if "if_enemy_near" in self.program_tree["conditions"] and state["enemy_distance"] < 50:
            actions.append("shoot")
        if "if_wall_close" in self.program_tree["conditions"] and state["wall_distance"] < 20:
            actions.append("turn")
        if "if_energy_low" in self.program_tree["conditions"] and state["energy"] < 30:
            actions.append("move")
        return actions

    def mutate(self):
        # Mutation: randomly change an action or condition
        if random.random() < 0.5:
            self.program_tree["actions"].append(random.choice(["move", "turn", "shoot", "scan"]))
        else:
            self.program_tree["conditions"].append(random.choice(["if_enemy_near", "if_energy_low", "if_wall_close"]))

    def crossover(self, partner):
        # Crossover with another robot
        child_program = {
            "actions": self.program_tree["actions"][:len(self.program_tree["actions"])//2] +
                       partner.program_tree["actions"][len(partner.program_tree["actions"])//2:],
            "strength": random.choice([self.program_tree["strength"], partner.program_tree["strength"]]),
            "conditions": random.choice([self.program_tree["conditions"], partner.program_tree["conditions"]])
        }
        return Robot(program_tree=child_program)


class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=10):
        self.population = [Robot() for _ in range(population_size)]
        self.generations = generations
        self.best_overall_robot = None  # Track the best robot across all generations

    def evaluate_fitness(self, robot):
        # Placeholder for battle fitness calculation
        simulated_state = {
            "enemy_distance": random.randint(0, 100),
            "wall_distance": random.randint(0, 50),
            "energy": random.randint(0, 100)
        }
        actions = robot.execute_program(simulated_state)
        
        # Assign fitness based on simulated actions (example rules)
        robot.fitness = len(actions) * 10
        if "shoot" in actions and simulated_state["enemy_distance"] < 30:
            robot.fitness += 20
        if "turn" in actions and simulated_state["wall_distance"] < 15:
            robot.fitness += 15

    def select_parents(self):
        return sorted(self.population, key=lambda r: r.fitness, reverse=True)[:10]

    def evolve_population(self):
        for generation in range(self.generations):
            for robot in self.population:
                self.evaluate_fitness(robot)

            # Track the best robot in this generation
            best_in_generation = max(self.population, key=lambda r: r.fitness)
            if not self.best_overall_robot or best_in_generation.fitness > self.best_overall_robot.fitness:
                self.best_overall_robot = best_in_generation  # Update the best overall robot

            # Log fitness details
            fitness_values = [robot.fitness for robot in self.population]
            print(f"Generation {generation+1} - Max Fitness: {max(fitness_values)}, Avg Fitness: {sum(fitness_values) / len(fitness_values)}")

            # Selection, crossover, mutation
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
        # Return the best robot tracked across all generations
        print("Best Robot Program:", self.best_overall_robot.program_tree)
        print("Best Fitness:", self.best_overall_robot.fitness)
        return self.best_overall_robot

# Running the Genetic Algorithm
if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=50, generations=10)
    ga.evolve_population()
    best_robot = ga.get_best_robot()


