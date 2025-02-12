# This code implements an evolutionary algorithm using expression trees to optimize a robot's behavior in a simulated environment. 
# The key parameters are:  

# -POPULATION_SIZE = 100 – Number of robots in the population.  
# - NUM_GENERATIONS = 200 – Total generations for evolution.  
# - MUTATION_RATE = 0.1– Probability of mutation in offspring.  
# - STAGNATION_THRESHOLD = 5 – Number of stagnant generations before increasing mutation.  

# Each robot is represented by a decision tree that evaluates its position, health, and enemy location to determine the best actions. 
# The algorithm evolves the population using:  

# - Selection:Tournament selection and elitism to keep the best individuals.  
# - Crossover: Subtree swapping between two parent trees.  
# - Mutation: Replacing subtrees for genetic diversity.  
# - Diversity Management: Adjusting mutation rates and partial population reinitialization.  

# Fitness is calculated based on distance to goal, health, enemy proximity, and tree complexity. 
# The algorithm dynamically adjusts mutation and reinitializes part of the population if diversity is low. 
# Finally, the best fitness evolution is visualized using matplotlib



import random
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import copy


POPULATION_SIZE = 100

NUM_GENERATIONS = 200
MUTATION_RATE = 0.1
STAGNATION_THRESHOLD = 5
BEST_SOLUTIONS = []

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

            distance_to_goal = abs(self.position[0] - 0.5) + abs(self.position[1] - 0.5)
            health_penalty = max(0, 1 - self.health)  
            distance_to_enemy = max(0, 1 - (abs(self.enemy_position[0] - self.position[0]) + abs(self.enemy_position[1] - self.position[1])) / 2)

            complexity_penalty = (len(tree_to_string(self.tree)) / 50.0)  


            raw_fitness = 1 - (0.4 * distance_to_goal + 0.4 * health_penalty + 0.2 * distance_to_enemy + complexity_penalty)
            
    
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

def tournament_selection(population, fitness_scores, tournament_size=5):  # Aumento del tamaño del torneo
    tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]

def elitism_selection(population, fitness_scores, elite_fraction=0.1):
    num_elites = max(1, int(len(population) * elite_fraction))
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elites = [ind for ind, _ in sorted_population[:num_elites]]
    return elites

def partial_reinitialization(population, fitness_scores, reinit_fraction=0.5):  
    num_to_reinit = int(len(population) * reinit_fraction)
    

    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    
    
    indices_to_reinit = sorted_indices[:num_to_reinit]
    
    
    for idx in indices_to_reinit:
        population[idx] = Robot(
            position=[random.uniform(0, 1), random.uniform(0, 1)],
            health=random.uniform(0, 1),
            enemy_position=[random.uniform(0, 1), random.uniform(0, 1)]
        )
        population[idx].tree = generate_random_tree()
    
    return population

def deeper_mutate(individual, mutation_rate=0.05):
    if random.random() < mutation_rate:
        
        subtree_to_mutate = random_subtree(individual.tree)
        
        
        new_subtree = generate_random_tree(depth=0, max_depth=3)  
        
      
        if subtree_to_mutate is individual.tree:  
            individual.tree = new_subtree
        else:
            replace_subtree(individual.tree, subtree_to_mutate, new_subtree)

def replace_subtree(root, old_subtree, new_subtree):
    if root is None:
        return None
    
    if root == old_subtree:
        return new_subtree  
    if root.left:
        root.left = replace_subtree(root.left, old_subtree, new_subtree)
    if root.right:
        root.right = replace_subtree(root.right, old_subtree, new_subtree)
    
    return root

def crossover_trees_with_subtrees(tree1, tree2):
    if tree1 is None or tree2 is None:
        return tree1, tree2

    if random.random() < 0.7:
        subtree1 = random_subtree(tree1)
        subtree2 = random_subtree(tree2)
        subtree1, subtree2 = subtree2, subtree1
    return tree1, tree2

def random_subtree(tree, depth=0, max_depth=5):
    if depth > max_depth or tree is None:
        return tree
    if random.random() < 0.5 and tree.left:
        return random_subtree(tree.left, depth+1)
    elif tree.right:
        return random_subtree(tree.right, depth+1)
    return tree

def calculate_population_diversity(population):
    unique_trees = set(tree_to_string(robot.tree) for robot in population)
    return len(unique_trees) / len(population)

def adjust_mutation_rate(current_rate, diversity, threshold=0.2):
    if diversity < threshold:  
        return min(0.6, current_rate * 1.5) 
    return max(0.1, current_rate * 0.8)  

def diversify_based_on_fitness(population, fitness_scores, threshold=0.01):
    std_fitness = np.std(fitness_scores)
    if std_fitness < threshold:  
        return partial_reinitialization(population, fitness_scores, reinit_fraction=0.5)
    return population


def update_best_solutions(best_individual, fitness, max_size=5):
    global BEST_SOLUTIONS
    BEST_SOLUTIONS.append((copy.deepcopy(best_individual), fitness))
    BEST_SOLUTIONS = sorted(BEST_SOLUTIONS, key=lambda x: x[1], reverse=True)[:max_size]


    

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

        print(f"Best fitness of this generation: {max_fitness:.5f}")
        fitness_over_time.append(max_fitness)
        update_best_solutions(best_individual, max_fitness)

        diversity = calculate_population_diversity(population)
        mutation_rate = adjust_mutation_rate(mutation_rate, diversity)
        print(f"Diversity: {diversity:.2f}, Mutation rate adjusted: {mutation_rate:.2f}")
        

        if stagnation_counter >= STAGNATION_THRESHOLD:
            mutation_rate = min(0.6, mutation_rate * 1.5)
            stagnation_counter = 0


        elites = elitism_selection(population, fitness_scores)
        new_population = []

        while len(new_population) < POPULATION_SIZE - len(elites):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1_tree, child2_tree = crossover_trees_with_subtrees(parent1.tree, parent2.tree)
            child1 = Robot(position=parent1.position, health=parent1.health, enemy_position=parent1.enemy_position)
            child1.tree = child1_tree
            child2 = Robot(position=parent2.position, health=parent2.health, enemy_position=parent2.enemy_position)
            child2.tree = child2_tree

            deeper_mutate(child1, mutation_rate)
            deeper_mutate(child2, mutation_rate)

            new_population.append(child1)
            new_population.append(child2)

        population = elites + new_population

        dynamic_threshold = 0.05 if generation > NUM_GENERATIONS // 2 else 0.01
        std_fitness = np.std(fitness_scores)  
        if std_fitness < dynamic_threshold:
            population = partial_reinitialization(population, fitness_scores, reinit_fraction=0.3)

        population = diversify_based_on_fitness(population, fitness_scores)
        population = partial_reinitialization(population, fitness_scores)

        if max_fitness == max(fitness_scores):
            stagnation_counter += 1
        else:
            stagnation_counter = 0

    plot_fitness(fitness_over_time)



def plot_fitness(fitness_over_time):
    plt.figure(figsize=(12, 6))
    
    plt.plot(fitness_over_time, label="Best Fitness", color='darkgreen', linestyle='--', marker='o', markersize=6, markerfacecolor='green', markeredgewidth=2)
    
    plt.title("Best Fitness Evolution Over Generations", fontsize=16, fontweight='bold', color='darkred')
    plt.xlabel("Generations", fontsize=12, fontweight='bold')
    plt.ylabel("Fitness", fontsize=12, fontweight='bold')
    
    plt.gca().set_facecolor('#f0f0f0')  
    plt.grid(True, which='both', linestyle='-.', color='gray', alpha=0.7)
    
    plt.legend(loc='upper left', fontsize=12, frameon=False, title="Métricas de Fitness", title_fontsize=13)

    plt.tight_layout()
    

    plt.show()


if __name__ == '__main__':
    run_evolution()
