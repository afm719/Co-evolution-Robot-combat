"""
Operations and Terminals: The program builds trees using operations and terminals. Each node represents an operation or a robot action, and these form the decision-making logic for the robot.

Node Class and Evaluation: The Node class is essential for the tree structure, and its evaluate() method recursively evaluates the value of the nodes based on the robot's state, using operations like addition, multiplication, and others.

Evolutionary Process:
    Selection: The tournament_selection() function selects the best individuals based on their fitness.
    Mutation and Crossover: These functions alter programs to introduce variation (mutate a program or mix two programs) to explore new solutions in the population.
    Elitism: Ensures that the best programs survive to the next generation.
    Fitness Calculation: Fitness is based on the health of the robots after they act, with a penalty for large health imbalances, encouraging balanced actions between the robots.

Main Function: The main() function manages the simulation and evolutionary process, generating random programs, evolving them over generations, and visualizing their performance with a plot.

This code uses a standard genetic algorithm approach to evolve a population of programs that determine the robots' actions. However, it can be further enhanced by refining the fitness function or introducing new strategies.
"""





import random
import copy
import matplotlib.pyplot as plt

# Operations and terminals represent the basic building blocks of the program.
# OPERATIONS are mathematical operations that can be applied between nodes.
# TERMINALS represent actions that robots can perform (move, shoot, repair, defend).
OPERATIONS = ['+', '-', '*', '/']
TERMINALS = ['move', 'shoot', 'repair', 'defend']

# The ROBOT_STATE holds the state of both robots, including their health and action costs.
ROBOT_STATE = {
    'robot1_health': 100,
    'robot2_health': 100,
    'move': 1,
    'shoot': 2,
    'repair': 3,
    'defend': 4
}

# Node class represents each node in the program's tree.
# It can either be an operation (like +, -, etc.) or a terminal (move, shoot, etc.).
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # The value of the node (either an operation or terminal).
        self.left = left    # Left child node (only for operations).
        self.right = right  # Right child node (only for operations).

    def evaluate(self, state):
        """
        Evaluates the node's value based on the state of the robots.
        - If it's a terminal (move, shoot, etc.), it returns a capped value from the state.
        - If it's an operation (+, -, *, /), it evaluates both children recursively.
        """
        if self.value in TERMINALS:
            return min(state[self.value], 10)  # Limit values in TERMINALS to 10 for balance.
        elif self.value in OPERATIONS:
            left = self.left.evaluate(state) if self.left else 0
            right = self.right.evaluate(state) if self.right else 1
            # Perform the corresponding operation with caps to avoid too large values.
            try:
                if self.value == '+': return min(left + right, 20)  # Limit sum to 20
                if self.value == '-': return max(left - right, -20)  # Limit difference to -20
                if self.value == '*': return min(left * right, 50)  # Limit product to 50
                if self.value == '/': return left / right if right != 0 else 1  # Avoid division by zero
            except:
                return 1  # If an error occurs, return a default value.
        return 0  # Return 0 if the node does not evaluate anything.

# The Program class represents a program, which is essentially a tree of nodes (Node objects).
# It has methods to evaluate its fitness, mutate, and perform crossover with another program.
class Program:
    def __init__(self, root):
        self.root = root  # The root node of the program's tree.

    def evaluate(self, state):
        """Evaluates the program by evaluating its root node with the current state."""
        return self.root.evaluate(state)

    def mutate(self):
        """Mutates the program by changing a random node in its tree structure."""
        return Program(mutate_node(copy.deepcopy(self.root)))

    def crossover(self, other):
        """Performs crossover between this program and another one."""
        return Program(crossover_nodes(copy.deepcopy(self.root), copy.deepcopy(other.root)))

# Functions to create random programs and random nodes (random trees).
# Programs are created with random nodes, and the depth of the tree can be controlled.
def create_random_program(max_depth):
    return Program(create_random_node(max_depth))

def create_random_node(max_depth):
    # If max_depth is 0 or randomly chosen, create a terminal node.
    if max_depth == 0 or (random.random() < 0.5 and max_depth > 0):
        return Node(random.choice(TERMINALS))
    else:
        op = random.choice(OPERATIONS)  # Choose an operation node (+, -, *, /).
        return Node(op, create_random_node(max_depth - 1), create_random_node(max_depth - 1))

# Mutation function modifies random parts of a program's tree structure.
def mutate_node(node):
    # With a 10% chance, replace the current node with a completely random subtree.
    if random.random() < 0.1:
        return create_random_node(2)  # Random subtree with depth 2.
    # If the node is an operation, recursively mutate its left or right children.
    elif node.value in OPERATIONS:
        if random.random() < 0.5 and node.left:
            node.left = mutate_node(node.left)
        if node.right:
            node.right = mutate_node(node.right)
    # If it's a terminal, randomly change it to another terminal.
    elif node.value in TERMINALS:
        node.value = random.choice(TERMINALS)
    return node

# The crossover function exchanges parts of two parent program trees to create an offspring.
def crossover_nodes(node1, node2):
    # With a 50% chance, swap the current node with the corresponding node from the other parent.
    if random.random() < 0.5:
        return copy.deepcopy(node2)
    # If both nodes are operations, perform crossover recursively on their children.
    if node1.value in OPERATIONS and node2.value in OPERATIONS:
        if random.random() < 0.5 and node1.left:
            node1.left = crossover_nodes(node1.left, node2.left or node2)
        if node1.right:
            node1.right = crossover_nodes(node1.right, node2.right or node2)
    return node1

# Evaluates the fitness of a program by simulating a fight between two robots.
# The fitness is based on the health of the robots and penalizes large health imbalances.

#The current fitness formula is based on two main factors:
# Health sum: The sum of the remaining health of both robots.
# Health difference: A penalty is applied if the health difference between the robots exceeds 20, incentivizing balanced outcomes.


def evaluate_fitness(program, state):
    state_copy = copy.deepcopy(state)  # Copy the state to avoid side effects.

    # Evaluate the actions of Robot 1 and Robot 2, adjusting their health based on the actions.
    robot1_action = program.evaluate(state_copy)
    state_copy['robot2_health'] -= robot1_action  # Robot 1 damages Robot 2.

    robot2_action = program.evaluate(state_copy)
    state_copy['robot1_health'] -= robot2_action  # Robot 2 damages Robot 1.

    # Calculate fitness based on the total health of both robots.
    health_sum = state_copy['robot1_health'] + state_copy['robot2_health']
    health_diff = abs(state_copy['robot1_health'] - state_copy['robot2_health'])

    # Apply a penalty if the health difference between robots is too large.
    if health_diff > 20:
        health_sum -= (health_diff - 20) * 2  # Penalty for large health difference.

    return health_sum  # We want the robots to remain healthy and balanced.

# Possible Improvements:
# Incorporate action efficiency: Penalize inefficient actions
# Aggressiveness/Strategy factor: Reward robots for strategically attacking or defending based on the opponent's health or situation.
# Normalize health: Normalize health values (e.g., between 0 and 1) to make the fitness function less sensitive to initial health values.
# Multi-objective approach: Combine multiple factors like damage dealt, damage received, and useful actions (repair, defend).

# Tournament selection method to choose the best program from a random subset of the population.
def tournament_selection(population, state, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda p: evaluate_fitness(p, state))

# Evolution function generates the next generation of programs.
# It uses elitism, mutation, and crossover to evolve the population.
def evolve(population, state, mutation_rate=0.2, crossover_rate=0.8, elitism=1, random_injection=0.2):
    # Sort the population based on fitness, so the best programs are at the front.
    population.sort(key=lambda p: evaluate_fitness(p, state), reverse=True)
    new_population = population[:elitism]  # Elitism: Keep the top programs.

    # Inject a few random programs to ensure genetic diversity.
    num_random = int(len(population) * random_injection)
    new_population.extend([create_random_program(max_depth=4) for _ in range(num_random)])

    # Fill the remaining spots by selecting parents and performing crossover or mutation.
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, state, tournament_size=2)
        parent2 = tournament_selection(population, state, tournament_size=2)

        if random.random() < crossover_rate:
            offspring = parent1.crossover(parent2)  # Crossover to produce offspring.
        else:
            offspring = parent1.mutate() if random.random() < mutation_rate else copy.deepcopy(parent1)
        new_population.append(offspring)

    return new_population

# Recursively prints the tree structure of a program for better readability.
def print_tree(node, depth=0):
    if node is None:
        return ""
    if node.left or node.right:  # Operator nodes have children.
        result = "  " * depth + f"Operator: {node.value}\n"
        result += print_tree(node.left, depth + 1)
        result += print_tree(node.right, depth + 1)
    else:  # Terminal nodes are leaf nodes with no children.
        result = "  " * depth + f"Value: {node.value}\n"
    return result

# The main function runs the evolutionary algorithm for a fixed number of generations.
# It tracks the fitness of the best program and plots the evolution over time.
def main():
    population_size = 100
    generations = 200

    # Initialize the population with random programs.
    population = [create_random_program(max_depth=7) for _ in range(population_size)]
    fitness_history = []

    for gen in range(generations):
        # Evolve the population to the next generation.
        population = evolve(population, ROBOT_STATE)
        best_program = max(population, key=lambda p: evaluate_fitness(p, ROBOT_STATE))
        best_fitness = evaluate_fitness(best_program, ROBOT_STATE)
        fitness_history.append(best_fitness)  # Track the best fitness over time.
        print(f"Generation {gen + 1}: Best fitness: {best_fitness}")

    # Plot the fitness evolution to visualize progress.
    plt.plot(range(1, generations + 1), fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Across Generations')
    plt.show()

    # Print the best program's tree structure for review.
    print("Best final program (tree structure):")
    print(print_tree(best_program.root))

if __name__ == "__main__":
    main()  # Run the main function when the script is executed.
