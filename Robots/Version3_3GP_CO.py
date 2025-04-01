import random
import copy
import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
from tkinter import Canvas


OPERATIONS = ['+', '-', '*', '/']
TERMINALS = ['move', 'shoot', 'repair', 'defend']

ROBOT_STATE = {
    'robot1_health': 100,
    'robot2_health': 100,
    'move': 1,
    'shoot': 2,
    'repair': 3,
    'defend': 4
}



class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, state):
        if self.value in TERMINALS:
            return min(state[self.value], 10)
        elif self.value in OPERATIONS:
            left = self.left.evaluate(state) if self.left else 0
            right = self.right.evaluate(state) if self.right else 1
            try:
                if self.value == '+': return min(left + right, 20)
                if self.value == '-': return max(left - right, -20)
                if self.value == '*': return min(left * right, 50)
                if self.value == '/': return left / right if right != 0 else 1
            except:
                return 1
        return 0

class Program:
    def __init__(self, root):
        self.root = root

    def evaluate(self, state):
        return self.root.evaluate(state)

    def mutate(self):
        return Program(mutate_node(copy.deepcopy(self.root)))

    def crossover(self, other):
        return Program(crossover_nodes(copy.deepcopy(self.root), copy.deepcopy(other.root)))

def create_random_program(max_depth):
    return Program(create_random_node(max_depth))

def create_random_node(max_depth):
    if max_depth == 0 or (random.random() < 0.5 and max_depth > 0):
        return Node(random.choice(TERMINALS))
    else:
        op = random.choice(OPERATIONS)
        return Node(op, create_random_node(max_depth - 1), create_random_node(max_depth - 1))

def mutate_node(node):
    if random.random() < 0.1:
        return create_random_node(2)
    elif node.value in OPERATIONS:
        if random.random() < 0.5 and node.left:
            node.left = mutate_node(node.left)
        if node.right:
            node.right = mutate_node(node.right)
    elif node.value in TERMINALS:
        node.value = random.choice(TERMINALS)
    return node

def crossover_nodes(node1, node2):
    if random.random() < 0.5:
        return copy.deepcopy(node2)
    if node1.value in OPERATIONS and node2.value in OPERATIONS:
        if random.random() < 0.5 and node1.left:
            node1.left = crossover_nodes(node1.left, node2.left or node2)
        if node1.right:
            node1.right = crossover_nodes(node1.right, node2.right or node2)
    return node1

def evaluate_fitness(program1, program2, state):
    state_copy = copy.deepcopy(state)
    robot1_action = program1.evaluate(state_copy)
    robot2_action = program2.evaluate(state_copy)
    
    state_copy['robot2_health'] -= robot1_action
    state_copy['robot1_health'] -= robot2_action
    
    health_sum = state_copy['robot1_health'] + state_copy['robot2_health']
    health_diff = abs(state_copy['robot1_health'] - state_copy['robot2_health'])
    
    if health_diff > 20:
        health_sum -= (health_diff - 20) * 2
    
    return health_sum

def tournament_selection(population1, population2, state, tournament_size=3):
    tournament1 = random.sample(population1, tournament_size)
    tournament2 = random.sample(population2, tournament_size)
    
    best1 = max(tournament1, key=lambda p: evaluate_fitness(p, random.choice(population2), state))
    best2 = max(tournament2, key=lambda p: evaluate_fitness(random.choice(population1), p, state))
    
    return best1, best2

def evolve(population1, population2, state, mutation_rate=0.2, crossover_rate=0.7, elitism=2, random_injection=0.2):
    population1.sort(key=lambda p: evaluate_fitness(p, random.choice(population2), state), reverse=True)
    population2.sort(key=lambda p: evaluate_fitness(random.choice(population1), p, state), reverse=True)
    
    new_population1 = population1[:elitism]
    new_population2 = population2[:elitism]
    
    num_random = int(len(population1) * random_injection)
    new_population1.extend([create_random_program(4) for _ in range(num_random)])
    new_population2.extend([create_random_program(4) for _ in range(num_random)])
    
    while len(new_population1) < len(population1):
        parent1, parent2 = tournament_selection(population1, population2, state)
        
        if random.random() < crossover_rate:
            offspring1 = parent1.crossover(parent2)
            offspring2 = parent2.crossover(parent1)
        else:
            offspring1 = parent1.mutate() if random.random() < mutation_rate else copy.deepcopy(parent1)
            offspring2 = parent2.mutate() if random.random() < mutation_rate else copy.deepcopy(parent2)
        
        new_population1.append(offspring1)
        new_population2.append(offspring2)
    
    return new_population1, new_population2


def add_edges(graph, node, parent=None, pos={}, level=0, x=0, dx=1.0):
    """ Agrega nodos y aristas al grafo para la representación del árbol. """
    if node:
        graph.add_node(node.value)
        if parent:
            graph.add_edge(parent, node.value)
        pos[node.value] = (x, -level)
        dx /= 2
        add_edges(graph, node.left, node.value, pos, level+1, x-dx, dx)
        add_edges(graph, node.right, node.value, pos, level+1, x+dx, dx)
    return pos

def plot_tree(root):
    """ Dibuja el árbol sintáctico del programa. """
    graph = nx.DiGraph()
    pos = add_edges(graph, root)
    plt.figure(figsize=(8, 5))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray")
    plt.show()

def print_tree(node, level=0):

    if node:
        print("  " * level + str(node.value))  
        print_tree(node.left, level + 1)
        print_tree(node.right, level + 1)


def main():
    population_size = 100
    generations = 200
    
    population1 = [create_random_program(7) for _ in range(population_size)]
    population2 = [create_random_program(7) for _ in range(population_size)]
    fitness_history = []
    
    for gen in range(generations):
        population1, population2 = evolve(population1, population2, ROBOT_STATE)
        
        best1 = max(population1, key=lambda p: evaluate_fitness(p, random.choice(population2), ROBOT_STATE))
        best2 = max(population2, key=lambda p: evaluate_fitness(random.choice(population1), p, ROBOT_STATE))
        best_fitness = evaluate_fitness(best1, best2, ROBOT_STATE)
        fitness_history.append(best_fitness)
        
        print(f"Generation {gen + 1}: Best fitness: {best_fitness}")
    
    
    plt.plot(range(1, generations + 1), fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution with Co-Evolution')
    plt.show()
    
    print("Best program for Robot 1:")
    print(best1.evaluate(ROBOT_STATE))
    print("Best program for Robot 2:")
    print(best2.evaluate(ROBOT_STATE))

    print("\nBest program for Robot 1 (Tree Structure):")
    print_tree(best1.root)  # Imprimir el árbol del mejor programa para Robot 1

    print("\nBest program for Robot 2 (Tree Structure):")
    print_tree(best2.root)  # Imprimir el árbol del mejor programa para Robot 2

if __name__ == "__main__":
    main()
    root = tk.Tk()
    root.mainloop()
    
