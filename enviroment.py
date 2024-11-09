import pygame
import sys
import numpy as np
import random

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Battle")

# Colors and Parameters
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
robot1_color = (255, 0, 0)  # Color for robot 1
robot2_color = (0, 0, 255)  # Color for robot 2

# Font for text
font = pygame.font.Font(None, 36)

# Actions
ACTIONS = ["approach", "move_away", "turn_left", "turn_right", "shoot", "dodge"]

# Tree Node Class for GP
class TreeNode:
    def __init__(self, action=None, condition=None, left=None, right=None):
        self.action = action
        self.condition = condition
        self.left = left
        self.right = right

    def evaluate(self, robot, opponent):
        if self.action:
            robot.execute_action(self.action, opponent)
        elif self.condition:
            if self.condition(robot, opponent) and self.left:
                self.left.evaluate(robot, opponent)
            elif self.right:
                self.right.evaluate(robot, opponent)

# Robot Class with GP-based control trees
class Robot:
    def __init__(self, x, y, color, gp_tree=None):
        self.x = x
        self.y = y
        self.color = color
        self.health = 100
        self.angle = random.uniform(0, 2 * np.pi)
        self.bullets = []
        self.gp_tree = gp_tree or self.random_tree()
        self.shoot_timer = 0

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 20)

    def draw_health(self, x_pos, y_pos):
        health_text = font.render(f"Health: {self.health}", True, self.color)
        screen.blit(health_text, (x_pos, y_pos))

    def sensors(self, opponent):
        distance = np.hypot(self.x - opponent.x, self.y - opponent.y)
        angle_to_opponent = np.arctan2(opponent.y - self.y, opponent.x - self.x)
        angle_diff = (angle_to_opponent - self.angle + np.pi) % (2 * np.pi) - np.pi
        return {"distance": distance, "angle_diff": angle_diff, "health": self.health}

    def approach(self, opponent):
        angle_to_opponent = np.arctan2(opponent.y - self.y, opponent.x - self.x)
        self.angle = angle_to_opponent
        self.x += np.cos(self.angle) * 3
        self.y += np.sin(self.angle) * 3

    def move_away(self, opponent):
        angle_to_opponent = np.arctan2(opponent.y - self.y, opponent.x - self.x) + np.pi
        self.angle = angle_to_opponent
        self.x += np.cos(self.angle) * 4
        self.y += np.sin(self.angle) * 4

    def turn_left(self):
        self.angle -= 0.1

    def turn_right(self):
        self.angle += 0.1

    def shoot(self):
        if self.shoot_timer == 0:
            bullet_x = self.x + np.cos(self.angle) * 20
            bullet_y = self.y + np.sin(self.angle) * 20
            self.bullets.append([bullet_x, bullet_y, self.angle])
            self.shoot_timer = 30

    def dodge(self):
        dodge_angle = self.angle + random.choice([-np.pi / 2, np.pi / 2])
        self.x += np.cos(dodge_angle) * 10
        self.y += np.sin(dodge_angle) * 10

    def execute_action(self, action, opponent):
        if action == "approach":
            self.approach(opponent)
        elif action == "move_away":
            self.move_away(opponent)
        elif action == "turn_left":
            self.turn_left()
        elif action == "turn_right":
            self.turn_right()
        elif action == "shoot":
            self.shoot()
        elif action == "dodge":
            self.dodge()

    def bullet_behavior(self, opponent):
        for bullet in self.bullets[:]:
            bullet[0] += np.cos(bullet[2]) * 10
            bullet[1] += np.sin(bullet[2]) * 10
            pygame.draw.circle(screen, self.color, (int(bullet[0]), int(bullet[1])), 5)
            if np.hypot(bullet[0] - opponent.x, bullet[1] - opponent.y) < 20:
                opponent.health -= 5
                self.bullets.remove(bullet)
            elif bullet[0] < 0 or bullet[0] > width or bullet[1] < 0 or bullet[1] > height:
                self.bullets.remove(bullet)

    def random_tree(self, depth=3):
        if depth == 0 or random.random() < 0.3:
            return TreeNode(action=random.choice(ACTIONS))
        condition = lambda r, o: r.sensors(o)["distance"] < 100 and abs(r.sensors(o)["angle_diff"]) < 0.5
        left = self.random_tree(depth - 1)
        right = self.random_tree(depth - 1)
        return TreeNode(condition=condition, left=left, right=right)

    def run_tree(self, opponent):
        if self.shoot_timer > 0:
            self.shoot_timer -= 1
        self.gp_tree.evaluate(self, opponent)

# Fitness function
def fitness(robot):
    return robot.health

# Mutation and Crossover (same as previous example)
def mutate(tree, depth=3, probability=0.1):
    if tree.action and random.random() < probability:
        tree.action = random.choice(ACTIONS)
    elif tree.condition:
        if random.random() < probability:
            tree.condition = lambda r, o: r.sensors(o)["distance"] < random.randint(50, 150)
        if tree.left and depth > 0:
            mutate(tree.left, depth - 1, probability)
        if tree.right and depth > 0:
            mutate(tree.right, depth - 1, probability)

def crossover(tree1, tree2, depth=3):
    if depth == 0 or (not tree1 and not tree2):
        return tree1 if random.random() > 0.5 else tree2
    if tree1 and tree2 and tree1.condition and tree2.condition:
        if random.random() > 0.5:
            new_left = crossover(tree1.left, tree2.left, depth - 1)
            return TreeNode(condition=tree1.condition, left=new_left, right=tree1.right)
        else:
            new_right = crossover(tree1.right, tree2.right, depth - 1)
            return TreeNode(condition=tree1.condition, left=tree1.left, right=new_right)
    return tree1 if random.random() > 0.5 else tree2

def evolve_population(population):
    fitness_scores = [(robot, fitness(robot)) for robot in population]
    sorted_population = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
    survivors = [robot for robot, _ in sorted_population[:len(population) // 2]]
    new_population = []
    for _ in range(len(population)):
        parent1, parent2 = random.choice(survivors), random.choice(survivors)
        child_tree = crossover(parent1.gp_tree, parent2.gp_tree)
        if random.random() < 0.1:
            mutate(child_tree)
        child_robot = Robot(random.randint(50, width-50), random.randint(50, height-50), parent1.color, gp_tree=child_tree)
        new_population.append(child_robot)
    return new_population

# Main Game Function
def main_game(robot1_color, robot2_color):
    population = [Robot(100, 100, robot1_color), Robot(700, 500, robot2_color)]
    generations = 10
    for generation in range(generations):
        running = True
        while running:
            screen.fill((30, 30, 30))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            for robot in population:
                robot.run_tree(population[1 - population.index(robot)])
                robot.draw()
            for robot in population:
                robot.bullet_behavior(population[1 - population.index(robot)])
                robot.draw_health(10 if robot.color == robot1_color else width - 150, 10)
            if any(robot.health <= 0 for robot in population):
                running = False
            pygame.display.flip()
            pygame.time.Clock().tick(60)
        population = evolve_population(population)

# Start the game
main_game(robot1_color, robot2_color)
