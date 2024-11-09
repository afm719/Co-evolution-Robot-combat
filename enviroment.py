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
robot1_color = (255, 0, 0)  # Default color for robot 1
robot2_color = (0, 0, 255)  # Default color for robot 2

# Font for text
font = pygame.font.Font(None, 36)

# Load music
pygame.mixer.music.load("./music/backgroundSounds.mp3")  # Replace with your file path if needed

# Actions
ACTIONS = ["move", "turn_left", "turn_right", "shoot", "dodge"]

# Robot Class with GP-based controls
class Robot:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.health = 100
        self.angle = random.uniform(0, 2 * np.pi)
        self.bullets = []
        self.genome = [random.choice(ACTIONS) for _ in range(10)]  # GP individual for actions
        self.genome_index = 0
        self.shoot_timer = 0  # Counter to manage shooting frequency

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 20)

    def draw_health(self, x_pos, y_pos):
        health_text = font.render(f"Health: {self.health}", True, self.color)
        screen.blit(health_text, (x_pos, y_pos))

    def sensors(self, opponent):
        distance = np.hypot(self.x - opponent.x, self.y - opponent.y)
        return {"distance": distance, "x": self.x, "y": self.y}

    def update_position(self):
        # Slight random adjustment in angle for dynamic movement
        self.angle += random.uniform(-0.1, 0.1)
        self.x += np.cos(self.angle) * 4
        self.y += np.sin(self.angle) * 4
        # Keep within bounds, if at the edge, adjust angle to move away
        if self.x <= 20 or self.x >= width - 20:
            self.angle = np.pi - self.angle
        if self.y <= 20 or self.y >= height - 20:
            self.angle = -self.angle

    def turn_left(self):
        self.angle -= 0.2

    def turn_right(self):
        self.angle += 0.2

    def shoot(self):
        if self.shoot_timer == 0:  # Shoot only if timer allows
            bullet_x = self.x + np.cos(self.angle) * 20
            bullet_y = self.y + np.sin(self.angle) * 20
            self.bullets.append([bullet_x, bullet_y, self.angle])
            self.shoot_timer = 30  # Reset shoot cooldown

    def dodge(self):
        dodge_angle = self.angle + random.choice([-np.pi / 2, np.pi / 2])
        self.x += np.cos(dodge_angle) * 10
        self.y += np.sin(dodge_angle) * 10
        # Boundary check after dodging
        self.x = max(20, min(width - 20, self.x))
        self.y = max(20, min(height - 20, self.y))

    def execute_action(self, action):
        if action == "move":
            self.update_position()
        elif action == "turn_left":
            self.turn_left()
        elif action == "turn_right":
            self.turn_right()
        elif action == "shoot":
            self.shoot()
        elif action == "dodge":
            self.dodge()

    def evaluate_action(self, action, opponent):
        original_x, original_y, original_angle = self.x, self.y, self.angle
        original_health = opponent.health
        self.execute_action(action)
        self.bullet_behavior(opponent)
        fitness = opponent.health - original_health
        self.x, self.y, self.angle = original_x, original_y, original_angle
        return fitness

    def select_best_action(self, opponent):
        best_action = None
        best_fitness = float("inf")
        for action in ACTIONS:
            action_fitness = self.evaluate_action(action, opponent)
            if action_fitness < best_fitness:
                best_fitness = action_fitness
                best_action = action
        return best_action

    def run_genome(self, opponent):
        if self.shoot_timer > 0:  # Decrease shoot cooldown
            self.shoot_timer -= 1
        action = self.select_best_action(opponent)
        self.execute_action(action)

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

# Fitness function to evaluate the performance of robots
def fitness(robot):
    return -robot.health

# Evolution Function to evolve the population of robots
def evolve_population(population):
    fitness_scores = [fitness(robot) for robot in population]
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    survivors = [robot for robot, score in sorted_population[:len(population) // 2]]
    new_population = []
    for _ in range(len(population)):
        parent1, parent2 = random.choice(survivors), random.choice(survivors)
        child_genome = [g1 if random.random() > 0.5 else g2 for g1, g2 in zip(parent1.genome, parent2.genome)]
        if random.random() < 0.1:
            mutation_index = random.randint(0, len(child_genome) - 1)
            child_genome[mutation_index] = random.choice(ACTIONS)
        child_robot = Robot(100, 100, parent1.color)
        child_robot.genome = child_genome
        new_population.append(child_robot)
    return new_population

# Draw the score for both robots
def draw_score(score, color1, color2):
    score_text1 = font.render(f"Robot 1 Wins: {score[0]}", True, color1)
    score_text2 = font.render(f"Robot 2 Wins: {score[1]}", True, color2)
    screen.blit(score_text1, (10, 50))
    screen.blit(score_text2, (width - 200, 50))

# Main Game Function
def main_game(robot1_color, robot2_color):
    population = [Robot(100, 100, robot1_color), Robot(700, 500, robot2_color)]
    score = [0, 0]
    generations = 10
    pygame.mixer.music.play(-1)

    for generation in range(generations):
        running = True
        clock = pygame.time.Clock()

        while running:
            screen.fill((30, 30, 30))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.mixer.music.stop()
                    pygame.quit()
                    sys.exit()

            for robot in population:
                robot.run_genome(population[1 - population.index(robot)])

            for robot in population:
                robot.bullet_behavior(population[1 - population.index(robot)])

            for i, robot in enumerate(population):
                robot.draw()
                robot.draw_health(10 if i == 0 else width - 160, 10)
                if robot.health <= 0:
                    score[1 - i] += 1
                    running = False

            draw_score(score, robot1_color, robot2_color)
            pygame.display.flip()
            clock.tick(60)

        population = evolve_population(population)

    pygame.quit()
    sys.exit()

# Start the game
main_game(robot1_color, robot2_color)
