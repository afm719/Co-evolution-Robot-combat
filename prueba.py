import random
import pygame

# Constants for screen and colors
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Robot class for handling genetic programming of each robot
class Robot:
    def __init__(self, program_tree=None, name="Robot", x=100, y=100, color=RED):
        # Initialize with a random program or an existing one
        self.program_tree = program_tree or self.random_program()
        self.fitness = 0
        self.name = name
        self.x = x
        self.y = y
        self.color = color
        self.energy = 100
        self.radius = 20

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

    def move(self):
        # Move the robot by a random small step
        dx, dy = random.choice([-5, 5]), random.choice([-5, 5])
        self.x = max(0, min(self.x + dx, SCREEN_WIDTH))
        self.y = max(0, min(self.y + dy, SCREEN_HEIGHT))

    def turn(self):
        # Turn the robot by a random small step
        dx, dy = random.choice([-5, 5]), random.choice([-5, 5])
        self.x = max(0, min(self.x + dx, SCREEN_WIDTH))
        self.y = max(0, min(self.y + dy, SCREEN_HEIGHT))

    def shoot(self, target):
        # Simulate shooting at another robot
        distance = ((self.x - target.x) ** 2 + (self.y - target.y) ** 2) ** 0.5
        if distance < 100:
            target.energy -= 10

    def draw(self, screen):
        # Draw the robot as a circle with energy level displayed
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        font = pygame.font.Font(None, 24)
        energy_text = font.render(f"{self.energy}", True, BLACK)
        screen.blit(energy_text, (self.x - 10, self.y - 30))

    def execute_actions(self, state, target):
        # Execute actions based on the state
        actions = self.execute_program(state)
        print(f"Actions executed by {self.name}: {actions}")
        
        if "move" in actions:
            self.move()
        if "turn" in actions:
            self.turn()
        if "shoot" in actions:
            self.shoot(target)

class BattleSimulator:
    def __init__(self, robot1, robot2):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Robot Battle Simulation")
        self.clock = pygame.time.Clock()
        self.robot1 = robot1
        self.robot2 = robot2

    def simulate_battle(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Calculate distances
            distance_x = self.robot1.x - self.robot2.x
            distance_y = self.robot1.y - self.robot2.y
            enemy_distance = (distance_x**2 + distance_y**2) ** 0.5

            # Distance to walls
            wall_distance_robot1 = min(self.robot1.x, 800 - self.robot1.x, self.robot1.y, 600 - self.robot1.y)
            wall_distance_robot2 = min(self.robot2.x, 800 - self.robot2.x, self.robot2.y, 600 - self.robot2.y)

            # Update state
            state_robot1 = {
                "enemy_distance": enemy_distance,
                "wall_distance": wall_distance_robot1,
                "energy": self.robot1.energy
            }

            state_robot2 = {
                "enemy_distance": enemy_distance,
                "wall_distance": wall_distance_robot2,
                "energy": self.robot2.energy
            }

            # Execute actions for each robot
            actions_robot1 = self.robot1.execute_program(state_robot1)
            actions_robot2 = self.robot2.execute_program(state_robot2)

            print(f"Actions executed by {self.robot1.name}: {actions_robot1}")
            print(f"Actions executed by {self.robot2.name}: {actions_robot2}")

            # Draw robots
            self.robot1.draw(self.screen)
            self.robot2.draw(self.screen)

            # Check if energy is depleted
            if self.robot1.energy <= 0 or self.robot2.energy <= 0:
                winner = self.robot1 if self.robot1.energy > 0 else self.robot2
                print(f"{winner.name} wins!")
                running = False

            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS

        pygame.quit()

if __name__ == "__main__":
    robot1 = Robot()
    robot2 = Robot()
    battle = BattleSimulator(robot1, robot2)
    battle.simulate_battle()