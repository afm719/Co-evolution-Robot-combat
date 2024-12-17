import pygame
import re
from io import StringIO
from RobotV2_1_GA import genetic_algorithm, Robot

# Inicializar Pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Wars Battle Simulation")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Fuente
FONT = pygame.font.SysFont("Arial", 20, bold=True)

# Velocidad de actualización
FPS = 60
clock = pygame.time.Clock()

class VisualRobot:
    def __init__(self, name, color, x, y):
        self.name = name
        self.color = color
        self.x = x
        self.y = y
        self.health = 100

    def draw(self):
        """Dibuja el robot y su barra de vida."""
        pygame.draw.rect(screen, self.color, (self.x * 50, self.y * 50, 40, 40))
        health_bar_width = int(100 * (self.health / 100))
        pygame.draw.rect(screen, GREEN, (self.x * 50, self.y * 50 - 10, health_bar_width, 5))

    def move(self, direction):
        """Mueve el robot según la dirección dada."""
        if direction == "turn right" and self.x < 15:
            self.x += 1
        elif direction == "turn left" and self.x > 0:
            self.x -= 1
        elif direction == "move forward" and self.y > 0:
            self.y -= 1
        elif direction == "move backward" and self.y < 11:
            self.y += 1

def parse_battle_log(log):
    """Interpreta el output textual de la batalla."""
    rounds = []
    pattern = re.compile(
        r"Round (\d+):\n"
        r"Robot 1 attacks with (\d+) damage\. Robot 2's health: (\d+)\n"
        r"Robot 2 attacks with (\d+) damage\. Robot 1's health: (\d+)\n"
        r"Robot 1 moves: (.+)\n"
        r"Robot 2 moves: (.+)"
    )
    matches = pattern.findall(log)
    for match in matches:
        rounds.append({
            "round": int(match[0]),
            "robot1_damage": int(match[1]),
            "robot2_health": int(match[2]),
            "robot2_damage": int(match[3]),
            "robot1_health": int(match[4]),
            "robot1_move": match[5],
            "robot2_move": match[6],
        })
    return rounds

# Crear robots
best_robots, _ = genetic_algorithm()
robot1 = best_robots[0]
robot2 = best_robots[1]

# Simular la batalla y capturar el log
output_buffer = StringIO()
Robot.battle(robot1, robot2, output=output_buffer)
battle_log = output_buffer.getvalue()
parsed_log = parse_battle_log(battle_log)

# Crear visualización de los robots
visual_robot1 = VisualRobot("Robot 1", RED, 5, 10)
visual_robot2 = VisualRobot("Robot 2", BLUE, 10, 10)

current_round = 0

running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if current_round < len(parsed_log):
        # Obtener los datos del round actual
        round_data = parsed_log[current_round]
        visual_robot1.health = round_data["robot1_health"]
        visual_robot2.health = round_data["robot2_health"]

        # Mover los robots según las instrucciones
        visual_robot1.move(round_data["robot1_move"])
        visual_robot2.move(round_data["robot2_move"])

        current_round += 1

    # Dibujar el estado actual
    screen.fill(WHITE)
    visual_robot1.draw()
    visual_robot2.draw()

    # Mostrar información de la batalla
    round_text = FONT.render(f"Round: {current_round}", True, BLACK)
    screen.blit(round_text, (10, 10))

    health_text1 = FONT.render(f"Robot 1 Health: {visual_robot1.health}", True, BLACK)
    health_text2 = FONT.render(f"Robot 2 Health: {visual_robot2.health}", True, BLACK)
    screen.blit(health_text1, (10, 40))
    screen.blit(health_text2, (10, 70))

    pygame.display.flip()

pygame.quit()
