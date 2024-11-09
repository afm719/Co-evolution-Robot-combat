import pygame
import sys

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Battle Configuration")

# Fonts and Colors
font = pygame.font.Font(None, 36)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
robot1_color = (255, 0, 0)  # Default red
robot2_color = (0, 0, 255)  # Default blue
health = 100  # Default health level

# Available colors
colors = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
}

def draw_text(text, x, y, color=BLACK):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def display_configuration():
    screen.fill(WHITE)
    draw_text("Robot Battle Configuration", 250, 50)

    # Health Configuration
    draw_text("Health Level:", 100, 150)
    draw_text(f"{health}", 250, 150)

    # Robot 1 Color Configuration
    draw_text("Robot 1 Color:", 100, 250)
    pygame.draw.circle(screen, robot1_color, (250, 260), 20)

    # Robot 2 Color Configuration
    draw_text("Robot 2 Color:", 100, 350)
    pygame.draw.circle(screen, robot2_color, (250, 360), 20)

    # Start Button
    pygame.draw.rect(screen, BLACK, (320, 450, 160, 50))
    draw_text("Start", 360, 460, WHITE)

    pygame.display.flip()

def main_game(robot1_color, robot2_color, health):
    # This function starts the main game interface after configuration
    # Code for the main battle game goes here; refer to the previous game setup code.
    # Pass robot1_color, robot2_color, and health to initialize the robots.
    pass

# Main loop for configuration screen
running = True
while running:
    display_configuration()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Handle key events for adjusting health level
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and health < 200:
                health += 10  # Increase health
            elif event.key == pygame.K_DOWN and health > 10:
                health -= 10  # Decrease health

        # Handle mouse events for selecting colors and starting the game
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos

            # Check if clicking within color selection areas
            if 100 < mouse_x < 300:
                if 250 < mouse_y < 270:
                    # Robot 1 color selection
                    robot1_color = colors[list(colors.keys())[(list(colors.values()).index(robot1_color) + 1) % len(colors)]]
                elif 350 < mouse_y < 370:
                    # Robot 2 color selection
                    robot2_color = colors[list(colors.keys())[(list(colors.values()).index(robot2_color) + 1) % len(colors)]]

            # Check if clicking the start button
            if 320 < mouse_x < 480 and 450 < mouse_y < 500:
                running = False
                main_game(robot1_color, robot2_color, health)  # Start game with selected configuration

pygame.quit()
