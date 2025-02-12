from RobotV2_1_GA_FinalAprox import genetic_algorithm, Robot
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def simulate_battles():
    # Get the best and worst robots
    best_robots, worst_robots = genetic_algorithm()
    # Battle between the best robots
    print("==================================================")
    print("Best battle:")
    with open("resultado_battle.txt", "w") as archivo:
        sys.stdout = archivo  # Cambiar la salida estándar
        Robot.battle(best_robots[0], best_robots[1])  # Ejecutar la batalla
        sys.stdout = sys.__stdout__  # Restaurar la salida estándar a la consola
    print("==================================================")

simulate_battles()


# Función para leer el archivo de texto y extraer los movimientos y daños
def leer_archivo(archivo):
    movimientos_robot1 = []
    movimientos_robot2 = []
    daños_robot1 = []
    daños_robot2 = []
    
    with open(archivo, "r") as file:
        lineas = file.readlines()
        
        # Recorremos las líneas del archivo
        for i, linea in enumerate(lineas):
            if "Robot 1 moves:" in linea:
                # Extraemos los movimientos
                movimientos_robot1.append(linea.strip().split(": ")[1])
            if "Robot 2 moves:" in linea:
                movimientos_robot2.append(linea.strip().split(": ")[1])
            if "Robot 1 attacks" in linea:
                # Extraemos el daño
                daños_robot1.append(int(linea.strip().split("with ")[1].split(" damage")[0]))
            if "Robot 2 attacks" in linea:
                # Extraemos el daño
                daños_robot2.append(int(linea.strip().split("with ")[1].split(" damage")[0]))
    
    return movimientos_robot1, movimientos_robot2, daños_robot1, daños_robot2

# Cargar los datos del archivo
movimientos_robot1, movimientos_robot2, daños_robot1, daños_robot2 = leer_archivo("resultado_battle.txt")

# Definir las posiciones iniciales de los robots
robot1_pos = np.array([0, 0])  # Posición inicial de Robot 1
robot2_pos = np.array([1, 0])  # Posición inicial de Robot 2

# Definir los límites de la pantalla
limite_x = 15
limite_y = 15

# Función para actualizar las posiciones de los robots y mostrar los daños
def update_positions(frame):
    global robot1_pos, robot2_pos
    
    # Calculamos la salud restante de los robots
    salud_robot1 = 100 - sum(daños_robot1[:frame+1])
    salud_robot2 = 100 - sum(daños_robot2[:frame+1])
    
    # Verificar si la salud de algún robot es 0 o menor
    if salud_robot1 <= 0 or salud_robot2 <= 0:
        winner = "Robot 1" if salud_robot1 > salud_robot2 else "Robot 2"
        ax.set_title(f'{winner} wins!\nGame Over')
        ani.event_source.stop()  # Detener la animación
        return

    # Actualizar las posiciones basadas en los movimientos
    if movimientos_robot1[frame] == "move forward":
        robot1_pos += np.array([1, 0])  # Movimiento hacia la derecha
    elif movimientos_robot1[frame] == "turn left":
        robot1_pos += np.array([0, 1])  # Movimiento hacia arriba
    
    if movimientos_robot2[frame] == "move forward":
        robot2_pos += np.array([1, 0])  # Movimiento hacia la derecha
    elif movimientos_robot2[frame] == "turn left":
        robot2_pos += np.array([0, 1])  # Movimiento hacia arriba
    elif movimientos_robot2[frame] == "turn right":
        robot2_pos += np.array([0, -1])  # Movimiento hacia abajo
    
    # Limitar las posiciones de los robots dentro de la pantalla
    robot1_pos = np.clip(robot1_pos, [0, 0], [limite_x, limite_y])
    robot2_pos = np.clip(robot2_pos, [0, 0], [limite_x, limite_y])

    # Limpiar y actualizar el gráfico
    ax.clear()
    ax.set_xlim(-1, limite_x + 1)
    ax.set_ylim(-1, limite_y + 1)
    ax.set_title(f'Round {frame + 1}\nRobot 1 Health: {salud_robot1}  Robot 2 Health: {salud_robot2}')
    
    ax.plot(robot1_pos[0], robot1_pos[1], 'bo', markersize=10, label="Robot 1")
    ax.plot(robot2_pos[0], robot2_pos[1], 'ro', markersize=10, label="Robot 2")
    
    ax.legend()

# Crear la figura y el eje para la animación
fig, ax = plt.subplots()

# Crear la animación
ani = animation.FuncAnimation(fig, update_positions, frames=range(len(movimientos_robot1)), interval=1000)

# Mostrar la animación
plt.show()
