import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from RobotV2_1_GA import Robot  # Importa la clase Robot
import numpy as np

# Parámetros de la simulación (puedes ajustar estos valores)
MAX_GENERATIONS = 200

class RobotWarsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Wars Simulator")
        self.root.geometry("800x600")

        # Botones y Labels para la interfaz
        self.start_button = tk.Button(root, text="Start Battle", command=self.start_battle)
        self.start_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Battle Result: ", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        self.battle_output = tk.Text(root, height=10, width=80)
        self.battle_output.pack(pady=20)

        self.figure = None
        self.canvas = None

    def start_battle(self):
        # Creamos los robots y simulamos la batalla
        robot1 = Robot()
        robot2 = Robot()

        # Iniciar la batalla y obtener el log de la batalla
        battle_log = Robot.battle(robot1, robot2)

        self.battle_output.delete(1.0, tk.END)
        self.result_label.config(text="Battle Result: Started")

        # Mostrar el log de la batalla en el cuadro de texto
        for line in battle_log:
            self.battle_output.insert(tk.END, f"{line}\n")

        self.show_evolution_graph()

    def show_evolution_graph(self):
        # Generar gráficos de ejemplo (estos deben reemplazarse por los datos reales de la evolución)
        generations = list(range(MAX_GENERATIONS))
        best_fitness = np.random.random(MAX_GENERATIONS) * 100  # Ejemplo de datos
        average_fitness = np.random.random(MAX_GENERATIONS) * 100
        worst_fitness = np.random.random(MAX_GENERATIONS) * 100

        # Crear gráfico
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        ax = self.figure.add_subplot(111)
        ax.plot(generations, best_fitness, label="Best Fitness", color='green')
        ax.plot(generations, average_fitness, label="Average Fitness", color='blue')
        ax.plot(generations, worst_fitness, label="Worst Fitness", color='red')

        ax.set_title('Evolution of Fitness Over Generations')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.legend()

        # Insertar gráfico en Tkinter
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # Eliminar el gráfico anterior si lo hay
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(pady=20)
        self.canvas.draw()
