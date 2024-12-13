import random
import numpy as np
import neat
import matplotlib.pyplot as plt

# Parámetros de la simulación
POPULATION_SIZE = 100
MAX_GENERATIONS = 200
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
NO_IMPROVEMENT_LIMIT = 10

# Tamaño de la entrada de la red neuronal
NUM_INPUTS = 5  # Ejemplo: 3 para la salud y 2 para la posición del oponente
NUM_OUTPUTS = 3  # Ejemplo: 3 acciones: mover izquierda, derecha, avanzar

# Definir el entorno del Robot Wars
class Robot:
    def __init__(self, genome=None):
        self.genome = genome
        self.fitness = 0
        self.health = 100
        self.position = (0, 0)  # Posición (x, y) del robot en el campo de batalla

    def attack(self):
        # Función de ataque basada en la salida de la red neuronal
        base_damage = np.sum(self.genome)  # Basado en el genoma
        random_factor = random.uniform(0.8, 1.2)  
        return int(base_damage * random_factor)

    def move(self, action):
        # Función de movimiento basada en las decisiones de la red neuronal
        if action == 0:  # Girar a la izquierda
            self.position = (self.position[0] - 1, self.position[1])
        elif action == 1:  # Girar a la derecha
            self.position = (self.position[0] + 1, self.position[1])
        elif action == 2:  # Avanzar
            self.position = (self.position[0], self.position[1] + 1)

    def evaluate(self, neural_network, opponent_position):
        # Evaluar el rendimiento del robot basándose en la red neuronal
        inputs = [self.health, opponent_position[0] - self.position[0], opponent_position[1] - self.position[1]]
        outputs = neural_network.activate(inputs)  # Salida de la red neuronal

        # La red neuronal decide entre tres acciones: mover a la izquierda, derecha o avanzar
        action = np.argmax(outputs)  # Decidir la acción con la mayor activación
        self.move(action)

# Función para evaluar el rendimiento de un robot
def evaluate_fitness(robot, neural_network, opponent_position):
    # La aptitud depende de la salud y el daño que haya causado
    robot.evaluate(neural_network, opponent_position)
    robot.fitness = robot.health  # Aquí puedes añadir más lógica para calcular la aptitud de manera más compleja

# Función para simular la batalla entre dos robots
def battle(robot1, robot2, neural_network1, neural_network2):
    round_number = 1
    while robot1.health > 0 and robot2.health > 0:
        print(f"Round {round_number}:")
        
        # Los robots atacan y se mueven
        damage1 = robot1.attack()
        robot2.health -= damage1
        print(f"Robot 1 attacks with {damage1} damage. Robot 2's health: {robot2.health}")

        if robot2.health <= 0:
            print("Robot 1 wins the battle!")
            break

        damage2 = robot2.attack()
        robot1.health -= damage2
        print(f"Robot 2 attacks with {damage2} damage. Robot 1's health: {robot1.health}")

        if robot1.health <= 0:
            print("Robot 2 wins the battle!")
            break
        
        # Los robots se mueven en función de las decisiones de sus redes neuronales
        robot1.evaluate(neural_network1, robot2.position)
        robot2.evaluate(neural_network2, robot1.position)

        round_number += 1

# Función para crear la configuración de NEAT sin archivo de configuración
def create_configuration():
    # Aquí se establece la configuración directamente en el código sin necesidad de un archivo externo
    config = neat.Config(
        neat.DefaultGenome,  # Tipo de genoma
        neat.DefaultReproduction,  # Tipo de reproducción
        neat.DefaultSpeciesSet,  # Tipo de conjunto de especies
        neat.DefaultStagnation,  # Criterios de estancamiento
        None  # No se especifica archivo de configuración, lo configuramos directamente
    )

    # Configuramos parámetros directamente en el código
    config.genome_config.num_inputs = NUM_INPUTS
    config.genome_config.num_outputs = NUM_OUTPUTS
    config.genome_config.num_hidden = 2  # Número de neuronas ocultas, ajusta según sea necesario

    config.reproduction_config.elitism = ELITE_COUNT
    config.reproduction_config.tournament_size = TOURNAMENT_SIZE

    return config

# Función de evaluación para NEAT
def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Crear la red neuronal a partir del genoma
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Crear un robot usando el genoma
        robot = Robot(genome=genome)

        # Crear un oponente con un genoma aleatorio para la batalla
        opponent_genome = random.choice(genomes)[1]  # Seleccionamos un oponente aleatorio
        opponent_network = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
        opponent_robot = Robot(genome=opponent_genome)

        # Simular la batalla entre el robot y su oponente
        battle(robot, opponent_robot, neural_network, opponent_network)

        # Evaluar el rendimiento del robot
        evaluate_fitness(robot, neural_network, opponent_robot.position)

        # Asignar la aptitud al genoma
        genome.fitness = robot.fitness

# Función principal que configura y ejecuta NEAT
def run_neat():
    # Crear configuración directamente
    config = create_configuration()

    # Inicializar la población de robots (genomas)
    population = neat.Population(config)

    # Agregar un reporter para ver el progreso
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(5))  # Guardar puntos de control

    # Ejecutar NEAT
    winner = population.run(evaluate_genomes, MAX_GENERATIONS)

    # Imprimir el genoma ganador
    print('\n¡El genoma ganador es!')
    print(winner)

    return winner

if __name__ == "__main__":
    run_neat()
