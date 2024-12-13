import numpy as np
import random
import mido
from mido import MidiFile, MidiTrack, Message

# Definimos algunas notas como ejemplo (usando el valor MIDI de las notas)
NOTAS = [60, 62, 64, 65, 67, 69, 71, 72]  # Notas: Do, Re, Mi, Fa, Sol, La, Si, Do

# Población inicial (secuencias de notas)
def crear_poblacion(tamaño_poblacion, longitud_secuencia):
    poblacion = []
    for _ in range(tamaño_poblacion):
        secuencia = [random.choice(NOTAS) for _ in range(longitud_secuencia)]
        poblacion.append(secuencia)
    return poblacion

# Nueva función de fitness basada en los intervalos y transiciones suaves
def fitness(secuencia, longitud_secuencia):
    # Penalizar grandes saltos
    disonancia = 0
    for i in range(1, len(secuencia)):
        intervalo = abs(secuencia[i] - secuencia[i - 1])
        if intervalo > 12:  # Penalizamos intervalos mayores a una octava
            disonancia += (intervalo - 12) ** 2  # A mayor salto, mayor penalización
        else:
            disonancia += 0  # Los intervalos dentro de una octava no se penalizan
    
    # Normalizamos el fitness en el rango [0, 1]
    max_disonancia = longitud_secuencia * 12  # Máxima disonancia posible (todos los intervalos > 12)
    fitness_normalizado = 1 - (disonancia / max_disonancia)
    return fitness_normalizado

# Selección por torneo
def seleccion_torneo(poblacion, longitud_secuencia, k=3):
    seleccionados = random.sample(poblacion, k)
    mejor_individuo = max(seleccionados, key=lambda x: fitness(x, longitud_secuencia))
    return mejor_individuo

# Cruce: combina dos secuencias de notas para crear un hijo (con múltiples puntos de cruce)
def cruzamiento(padre1, padre2):
    punto_cruce_1 = random.randint(1, len(padre1) - 1)
    punto_cruce_2 = random.randint(punto_cruce_1, len(padre1) - 1)
    hijo = padre1[:punto_cruce_1] + padre2[punto_cruce_1:punto_cruce_2] + padre1[punto_cruce_2:]
    return hijo

# Mutación: Cambia una nota aleatoria con una probabilidad dada
def mutacion(secuencia, probabilidad_mutacion=0.5):  # Aumentamos la probabilidad de mutación
    if random.random() < probabilidad_mutacion:
        indice_mutacion = random.randint(0, len(secuencia) - 1)
        secuencia[indice_mutacion] = random.choice(NOTAS)
    return secuencia

# Generación de una nueva población
def nueva_generacion(poblacion, longitud_secuencia):
    nueva_poblacion = []
    for _ in range(len(poblacion)):
        # Selección por torneo
        padre1 = seleccion_torneo(poblacion, longitud_secuencia)
        padre2 = seleccion_torneo(poblacion, longitud_secuencia)
        
        hijo = cruzamiento(padre1, padre2)
        hijo_mutado = mutacion(hijo)
        nueva_poblacion.append(hijo_mutado)
    return nueva_poblacion

# Crear archivo MIDI a partir de la secuencia de notas
def crear_midi(secuencia, nombre_archivo='musica_genetica.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    for nota in secuencia:
        track.append(Message('note_on', note=nota, velocity=64, time=480))  # Inicia la nota
        track.append(Message('note_off', note=nota, velocity=64, time=480))  # Termina la nota
    
    midi.save(nombre_archivo)

# Parámetros
tamaño_poblacion = 100
longitud_secuencia = 8
generaciones = 50
fitness_objetivo = 0.95  # Definimos el fitness objetivo

# Inicializar población
poblacion = crear_poblacion(tamaño_poblacion, longitud_secuencia)

# Evolución
solucion_encontrada = False
for generacion in range(generaciones):
    poblacion = nueva_generacion(poblacion, longitud_secuencia)
    mejor_individuo = max(poblacion, key=lambda x: fitness(x, longitud_secuencia))
    mejor_fitness = fitness(mejor_individuo, longitud_secuencia)
    
    print(f"Generación {generacion + 1}, Mejor fitness: {mejor_fitness:.2f}")
    
    # Condición de parada: Si encontramos una secuencia con un fitness mayor o igual a 0.95
    if mejor_fitness >= fitness_objetivo:
        print(f"Solución óptima encontrada en la generación {generacion + 1}")
        solucion_encontrada = True
        break

# Si no encontramos una solución óptima en las 50 generaciones, terminamos
if not solucion_encontrada:
    mejor_individuo = max(poblacion, key=lambda x: fitness(x, longitud_secuencia))

# Guardar el mejor individuo como archivo MIDI
crear_midi(mejor_individuo)
print("Música generada y guardada en 'musica_genetica.mid'")
