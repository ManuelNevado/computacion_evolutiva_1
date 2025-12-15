import numpy as np
import random
import copy
import argparse
import sys

class Sudoku:
    def __init__(self, filename):
        self.original_board = self.load_board(filename)
        self.fixed_mask = self.original_board != 0
        
    def load_board(self, filename):
        """Lee el tablero de Sudoku desde un archivo de texto."""
        try:
            board = []
            with open(filename, 'r') as f:
                for line in f:
                    row = [int(x) for x in line.strip().split()]
                    if len(row) != 9:
                        raise ValueError("Invalid row length")
                    board.append(row)
            
            if len(board) != 9:
                raise ValueError("Invalid number of rows")
                
            return np.array(board)
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def display(self, board=None):
        """Imprime el tablero en la consola."""
        if board is None:
            board = self.original_board
            
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("| ", end="")
                print(f"{board[i][j]} ", end="")
            print()

class EvolutionaryAlgorithm:
    """Clase genérica de Algoritmo Evolutivo."""
    def __init__(self, population_size=1000, mutation_rate=0.02, elite_size=20, crossover_rate=0.9, elitism_replacement='worst'):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.elitism_replacement = elitism_replacement
        self.population = []

    def selection(self, population, fitnesses):
        """Selección por torneo (Minimización). Método genérico."""
        tournament_size = 5
        selected = []
        for _ in range(len(population)):
            candidates_idx = random.sample(range(len(population)), tournament_size)
            best_idx = candidates_idx[0]
            for idx in candidates_idx[1:]:
                if fitnesses[idx] < fitnesses[best_idx]: # Minimización
                    best_idx = idx
            selected.append(population[best_idx])
        return selected

    def solve(self, generations=1000):
        """Bucle evolutivo principal. Método genérico."""
        print(f"Starting evolution with population size {self.population_size}...")
        
        self.population = self.initialize_population()
        
        # Calcular fitness inicial
        fitnesses = [self.calculate_fitness(ind) for ind in self.population]
        
        history = [] # Almacenar tuplas (mejor_fitness, avg_fitness)
        
        for gen in range(generations):
            best_fitness_current = min(fitnesses)
            best_idx_current = fitnesses.index(best_fitness_current)
            best_individual_current = copy.deepcopy(self.population[best_idx_current])
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            history.append((best_fitness_current, avg_fitness))
            
            # Monitorear cada 10 generaciones
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Fitness = {best_fitness_current:.2f} | Avg Fitness = {avg_fitness:.2f}")
                
            if best_fitness_current == 0:
                print(f"Solution found at generation {gen}!")
                return self.population[best_idx_current], history
            
            # Selección para toda la nueva generación
            parents = self.selection(self.population, fitnesses)
            
            new_population = []
            
            # Crear nueva generación
            while len(new_population) < self.population_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                
                if random.random() < self.crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = np.copy(p1), np.copy(p2)
                    
                new_population.append(self.mutate(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(c2))
            
            # Calcular fitness para la nueva población
            new_fitnesses = [self.calculate_fitness(ind) for ind in new_population]
            best_fitness_new = min(new_fitnesses)
            
            # Elitismo condicional
            # Si el mejor actual es mejor que CUALQUIERA en la nueva generación, preservarlo
            if best_fitness_current < best_fitness_new:
                # Elegir individuo a reemplazar
                if self.elitism_replacement == 'random':
                    replace_idx = random.randint(0, self.population_size - 1)
                else: # 'peor'
                    replace_idx = np.argmax(new_fitnesses) # El fitness máximo es el peor en minimización
                
                new_population[replace_idx] = best_individual_current
                new_fitnesses[replace_idx] = best_fitness_current # Actualizar también la lista de fitness
            
            self.population = new_population
            fitnesses = new_fitnesses
            
        print("Max generations reached.")
        best_idx = np.argmin(fitnesses)
        return self.population[best_idx], history

class SudokuGA(EvolutionaryAlgorithm):
    """Implementación específica para Sudoku del Algoritmo Evolutivo."""
    def __init__(self, initial_board, population_size=1000, mutation_rate=0.02, elite_size=20, crossover_rate=0.9, elitism_replacement='worst', crossover_type='single_point'):
        super().__init__(population_size, mutation_rate, elite_size, crossover_rate, elitism_replacement)
        self.initial_board = initial_board
        self.fixed_mask = initial_board != 0
        self.crossover_type = crossover_type
        self.domains = self.precompute_domains()

    def precompute_domains(self):
        """
        Precomputa el dominio de valores válidos para cada celda vacía
        basado en los números fijos iniciales.
        """
        domains = {}
        for r in range(9):
            for c in range(9):
                if not self.fixed_mask[r][c]:
                    # Empezar con todos los valores posibles
                    possible = set(range(1, 10))
                    
                    # Eliminar valores presentes en la misma fila (solo fijos)
                    possible -= set(self.initial_board[r, :])
                    
                    # Eliminar valores presentes en la misma columna (solo fijos)
                    possible -= set(self.initial_board[:, c])
                    
                    # Eliminar valores presentes en el mismo subcuadrado (solo fijos)
                    start_r, start_c = (r // 3) * 3, (c // 3) * 3
                    subgrid = self.initial_board[start_r:start_r+3, start_c:start_c+3]
                    possible -= set(subgrid.flatten())
                    
                    # Si el dominio está vacío (¿puzzle inválido o demasiado restringido?), usar 1-9 como fallback
                    if not possible:
                        possible = set(range(1, 10))
                        
                    domains[(r, c)] = list(possible)
        return domains

    def initialize_population(self):
        """Crea la población inicial usando inicialización basada en dominios."""
        population = []
        for _ in range(self.population_size):
            individual = np.copy(self.initial_board)
            for r in range(9):
                for c in range(9):
                    if not self.fixed_mask[r][c]:
                        # Elegir un valor aleatorio del dominio precomputado
                        domain = self.domains[(r, c)]
                        individual[r][c] = random.choice(domain)
            population.append(individual)
        return population

    def calculate_fitness(self, individual):
        """Calcula el fitness basado en la fórmula específica:
        Suma( (f_i + c_i + s_i) / 2 ) para todas las celdas.
        """
        total_fitness = 0
        
        for r in range(9):
            for c in range(9):
                val = individual[r][c]
                
                # f_i: Conflictos de fila
                f_i = np.sum(individual[r, :] == val) - 1
                
                # c_i: Conflictos de columna
                c_i = np.sum(individual[:, c] == val) - 1
                
                # s_i: Conflictos de subcuadrado
                start_r, start_c = (r // 3) * 3, (c // 3) * 3
                subgrid = individual[start_r:start_r+3, start_c:start_c+3]
                s_i = np.sum(subgrid == val) - 1
                
                total_fitness += (f_i + c_i + s_i)
                
        return total_fitness / 2

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'rows':
            return self.crossover_rows(parent1, parent2)
        else:
            return self.crossover_single_point(parent1, parent2)

    def crossover_single_point(self, parent1, parent2):
        """Cruce de un solo punto en el tablero aplanado."""
        # Aplanar
        flat1 = parent1.flatten()
        flat2 = parent2.flatten()
        
        # Punto de cruce aleatorio
        point = random.randint(1, 80)
        
        # Crear hijos
        child1_flat = np.concatenate((flat1[:point], flat2[point:]))
        child2_flat = np.concatenate((flat2[:point], flat1[point:]))
        
        # Remodelar de nuevo a 9x9
        child1 = child1_flat.reshape((9, 9))
        child2 = child2_flat.reshape((9, 9))
                
        return child1, child2

    def crossover_rows(self, parent1, parent2):
        """Cruce uniforme en filas. Respeta las restricciones de las filas."""
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        
        for i in range(9):
            if random.random() > 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
                
        return child1, child2

    def mutate(self, individual):
        """Realiza la mutación en cada gen con una probabilidad dada.
        Usa dominios precomputados para asegurar que los nuevos valores sean válidos con respecto a las restricciones iniciales.
        """
        for r in range(9):
            for c in range(9):
                # Omitir celdas fijas
                if self.fixed_mask[r][c]:
                    continue
                
                if random.random() < self.mutation_rate:
                    current_val = individual[r][c]
                    # Elegir un nuevo valor del dominio excluyendo el actual
                    domain = self.domains[(r, c)]
                    possible_values = [v for v in domain if v != current_val]
                    
                    if possible_values:
                        individual[r][c] = random.choice(possible_values)
                    
        return individual

import matplotlib.pyplot as plt

# ... (resto de importaciones)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sudoku Solver using Genetic Algorithm')
    parser.add_argument('filename', nargs='?', default='sencillo.txt', help='Path to Sudoku file')
    parser.add_argument('--crossover', choices=['single_point', 'rows'], default='single_point', 
                        help='Crossover type: single_point (default) or rows (respects row constraints)')
    parser.add_argument('--elitism_replacement', choices=['worst', 'random'], default='worst',
                        help='Elitism replacement strategy: worst (default) or random')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for VAMM calculation')
    
    args = parser.parse_args()
    
    print(f"Loading board from {args.filename}...")
    game = Sudoku(args.filename)
    print("Initial Board:")
    game.display()
    
    print(f"\nSolving using {args.crossover} crossover and {args.elitism_replacement} elitism replacement...")
    
    best_fitnesses = []
    all_histories = []
    
    for i in range(args.runs):
        print(f"\n--- Run {i+1}/{args.runs} ---")
        # Usando parámetros razonables para la convergencia con una población de 100
        ga = SudokuGA(game.original_board, population_size=100, mutation_rate=0.01, elite_size=5, crossover_rate=0.9, elitism_replacement=args.elitism_replacement, crossover_type=args.crossover)
        solution, history = ga.solve(generations=1000)
        
        final_fitness = ga.calculate_fitness(solution)
        best_fitnesses.append(final_fitness)
        all_histories.append(history)
        
        print("\nFinal Solution:")
        game.display(solution)
        print(f"Final Fitness: {final_fitness}")

    # Cálculo de VAMM
    mean_best_fitness = np.mean(best_fitnesses)
    std_best_fitness = np.std(best_fitnesses)
    print(f"\nVAMM (Mean Best Fitness over {args.runs} runs): {mean_best_fitness:.2f} +/- {std_best_fitness:.2f}")
    
    # Graficando
    # Graficaremos el historial de la primera ejecución por simplicidad, o podríamos promediarlos.
    # Grafiquemos la primera ejecución como ejemplo, o si hay múltiples ejecuciones, ¿quizás la mejor?
    # El usuario pidió "plotear el valor de fitness mejor y el valor de fitness promedio de la ejecucion".
    # Implica una única ejecución o una representativa. Grafiquemos la ejecución con el mejor fitness final.
    
    best_run_idx = np.argmin(best_fitnesses)
    best_history = all_histories[best_run_idx]
    
    generations = range(len(best_history))
    best_fits = [h[0] for h in best_history]
    avg_fits = [h[1] for h in best_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fits, label='Best Fitness')
    plt.plot(generations, avg_fits, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Evolution (Run {best_run_idx+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_plot.png')
    print("\nFitness plot saved to 'fitness_plot.png'")
