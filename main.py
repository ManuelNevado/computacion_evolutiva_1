import numpy as np
import random
import copy
import argparse
import sys
import datetime
import os

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
    """
    Implementación adaptada a restricciones:
    - Genotipo: Cadena de 81 enteros (Un gen por celda).
    - Representación: 'Block-Major' (Los primeros 9 genes son el Bloque 0, etc.) para minimizar el daño del cruce.
    - Cruce: Un punto estándar (puede cortar dentro de un bloque).
    """
    def __init__(self, initial_board, population_size=100, mutation_rate=0.01, elite_size=20, crossover_rate=0.9, elitism_replacement='worst', crossover_type='single_point'):
        super().__init__(population_size, mutation_rate, elite_size, crossover_rate, elitism_replacement)
        self.initial_board = initial_board
        self.fixed_mask = initial_board != 0
        self.crossover_type = crossover_type
        
        # Precomputación: Mapeo de índices Lineales (Block-Major) a Coordenadas (Row-Major)
        self.idx_to_coord = []
        self.coord_to_idx = {}
        
        idx = 0
        for b_row in range(3):
            for b_col in range(3):
                for r in range(3):
                    for c in range(3):
                        global_r = b_row * 3 + r
                        global_c = b_col * 3 + c
                        self.idx_to_coord.append((global_r, global_c))
                        self.coord_to_idx[(global_r, global_c)] = idx
                        idx += 1
                        
        self.block_fixed_data = self.precompute_fixed_blocks()

    def precompute_fixed_blocks(self):
        """Identifica valores fijos por bloque para inicialización inteligente."""
        blocks = []
        for b_idx in range(9):
            fixed_vals = set()
            # Indices en el genoma lineal que pertenecen a este bloque
            start = b_idx * 9
            end = start + 9
            
            for i in range(start, end):
                r, c = self.idx_to_coord[i]
                val = self.initial_board[r][c]
                if val != 0:
                    fixed_vals.add(val)
            
            blocks.append({
                'fixed': fixed_vals,
                'indices': list(range(start, end)),
                'needed': [v for v in range(1, 10) if v not in fixed_vals]
            })
        return blocks

    def initialize_population(self):
        """Genotipo: Lista plana de 81 enteros (Block-Major)."""
        population = []
        for _ in range(self.population_size):
            genome = [0] * 81
            
            # Rellenar bloque a bloque (Permutaciones)
            for b_idx in range(9):
                data = self.block_fixed_data[b_idx]
                needed = data['needed'][:]
                random.shuffle(needed)
                
                needed_idx = 0
                for genome_idx in data['indices']:
                    r, c = self.idx_to_coord[genome_idx]
                    if self.initial_board[r][c] != 0:
                        genome[genome_idx] = self.initial_board[r][c]
                    else:
                        genome[genome_idx] = needed[needed_idx]
                        needed_idx += 1
            
            population.append(np.array(genome)) # Usamos np.array para slicing fácil pero es 1D
        return population

    def to_phenotype(self, genome):
        """Convierte Genotipo Lineal (Block-Major) a Tablero 9x9 (Row-Major)"""
        board = np.zeros((9, 9), dtype=int)
        for i in range(81):
            r, c = self.idx_to_coord[i]
            board[r][c] = genome[i]
        return board

    def calculate_fitness(self, individual):
        """
        Fitness completo: Filas + Columnas + Bloques.
        (El cruce puede haber roto la integridad de los bloques, así que hay que verificarlos).
        """
        board = self.to_phenotype(individual)
        conflicts = 0
        
        # Filas
        for r in range(9):
            conflicts += (9 - len(np.unique(board[r, :])))
        
        # Columnas
        for c in range(9):
            conflicts += (9 - len(np.unique(board[:, c])))
            
        # Bloques (Aunque iniciamos bien, el cruce puede romperlos)
        for br in range(3):
            for bc in range(3):
                block = board[br*3:(br+1)*3, bc*3:(bc+1)*3]
                conflicts += (9 - len(np.unique(block)))
                
        return conflicts

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'rows':
             # Fallback a single point estándar si se pide rows, para simplificar
             return self.crossover_single_point(parent1, parent2)
        else:
             return self.crossover_single_point(parent1, parent2)

    def crossover_single_point(self, parent1, parent2):
        """
        Cruce estricto de UN PUNTO en la cadena de genes (1-80).
        Respeta la restricción del enunciado.
        """
        point = random.randint(1, 80)
        
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        
        return child1, child2

    def mutate(self, individual):
        """
        Mutación Swap Intra-Bloque.
        Intenta preservar la estructura de bloque si es posible.
        """
        if random.random() < self.mutation_rate:
            # Elegir un bloque al azar
            b_idx = random.randint(0, 8)
            data = self.block_fixed_data[b_idx]
            indices = data['indices']
            
            # Filtrar solo índices mutables (no fijos)
            mutable_indices = [idx for idx in indices if self.initial_board[self.idx_to_coord[idx][0]][self.idx_to_coord[idx][1]] == 0]
            
            if len(mutable_indices) >= 2:
                idx1, idx2 = random.sample(mutable_indices, 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
                    
        return individual

import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        self.terminal.flush()

    def flush(self):
        # needed for python 3 compatibility.
        self.terminal.flush()
        self.log.flush()

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
    
    # Configuración del Logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"sudoku_ga_log_{timestamp}.txt"
    sys.stdout = Logger(log_filename)
    
    print("="*60)
    print(f"Log started at: {timestamp}")
    print(f"Log file: {log_filename}")
    print("="*60)
    
    # Parámetros del experimento
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.01
    ELITE_SIZE = 5
    CROSSOVER_RATE = 0.9
    
    print("\n" + "="*20 + " EXPERIMENT CONFIGURATION " + "="*20)
    print(f"Sudoku File:          {args.filename}")
    print(f"Algorithm Runs:       {args.runs}")
    print(f"Population Size:      {POPULATION_SIZE}")
    print(f"Mutation Rate:        {MUTATION_RATE}")
    print(f"Elite Size:           {ELITE_SIZE}")
    print(f"Crossover Rate:       {CROSSOVER_RATE}")
    print(f"Crossover Type:       {args.crossover}")
    print(f"Elitism Replacement:  {args.elitism_replacement}")
    print("="*64 + "\n")

    print(f"Loading board from {args.filename}...")
    game = Sudoku(args.filename)
    print("Initial Board:")
    game.display()
    
    # Configuración del experimento de barrido
    mutation_rates = [round(x * 0.1, 1) for x in range(1, 8)] # 0.1, 0.2, ..., 0.7
    runs_per_rate = 10
    
    print(f"\nStarting Mutation Rate Sweep: {mutation_rates}")
    print(f"Runs per rate: {runs_per_rate}")
    
    overall_best_fitness = float('inf')
    overall_best_solution = None
    vamm_results = []
    
    for m_rate in mutation_rates:
        print(f"\n" + "#"*40)
        print(f"### TESTING MUTATION RATE: {m_rate} ###")
        print("#"*40)
        
        rate_best_fitnesses = []
        rate_histories = []
        
        for i in range(runs_per_rate):
            print(f"\n--- Mutation {m_rate} | Run {i+1}/{runs_per_rate} ---")
            
            ga = SudokuGA(game.original_board, 
                          population_size=POPULATION_SIZE, 
                          mutation_rate=m_rate, 
                          elite_size=ELITE_SIZE, 
                          crossover_rate=CROSSOVER_RATE, 
                          elitism_replacement=args.elitism_replacement, 
                          crossover_type=args.crossover)
                          
            solution, history = ga.solve(generations=1000)
            
            final_fitness = ga.calculate_fitness(solution)
            rate_best_fitnesses.append(final_fitness)
            rate_histories.append(history)
            
            print(f"Run Finished. Final Fitness: {final_fitness}")
            
            if final_fitness < overall_best_fitness:
                overall_best_fitness = final_fitness
                overall_best_solution = solution

        # Estadísticas por tasa de mutación (VAMM)
        mean_fitness = np.mean(rate_best_fitnesses)
        std_fitness = np.std(rate_best_fitnesses)
        vamm_entry = {
            'mutation_rate': m_rate,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness
        }
        vamm_results.append(vamm_entry)
        
        print(f"\nVAMM for Mutation Rate {m_rate}: {mean_fitness:.2f} +/- {std_fitness:.2f}")

        # Graficar para esta tasa de mutación (Mejor ejecución del lote)
        best_run_idx = np.argmin(rate_best_fitnesses)
        best_history = rate_histories[best_run_idx]
        
        generations = range(len(best_history))
        best_fits = [h[0] for h in best_history]
        avg_fits = [h[1] for h in best_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fits, label='Best Fitness')
        plt.plot(generations, avg_fits, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'Fitness Evolution (Mutation {m_rate}, Run {best_run_idx+1})')
        plt.legend()
        plt.grid(True)
        
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Nombre de archivo específico para esta tasa
        plot_filename = os.path.join(plots_dir, f'fitness_plot_mut{m_rate}_{timestamp}.png')
        plt.savefig(plot_filename)
        plt.close() 
        print(f"Plot saved to '{plot_filename}'")
        
    print(f"\nSweep Completed.")
    print(f"Overall Best Fitness Found: {overall_best_fitness}")
    
    # Guardar reporte de VAMM en archivo separado
    vamm_filename = f"vamm_summary_{timestamp}.txt"
    with open(vamm_filename, "w") as f:
        f.write("Prediction Analysis - VAMM Summary\n")
        f.write("==================================\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"{'Mutation Rate':<15} | {'Mean Best Fitness (VAMM)':<25} | {'Std Dev':<10}\n")
        f.write("-" * 55 + "\n")
        
        print("\n" + "="*20 + " VAMM SUMMARY " + "="*20)
        print(f"{'Rate':<10} | {'Mean':<10} | {'Std':<10}")
        print("-" * 36)
        
        for entry in vamm_results:
            line = f"{entry['mutation_rate']:<15} | {entry['mean_fitness']:<25.2f} | {entry['std_fitness']:<10.2f}\n"
            f.write(line)
            print(f"{entry['mutation_rate']:<10} | {entry['mean_fitness']:<10.2f} | {entry['std_fitness']:<10.2f}")
            
    print(f"\nVAMM summary saved to '{vamm_filename}'")
