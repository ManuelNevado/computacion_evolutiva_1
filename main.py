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
        """Reads the Sudoku board from a text file."""
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
        """Prints the board to console."""
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
    """Generic Evolutionary Algorithm class."""
    def __init__(self, population_size=1000, mutation_rate=0.02, elite_size=20, crossover_rate=0.9, elitism_replacement='worst'):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.elitism_replacement = elitism_replacement
        self.population = []

    def selection(self, population, fitnesses):
        """Tournament selection (Minimization). Generic method."""
        tournament_size = 5
        selected = []
        for _ in range(len(population)):
            candidates_idx = random.sample(range(len(population)), tournament_size)
            best_idx = candidates_idx[0]
            for idx in candidates_idx[1:]:
                if fitnesses[idx] < fitnesses[best_idx]: # Minimization
                    best_idx = idx
            selected.append(population[best_idx])
        return selected

    def solve(self, generations=1000):
        """Main evolutionary loop. Generic method."""
        print(f"Starting evolution with population size {self.population_size}...")
        
        self.population = self.initialize_population()
        
        # Calculate initial fitnesses
        fitnesses = [self.calculate_fitness(ind) for ind in self.population]
        
        history = [] # Store (best_fitness, avg_fitness) tuples
        
        for gen in range(generations):
            best_fitness_current = min(fitnesses)
            best_idx_current = fitnesses.index(best_fitness_current)
            best_individual_current = copy.deepcopy(self.population[best_idx_current])
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            history.append((best_fitness_current, avg_fitness))
            
            # Monitor every 10 generations
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Fitness = {best_fitness_current:.2f} | Avg Fitness = {avg_fitness:.2f}")
                
            if best_fitness_current == 0:
                print(f"Solution found at generation {gen}!")
                return self.population[best_idx_current], history
            
            # Selection for the entire new generation
            parents = self.selection(self.population, fitnesses)
            
            new_population = []
            
            # Create new generation
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
            
            # Calculate fitness for new population
            new_fitnesses = [self.calculate_fitness(ind) for ind in new_population]
            best_fitness_new = min(new_fitnesses)
            
            # Conditional Elitism
            # If current best is better than ANY in new generation, preserve it
            if best_fitness_current < best_fitness_new:
                # Choose individual to replace
                if self.elitism_replacement == 'random':
                    replace_idx = random.randint(0, self.population_size - 1)
                else: # 'worst'
                    replace_idx = np.argmax(new_fitnesses) # Max fitness is worst in minimization
                
                new_population[replace_idx] = best_individual_current
                new_fitnesses[replace_idx] = best_fitness_current # Update fitness list too
            
            self.population = new_population
            fitnesses = new_fitnesses
            
        print("Max generations reached.")
        best_idx = np.argmin(fitnesses)
        return self.population[best_idx], history

class SudokuGA(EvolutionaryAlgorithm):
    """Sudoku specific implementation of Evolutionary Algorithm."""
    def __init__(self, initial_board, population_size=1000, mutation_rate=0.02, elite_size=20, crossover_rate=0.9, elitism_replacement='worst', crossover_type='single_point'):
        super().__init__(population_size, mutation_rate, elite_size, crossover_rate, elitism_replacement)
        self.initial_board = initial_board
        self.fixed_mask = initial_board != 0
        self.crossover_type = crossover_type
        self.domains = self.precompute_domains()

    def precompute_domains(self):
        """
        Precomputes the domain of valid values for each empty cell
        based on the initial fixed numbers.
        """
        domains = {}
        for r in range(9):
            for c in range(9):
                if not self.fixed_mask[r][c]:
                    # Start with all possible values
                    possible = set(range(1, 10))
                    
                    # Remove values present in the same row (fixed only)
                    possible -= set(self.initial_board[r, :])
                    
                    # Remove values present in the same column (fixed only)
                    possible -= set(self.initial_board[:, c])
                    
                    # Remove values present in the same subgrid (fixed only)
                    start_r, start_c = (r // 3) * 3, (c // 3) * 3
                    subgrid = self.initial_board[start_r:start_r+3, start_c:start_c+3]
                    possible -= set(subgrid.flatten())
                    
                    # If domain is empty (puzzle invalid or too constrained?), fallback to 1-9
                    if not possible:
                        possible = set(range(1, 10))
                        
                    domains[(r, c)] = list(possible)
        return domains

    def initialize_population(self):
        """Creates initial population using domain-based initialization."""
        population = []
        for _ in range(self.population_size):
            individual = np.copy(self.initial_board)
            for r in range(9):
                for c in range(9):
                    if not self.fixed_mask[r][c]:
                        # Pick a random value from the precomputed domain
                        domain = self.domains[(r, c)]
                        individual[r][c] = random.choice(domain)
            population.append(individual)
        return population

    def calculate_fitness(self, individual):
        """Calculates fitness based on the specific formula:
        Sum( (f_i + c_i + s_i) / 2 ) for all cells.
        """
        total_fitness = 0
        
        for r in range(9):
            for c in range(9):
                val = individual[r][c]
                
                # f_i: Row conflicts
                f_i = np.sum(individual[r, :] == val) - 1
                
                # c_i: Column conflicts
                c_i = np.sum(individual[:, c] == val) - 1
                
                # s_i: Subgrid conflicts
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
        """Single point crossover on flattened board."""
        # Flatten
        flat1 = parent1.flatten()
        flat2 = parent2.flatten()
        
        # Random crossover point
        point = random.randint(1, 80)
        
        # Create children
        child1_flat = np.concatenate((flat1[:point], flat2[point:]))
        child2_flat = np.concatenate((flat2[:point], flat1[point:]))
        
        # Reshape back to 9x9
        child1 = child1_flat.reshape((9, 9))
        child2 = child2_flat.reshape((9, 9))
                
        return child1, child2

    def crossover_rows(self, parent1, parent2):
        """Uniform crossover on rows. Respects row constraints."""
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        
        for i in range(9):
            if random.random() > 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
                
        return child1, child2

    def mutate(self, individual):
        """Performs mutation on each gene with a given probability.
        Uses precomputed domains to ensure new values are valid w.r.t initial constraints.
        """
        for r in range(9):
            for c in range(9):
                # Skip fixed cells
                if self.fixed_mask[r][c]:
                    continue
                
                if random.random() < self.mutation_rate:
                    current_val = individual[r][c]
                    # Choose new value from domain excluding current
                    domain = self.domains[(r, c)]
                    possible_values = [v for v in domain if v != current_val]
                    
                    if possible_values:
                        individual[r][c] = random.choice(possible_values)
                    
        return individual

import matplotlib.pyplot as plt

# ... (rest of imports)

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
        # Using reasonable parameters for convergence with population 100
        ga = SudokuGA(game.original_board, population_size=100, mutation_rate=0.01, elite_size=5, crossover_rate=0.9, elitism_replacement=args.elitism_replacement, crossover_type=args.crossover)
        solution, history = ga.solve(generations=1000)
        
        final_fitness = ga.calculate_fitness(solution)
        best_fitnesses.append(final_fitness)
        all_histories.append(history)
        
        print("\nFinal Solution:")
        game.display(solution)
        print(f"Final Fitness: {final_fitness}")

    # VAMM Calculation
    mean_best_fitness = np.mean(best_fitnesses)
    std_best_fitness = np.std(best_fitnesses)
    print(f"\nVAMM (Mean Best Fitness over {args.runs} runs): {mean_best_fitness:.2f} +/- {std_best_fitness:.2f}")
    
    # Plotting
    # We will plot the history of the first run for simplicity, or we could average them.
    # Let's plot the first run as an example, or if multiple runs, maybe the best one?
    # User asked for "plotear el valor de fitnes mejor y el valor de fitness promedio de la ejecucion".
    # Implies a single execution or a representative one. Let's plot the run with the best final fitness.
    
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
