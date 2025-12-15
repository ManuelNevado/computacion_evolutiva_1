import numpy as np
import random
import argparse
import sys
from main import Sudoku, EvolutionaryAlgorithm

class OptimizedSudokuGA(EvolutionaryAlgorithm):
    """
    Optimized Sudoku GA that enforces row constraints.
    Search space is reduced by ensuring every row is always a valid permutation of 1-9.
    """
    def __init__(self, initial_board, population_size=1000, mutation_rate=0.02, elite_size=20):
        super().__init__(population_size, mutation_rate, elite_size)
        self.initial_board = initial_board
        self.fixed_mask = initial_board != 0

    def initialize_population(self):
        """Creates initial population where each row is a random permutation of missing numbers."""
        population = []
        for _ in range(self.population_size):
            individual = np.copy(self.initial_board)
            for i in range(9):
                # Find missing numbers in the row
                row = individual[i]
                missing = [n for n in range(1, 10) if n not in row]
                random.shuffle(missing)
                
                # Fill empty cells with the shuffled missing numbers
                missing_idx = 0
                for j in range(9):
                    if individual[i][j] == 0:
                        individual[i][j] = missing[missing_idx]
                        missing_idx += 1
            population.append(individual)
        return population

    def calculate_fitness(self, individual):
        """
        Calculates fitness based on Column and Subgrid conflicts only.
        Row conflicts are guaranteed to be 0 by representation.
        Formula: Sum( (c_i + s_i) ) for all cells.
        """
        total_fitness = 0
        
        for r in range(9):
            for c in range(9):
                val = individual[r][c]
                
                # c_i: Column conflicts
                c_i = np.sum(individual[:, c] == val) - 1
                
                # s_i: Subgrid conflicts
                start_r, start_c = (r // 3) * 3, (c // 3) * 3
                subgrid = individual[start_r:start_r+3, start_c:start_c+3]
                s_i = np.sum(subgrid == val) - 1
                
                total_fitness += (c_i + s_i)
                
        return total_fitness

    def crossover(self, parent1, parent2):
        """
        Row-based crossover. Swaps entire rows between parents.
        Preserves the permutation property of rows.
        """
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        
        for i in range(9):
            if random.random() > 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
                
        return child1, child2

    def mutate(self, individual):
        """
        Swap mutation. Swaps two mutable numbers within the same row.
        Preserves the permutation property of rows.
        """
        for r in range(9):
            if random.random() < self.mutation_rate:
                # Find mutable positions in this row
                mutable_indices = [c for c in range(9) if not self.fixed_mask[r][c]]
                
                # Need at least 2 mutable cells to swap
                if len(mutable_indices) >= 2:
                    idx1, idx2 = random.sample(mutable_indices, 2)
                    # Swap
                    individual[r][idx1], individual[r][idx2] = individual[r][idx2], individual[r][idx1]
                    
        return individual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Sudoku Solver (Reduced Search Space)')
    parser.add_argument('filename', nargs='?', default='sencillo.txt', help='Path to Sudoku file')
    
    args = parser.parse_args()
    
    print(f"Loading board from {args.filename}...")
    game = Sudoku(args.filename)
    print("Initial Board:")
    game.display()
    
    print("\nSolving with Optimized GA (Row Permutations)...")
    # Note: Mutation rate applies per row now, so 0.02 might be too low if we interpret it as prob of row mutation.
    # But since we iterate 9 rows, 0.2-0.5 might be better. Let's try 0.5 for row mutation probability.
    ga = OptimizedSudokuGA(game.original_board, population_size=1000, mutation_rate=0.5, elite_size=20)
    solution = ga.solve(generations=1000)
    
    print("\nFinal Solution:")
    game.display(solution)
    
    print(f"\nFinal Fitness: {ga.calculate_fitness(solution)}")
