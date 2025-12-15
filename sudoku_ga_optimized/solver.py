import numpy as np
import random
import copy
import argparse
import sys
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, fixed_board, original_indices):
        """
        fixed_board: 9x9 array con los valores fijos y 0 en los vacios.
        But crucially, we manage 'subgrids'.
        original_indices: Lista de indices (r, c) para cada subgrid que NO son fijos.
        """
        self.genome = [] # Lista de 9 listas. Cada lista es una permutación de los valores faltantes en ese subgrid.
        self.fitness = 0.0
        
        # Inicializar genoma aleatorio válido para los subgrids
        for i in range(9):
            # Obtener valores fijos en este subgrid
            # y determinar qué valores faltan
            # (Esto se asume precalculado o pasado, pero lo haremos aquí por claridad si no es costoso)
            pass 

    def calculate_fitness(self):
        # Convertir genoma a tablero 9x9
        # Contar duplicados en filas y columnas
        pass

class SudokuSolver:
    def __init__(self, filename, population_size=200, mutation_rate=0.05):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.load_board(filename)
        self.precompute_subgrids()
        
    def load_board(self, filename):
        board = []
        with open(filename, 'r') as f:
            for line in f:
                board.append([int(x) for x in line.strip().split()])
        self.fixed_board = np.array(board)
        self.fixed_mask = self.fixed_board != 0
        
    def precompute_subgrids(self):
        """
        Identifica qué numeros son fijos en cada subgrid y qué posiciones son libres.
        """
        self.subgrid_fixed_values = [] # Lista de sets
        self.subgrid_free_indices = [] # Lista de listas de tuplas (r_global, c_global)
        
        for br in range(3):
            for bc in range(3):
                # Subgrid coordinates
                vals = set()
                indices = []
                for r in range(3):
                    for c in range(3):
                        global_r = br * 3 + r
                        global_c = bc * 3 + c
                        val = self.fixed_board[global_r][global_c]
                        if val != 0:
                            vals.add(val)
                        else:
                            indices.append((global_r, global_c))
                self.subgrid_fixed_values.append(vals)
                self.subgrid_free_indices.append(indices)

    def create_individual(self):
        ind = []
        for i in range(9):
            fixed = self.subgrid_fixed_values[i]
            # Valores que faltan en el bloque (1-9 menos los fijos)
            needed = [x for x in range(1, 10) if x not in fixed]
            random.shuffle(needed)
            ind.append(needed)
        return ind # El genoma es una lista de listas

    def get_phenotype(self, genome):
        """Reconstruye el tablero 9x9 a partir del genoma"""
        board = np.copy(self.fixed_board)
        for i in range(9): # Para cada subgrid
            values = genome[i]
            indices = self.subgrid_free_indices[i]
            for val, (r, c) in zip(values, indices):
                board[r][c] = val
        return board
        
    def calculate_fitness(self, genome):
        board = self.get_phenotype(genome)
        fitness = 0
        for i in range(9):
            row = board[i, :]
            fitness += (9 - len(np.unique(row)))
            col = board[:, i]
            fitness += (9 - len(np.unique(col)))
        return fitness

    def crossover(self, p1, p2):
        """Intercambio de subgrids completos"""
        c1 = []
        c2 = []
        for i in range(9):
            if random.random() > 0.5:
                # Intercambiar
                c1.append(p2[i][:]) # Copia profunda de la lista del bloque
                c2.append(p1[i][:])
            else:
                # Mantener
                c1.append(p1[i][:])
                c2.append(p2[i][:])
        return c1, c2

    def mutate(self, genome):
        """Intercambio (Swap) dentro de un subgrid"""
        if random.random() < self.mutation_rate:
            # Elegir un subgrid al azar
            block_idx = random.randint(0, 8)
            block = genome[block_idx]
            # Necesitamos al menos 2 huecos para hacer swap
            if len(block) >= 2:
                idx1, idx2 = random.sample(range(len(block)), 2)
                block[idx1], block[idx2] = block[idx2], block[idx1]
        return genome

    def solve(self):
        # Initial Population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitnesses = [self.calculate_fitness(ind) for ind in population]
        
        best_fitness = min(fitnesses)
        best_gen = 0
        stagnation_counter = 0
        
        history = []
        
        generation = 0
        while best_fitness > 0 and generation < 5000:
            generation += 1
            
            # Elitism: keep best
            best_idx = np.argmin(fitnesses)
            elite = copy.deepcopy(population[best_idx])
            
            # Tournament Selection
            new_pop = [elite] # Elitismo size 1
            
            while len(new_pop) < self.population_size:
                # Tournament
                candidates = random.sample(list(enumerate(fitnesses)), 3) # Size 3
                # (index, fit) -> min fit wins
                winner_idx = min(candidates, key=lambda x: x[1])[0]
                p1 = population[winner_idx]
                
                candidates = random.sample(list(enumerate(fitnesses)), 3)
                winner_idx = min(candidates, key=lambda x: x[1])[0]
                p2 = population[winner_idx]
                
                c1, c2 = self.crossover(p1, p2)
                
                # Mutate (siempre intentamos mutar con la tasa dada)
                # Aquí la tasa de mutación se aplica POR INDIVIDUO, no por gen.
                # O podemos iterar sobre los hijos.
                # El método mutate de arriba muta con probabilidad self.mutation_rate
                # Así que llamamos siempre.
                
                # Pero en mi implementacion mutate usa random() < rate DENTRO.
                # Si rate es 0.05, solo el 5% de los hijos mutan.
                # Para sudoku optimizado, a veces queremos mutacion agresiva.
                # Vamos a asegurarnos de que el loop de arriba está bien.
                self.mutate(c1)
                self.mutate(c2)
                
                new_pop.append(c1)
                if len(new_pop) < self.population_size:
                    new_pop.append(c2)
            
            population = new_pop
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            current_best = min(fitnesses)
            
            if current_best < best_fitness:
                best_fitness = current_best
                best_gen = generation
                stagnation_counter = 0
                print(f"Gen {generation}: New Best Fitness = {best_fitness}")
            else:
                stagnation_counter += 1
                if generation % 100 == 0:
                    print(f"Gen {generation}: Best = {best_fitness}")

            # Restart logic "Cataclismo"
            if stagnation_counter > 150:
                print("Stagnation detected. Triggering soft restart...")
                # Mantener elite, randomizar el resto
                population = [self.create_individual() for _ in range(self.population_size - 1)]
                population.append(elite)
                fitnesses = [self.calculate_fitness(ind) for ind in population]
                stagnation_counter = 0

            history.append(current_best)

        if best_fitness == 0:
            print(f"Solution Found in {generation} generations!")
            best_idx = np.argmin(fitnesses)
            solution_board = self.get_phenotype(population[best_idx])
            self.print_board(solution_board)
            return True, history
        else:
            print("Failed to find solution.")
            return False, history

    def print_board(self, board):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("| ", end="")
                print(f"{board[i][j]} ", end="")
            print()

if __name__ == "__main__":
    solver = SudokuSolver("../complejo.txt", population_size=1000, mutation_rate=0.4) 
    # Tasa de mutacion ALTA (0.4) porque en permutaciones un swap es un cambio pequeño.
    solver.solve()
