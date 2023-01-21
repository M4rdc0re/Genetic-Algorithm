#!/usr/bin/env python
# Author: Mardcore

import Utils
import numpy as np
import json
from functools import lru_cache
import time
import random
import copy


class Individual:

    def __init__(self, num_genes, generation_type, crossover_type, mutation_type):
        self.num_genes = num_genes
        self.generation_type = generation_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.solution = self.generate(self.num_genes, self.generation_type)
        self.fitness = 0

    @staticmethod
    def generate(num_genes, generation_type):
        if generation_type == "random":
            return np.random.permutation(num_genes)

    def crossover(self, individual):
        if self.crossover_type == "two_points":
            point1 = np.random.randint(1, self.num_genes - 1)
            point2 = np.random.randint(1, self.num_genes - 1)
            while point1 == point2:
                point2 = np.random.randint(0, self.num_genes)
            if point1 > point2:
                point1, point2 = point2, point1
            child1 = Individual(self.num_genes, self.generation_type, self.crossover_type, self.mutation_type)
            child2 = Individual(individual.num_genes, individual.generation_type, individual.crossover_type,
                                individual.mutation_type)
            child1.solution = copy.deepcopy(self.solution)
            child2.solution = copy.deepcopy(individual.solution)
            child1.solution[point1:point2 + 1] = sorted(self.solution[point1:point2 + 1],
                                                        key=lambda x: individual.solution.tolist().index(x))
            child2.solution[point1:point2 + 1] = sorted(individual.solution[point1:point2 + 1],
                                                        key=lambda x: self.solution.tolist().index(x))
            return child1, child2
        elif self.crossover_type == "one_point":
            point = np.random.randint(1, self.num_genes - 1)
            child1 = Individual(self.num_genes, self.generation_type, self.crossover_type, self.mutation_type)
            child2 = Individual(individual.num_genes, individual.generation_type, individual.crossover_type,
                                individual.mutation_type)
            child1.solution = copy.deepcopy(self.solution)
            child2.solution = copy.deepcopy(individual.solution)
            child1.solution[point:] = sorted(self.solution[point:], key=lambda x: individual.solution.tolist().index(x))
            child2.solution[point:] = sorted(individual.solution[point:], key=lambda x: self.solution.tolist().index(x))
            return child1, child2

    def mutation(self):
        if self.mutation_type == "swap":
            point1 = np.random.randint(self.num_genes)
            point2 = np.random.randint(self.num_genes)
            while point1 == point2:
                point2 = np.random.randint(self.num_genes)
            self.solution[point1], self.solution[point2] = self.solution[point2], self.solution[point1]
            return self.solution
        elif self.mutation_type == "insertion":
            point1 = np.random.randint(self.num_genes)
            point2 = np.random.randint(self.num_genes)
            while point1 == point2:
                point2 = np.random.randint(self.num_genes)
            number = self.solution[point1]
            self.solution = np.delete(self.solution, point1)
            self.solution = np.insert(self.solution, point2, number)
            point3 = np.random.randint(self.num_genes)
            point4 = np.random.randint(self.num_genes)
            while point3 == point4:
                point3 = np.random.randint(self.num_genes)
            number = self.solution[point3]
            self.solution = np.delete(self.solution, point3)
            self.solution = np.insert(self.solution, point4, number)
            return self.solution

    def evaluate(self, problem):
        self.fitness = problem.evaluate(self.solution)
        return self.fitness

    def __str__(self):
        return str(self.solution)

    def __eq__(self, other):
        return isinstance(other,
                          self.__class__) and self.solution.any() == other.solution.any() and self.fitness == other.fitness

    def __lt__(self, other):
        return isinstance(other, self.__class__) and self.fitness < other.fitness


class Genetic:

    def __init__(self, population_size, num_generations, selection_type, crossover_type, crossover_probability,
                 mutation_type, mutation_probability, keep_elitism, random_state):
        self.num_individuals = population_size
        self.num_generations = num_generations
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.keep_elitism = keep_elitism
        self.random_state = random_state
        self.best_individuals = []
        self.best_fitness = []
        self.execution_time = 0
        self.final_score = 0
        self.final_solution = []

    def __call__(self, problem):
        start_time = time.perf_counter()
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        t = 0
        population = self.generate_population(problem)
        scores = self.evaluate(population, problem)
        scores = sorted(scores)
        population = sorted(population, key=lambda x: x.fitness)
        self.best_individuals = copy.deepcopy(population[:self.keep_elitism])
        self.best_fitness = scores[:self.keep_elitism]
        while t < self.num_generations:
            t += 1
            parents = self.select_population(population, scores)
            children = self.crossover(parents)
            self.mutation(children)
            self.evaluate(children, problem)
            population = self.combine(parents, children)
            population = sorted(population, key=lambda x: x.fitness)
            scores = [individual.fitness for individual in population]
            self.best_individuals = []
            self.best_fitness = []
            self.best_individuals = copy.deepcopy(population[:self.keep_elitism])
            self.best_fitness = scores[:self.keep_elitism]
        self.final_score = population[0].fitness
        self.final_solution = population[0].solution
        self.execution_time = time.perf_counter() - start_time
        return population[0]

    def generate_population(self, problem):
        population = [Individual(len(problem.parcels), "random", self.crossover_type, self.mutation_type) for _ in
                      range(self.num_individuals)]
        return population

    def select_population(self, population, scores):
        parents = []
        if self.selection_type == "fitness":
            scores = [(1 / score) for score in scores]
            total = sum(scores)
            probabilities = [score / total for score in scores]
            parents = np.random.choice(population, len(population), p=probabilities)
            return parents
        elif self.selection_type == "tournament":
            for _ in range(len(population)):
                candidates = np.random.choice(population, 6)
                candidates = sorted(candidates, key=lambda x: x.fitness)
                parents.append(copy.deepcopy(candidates[0]))
            return parents

    def crossover(self, population):
        children = []
        if len(population) % 2 != 0:
            population.pop(-1)
        for i in range(0, len(population), 2):
            if random.random() < self.crossover_probability:
                children += copy.deepcopy(population[i].crossover(population[i + 1]))
        return children

    def mutation(self, population):
        for i in range(len(population)):
            if random.random() < self.mutation_probability:
                population[i].mutation()

    def evaluate(self, population, problem):
        return [individual.evaluate(problem) for individual in population]

    def combine(self, parents, children):
        population = []
        if self.keep_elitism > 0:
            population.extend(children)
            population.extend(parents)
            best = sorted(population, key=lambda individual: individual.fitness)
            population = copy.deepcopy(self.best_individuals)
            population.extend(best[:self.num_individuals - self.keep_elitism])
            return population
        else:
            population.extend(children)
            population.extend(parents)
            population = sorted(population, key=lambda individual: individual.fitness)
            return population[:self.num_individuals]

    def print_statistics(self):
        print("Population size: " + str(self.num_individuals))
        print("Number of generations: " + str(self.num_generations))
        print("Generation type: random")
        print("Selection type: " + str(self.selection_type))
        print("Crossover type: " + str(self.crossover_type))
        print("Crossover probability: " + str(self.crossover_probability))
        print("Mutation type: " + str(self.mutation_type))
        print("Mutation probability: " + str(self.mutation_probability))
        print("Elitism: " + str(self.keep_elitism))
        print("Seed: " + str(self.random_state))
        print("----------------------------------------------------")
        print("Score: " + str(self.final_score))
        print("Time: " + str(self.execution_time))
        print("Solution: " + str(self.final_solution))


class CVRP:

    def __init__(self, filename, algorithm):
        with open(filename, 'r', encoding='utf8') as file:
            problem = json.load(file)

        self.problem = Utils.Problem(problem)
        self.algorithm = algorithm
        self.max_capacity = problem['vehicles'][0]['capacity']
        self.capacity = 0
        self.warehouse = problem['warehouse']
        self.parcels = self.parcelsdict(problem)
        self.search.cache_clear()

    def parcelsdict(self, problem):
        parcels = {}
        for parcel in problem['parcels']:
            values = (parcel["city"], parcel["weight"])
            parcels[parcel["id"]] = values
        return parcels

    def __call__(self):
        return self.algorithm(self)

    def evaluate(self, solution):
        self.capacity = 0
        max_fitness = 0
        warehouse = self.warehouse
        fitness, path = self.search(warehouse, self.parcels[solution[0]][0])
        max_fitness += fitness
        self.capacity += self.parcels[solution[0]][1]
        for i in range(0, len(solution) - 1):
            if self.parcels[solution[i]][0] == self.parcels[solution[i + 1]][0]:
                self.capacity += self.parcels[solution[i + 1]][1]
            else:
                fitness, path = self.search(self.parcels[solution[i]][0], self.parcels[solution[i + 1]][0])
                max_fitness += fitness
                for j in path:
                    if j.origin == self.warehouse or j.destination == self.warehouse:
                        self.capacity = 0
                        break
                self.capacity += self.parcels[solution[i + 1]][1]
            if self.capacity > self.max_capacity:
                self.capacity -= self.parcels[solution[i + 1]][1]
                fitness, path = self.search(self.parcels[solution[i + 1]][0], warehouse)
                max_fitness += fitness
                self.capacity = 0
                fitness, path = self.search(warehouse, self.parcels[solution[i + 1]][0])
                max_fitness += fitness
                self.capacity += self.parcels[solution[i + 1]][1]
        fitness, path = self.search(self.parcels[solution[-1]][0], warehouse)
        max_fitness += fitness
        return max_fitness

    @lru_cache(maxsize=4000)
    def search(self, departure, goal):
        self.problem.setDepartureGoal(departure, goal)
        busquedaAStar = Utils.AStar(self.problem, Utils.AStarHeuristic(self.problem))
        busquedaAStar.do_search()
        return busquedaAStar.solution_cost, busquedaAStar.solution_actions


genetic = Genetic(50, 100, "tournament", "one_point", 0.9, "insertion", 0.1, 5, 0)
problem = CVRP("example.json", genetic)
problem()
genetic.print_statistics()
print()

genetic = Genetic(50, 100, "fitness", "two_points", 0.9, "swap", 0.1, 5, 0)
small = CVRP("small.json", genetic)
small()
genetic.print_statistics()
print()

genetic = Genetic(50, 100, "tournament", "one_point", 0.9, "swap", 0.1, 5, 0)
medium = CVRP("medium.json", genetic)
medium()
genetic.print_statistics()
print()

genetic = Genetic(50, 100, "fitness", "two_points", 0.9, "insertion", 0.1, 5, 0)
large = CVRP("large.json", genetic)
large()
genetic.print_statistics()