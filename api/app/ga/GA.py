"""
This module implements the GeneticAlgorithm class used to predict NBA games.
"""
import json
import math
import random
import statistics as stats

import pandas as pd
import numpy as np
from typing import Tuple, List

from ga.utils import random_weighted_sample_no_replacement


class GAPredictor:
    def __init__(
            self,
            population_size: int,
            bits_per_parameter: int=8,
            crossover_rate: float=0.5,
            mutation_rate: float=0.1,
            home_win_rate: float=0.6,
            num_offspring: int=50,
            n_points: int=3,
            cross_over_type: str="n-points",
            fitness_threshold: int=10,
            tolerance_threshold: float = 0.3,
            tolerance: int = 20):
        
        self.population_size = population_size
        self.bits_per_parameter = bits_per_parameter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.home_win_rate = home_win_rate
        self.num_offspring = num_offspring
        self.n_points = n_points
        self.cross_over_type = cross_over_type
        self.fitness_threshold = fitness_threshold
        self.tolerance_threshold = tolerance_threshold
        self.tolerance = tolerance

        self.stagnant_generations = 0
        self.initial_crossover_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate

        self.num_teams = 30
        self.num_parameters = 4
        self.solution = None
        self.initialize_population()

    def generate_binary_string(self, length: int) -> str:
        """Returns a random binary string."""
        return "".join(random.choice("01") for _ in range(length))

    def generate_individual(self) -> str:
        """Returns random individual."""
        individual = ""
        
        for _ in range(self.num_teams):
            parameters = []
            for _ in range(self.num_parameters):
                random_param = self.generate_binary_string(self.bits_per_parameter)
                parameters.append(random_param)
            
            individual += "".join(parameters)

        return individual
    
    def initialize_population(self):
        """Generate random population."""
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    def sigmoid(self, x) -> float:
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def extract_team_info(
            self,
            team_index: int,
            individual: str) -> dict:
        """Extract a team"s stats on and individual."""
        BITS_PER_TEAM = self.num_parameters * self.bits_per_parameter
        
        start_pos = team_index * BITS_PER_TEAM
        end_pos = start_pos + BITS_PER_TEAM
        
        team_info = individual[start_pos:end_pos]
        
        ah = team_info[:self.bits_per_parameter]
        aa = team_info[self.bits_per_parameter:2 * self.bits_per_parameter]
        dh = team_info[2 * self.bits_per_parameter:3 * self.bits_per_parameter]
        da = team_info[3 * self.bits_per_parameter:4 * self.bits_per_parameter]
        
        ah_decimal = int(ah, 2)
        aa_decimal = int(aa, 2)
        dh_decimal = int(dh, 2)
        da_decimal = int(da, 2)
        
        team_info_dict = {
            "ah": ah_decimal,
            "aa": aa_decimal,
            "dh": dh_decimal,
            "da": da_decimal,
        }
        
        return team_info_dict
    
    def predict_proba(
            self,
            team_h: dict,
            team_a: dict) -> float:
        """Predict the probability of home team win."""
        dh_a = team_a["dh"]
        da_a = team_a["da"]

        ah_h = team_h["ah"]
        aa_h = team_h["aa"]

        sa = self.sigmoid((ah_h - da_a + 1) / 2)
        sb = self.sigmoid((dh_a - aa_h + 1) / 2)

        d = (sa - sb + 1) / 2

        return d

    def get_season_weight(self, games: pd.DataFrame) -> int:
        """
        Returns the difference between
        the max season and the training season.
        """
        with open("./ga/data/seasons.json", "r") as json_file:
            seasons = json.load(json_file)

        curr_season = games["SEASON"].min()
        season_distance = max(seasons) - int(curr_season) + 1

        return season_distance
    
    def calculate_fitness(
            self,
            individual: str,
            traning_games: pd.DataFrame) -> float:
        """Compute the fitness of an individual based on training games."""
        errors = []
        for _, row in traning_games.iterrows():
            home_id = row["TEAM_HOME"]
            away_id = row["TEAM_AWAY"]
            y_true = row["TARGET"]

            team_h = self.extract_team_info(home_id, individual)
            team_a = self.extract_team_info(away_id, individual)

            proba = self.predict_proba(team_h, team_a)
            y_pred = int(proba > self.home_win_rate)

            diff = abs(y_pred - y_true)
            errors.append(diff)

        accumulated_fitness = sum(errors)
        season_weight = self.get_season_weight(traning_games)

        return accumulated_fitness / season_weight
    
    def parent_selection(
            self,
            fitness_scores, num_parents=2) -> List[str]:
        """Select N parents based on roulette-wheel.

        Here is the reference for the roulette
        wheel selection implemented with a binary heap.
        Reference: https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights 
        """
        items = list(zip(fitness_scores, self.population))

        selected_parents = list(random_weighted_sample_no_replacement(
            items,
            num_parents
        ))

        return selected_parents
    
    def single_point_crossover(
            self,
            parent0: str,
            parent1: str,
            cross_point: int) -> Tuple[str, str]:
        """
        Generate random children crossing-over
        multiple parts of the parents.
        """
        copy_0 = parent0[:]
        copy_1 = parent1[:]

        if np.random.random() > self.crossover_rate:
            return copy_0, copy_1
        
        child0 = copy_0[:cross_point] + copy_1[cross_point:]
        child1 = copy_1[:cross_point] + copy_0[cross_point:]
        return child0, child1
    
    def multipoint_crossover(
            self,
            parent0: str,
            parent1: str,
            n_points: int=2) -> Tuple[str, str]:
        """
        Generate random children crossing-over
        multiple parts of the parents.
        """
        copy_0 = parent0[:]
        copy_1 = parent1[:]

        if np.random.random() > self.crossover_rate:
            return copy_0, copy_1
        
        for i in range(n_points):
            copy_0, copy_1 = self.single_point_crossover(
                copy_0, copy_1, i
            )

        return copy_0, copy_1

    def uniform_crossover(self, parent0: str, parent1: str):
        """
        Generate random children crossing-over parents" genes uniformly.
        """
        assert len(parent0) == len(parent1), "Parents must have the same length."

        child1 = ""
        child2 = ""

        for gene1, gene2 in zip(parent0, parent1):
            if random.random() < self.crossover_rate:
                child1 += gene1
                child2 += gene2
            else:
                child1 += gene2
                child2 += gene1

        return child1, child2

    def cross_over(
            self,
            parent0: str,
            parent1: str) -> Tuple[str, str]:
        """Return 2 children crossing-over 2 parents."""
        if self.cross_over_type == "n-points":
            return self.multipoint_crossover(parent0, parent1, n_points=self.n_points)
        elif self.cross_over_type == "uniform":
            return self.uniform_crossover(parent0, parent1)

    def mutation(self, individual: str) -> str:
        """Mutate individual by randomly flipping the gene-bit."""
        mutated_individual = ""

        for gene in individual:
            if np.random.random() <= self.mutation_rate:
                mutated_gene = "0" if gene == "1" else "1"
            else:
                mutated_gene = gene

            mutated_individual += mutated_gene

        return mutated_individual
    
    def select_survivors(self, offspring, games: pd.DataFrame):
        """Select offspring suvivors based on elitism."""
        combined_population = self.population + offspring

        orderned_combined = sorted(
            combined_population,
            key=lambda x: self.calculate_fitness(x, games))

        return orderned_combined[:self.population_size]
    
    def fit(
            self,
            games: pd.DataFrame,
            max_generations: int=500,
            verbose: bool=True):
        fitness_scores = list(map(lambda x: self.calculate_fitness(x, games), self.population))
        mean_fitness = stats.mean(fitness_scores)
        best_fitness = min(fitness_scores)

        mean_fitness_per_generation = [mean_fitness]
        best_fitness_per_generation = [best_fitness]

        generations = 0
        is_timeout = (generations >= max_generations)
        is_solution = best_fitness <= self.fitness_threshold

        previous_best_fitness = float("inf")
        while not is_timeout and not is_solution:
            offspring = []

            for _ in range(self.num_offspring // 2):
                parent0, parent1 = self.parent_selection(fitness_scores)
                child0, child1 = self.cross_over(parent0, parent1)
                mutated_child0 = self.mutation(child0)
                mutated_child1 = self.mutation(child1)
                offspring.extend([mutated_child0, mutated_child1])

            self.population = self.select_survivors(offspring, games)

            fitness_scores = list(map(lambda x: self.calculate_fitness(x, games), self.population))
            mean_fitness = stats.mean(fitness_scores)
            best_fitness = min(fitness_scores)
            improve_rate = best_fitness - previous_best_fitness
            is_local_minimum = improve_rate >= self.tolerance_threshold

            if is_local_minimum:
                self.stagnant_generations += 1
            else:
                self.stagnant_generations = 0
                self.mutation_rate = self.initial_mutation_rate
                self.crossover_rate = self.initial_crossover_rate

            if self.stagnant_generations > self.tolerance:
                self.mutation_rate += 0.1 * self.mutation_rate
                self.crossover_rate -= 0.1 * self.crossover_rate

            previous_best_fitness = best_fitness

            mean_fitness_per_generation.append(mean_fitness)
            best_fitness_per_generation.append(best_fitness)

            if verbose:
                print(
                    f"Generation: {generations}, Best Fitness: {best_fitness}, Mean Fitness: {mean_fitness}")
                
            generations += 1
            is_timeout = (generations >= max_generations)
            is_solution = best_fitness <= self.fitness_threshold

        self.solution = min(
            self.population,
            key=lambda x: self.calculate_fitness(x, games))
        best_fitness = min(fitness_scores)

        if verbose:
            print("Optimization finished.")
            print(f"Best Individual: {self.solution}")
            print(f"Best Fitness: {best_fitness}")

        return {
            "Solution": self.solution,
            "Fitness": best_fitness,
            "Mean Fit per Generation": mean_fitness_per_generation,
            "Best Fit per Generation": best_fitness_per_generation,
            "Generations": generations
        }

    def predict(
            self,
            home_id: int,
            away_id: int) -> Tuple[int, float]:
        """Return the game outcome and the home team win proba."""
        team_h = self.extract_team_info(home_id, self.solution)
        team_a = self.extract_team_info(away_id, self.solution)

        proba = self.predict_proba(team_h, team_a)

        return int(proba > self.home_win_rate), proba
