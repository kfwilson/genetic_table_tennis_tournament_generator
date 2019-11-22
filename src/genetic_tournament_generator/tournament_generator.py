from collections import Counter
import random
from statistics import variance
import time
from typing import List, Set, Tuple, Union


class Match:  # Gene
    def __init__(self, team_1: Union[List, Set, Tuple], team_2: Union[List, Set, Tuple]):
        self.team_1 = set(team_1)
        self.team_2 = set(team_2)
        assert not self.team_1.intersection(self.team_2), "No overlapping players between teams"

    def __eq__(self, other):
        return ((self.team_1 == other.team_1) and (self.team_2 == other.team_2)
                or (self.team_1 == other.team_2) and (self.team_2 == other.team_1))

    def __hash__(self):
        return hash(f"{sorted(self.team_1)}.{sorted(self.team_2)}")

    def __str__(self):
        return f'{self.team_1} - {self.team_2}'

    def to_list(self):
        return list(self.team_1) + list(self.team_2)


# For simplicity, let's start with the case of a single Gene (match) played
# simultaneously.

class Tournament:  # Genome
    def __init__(self, num_players, num_matches, matches: List['Match'] = None):
        self.num_players = num_players
        self.num_matches = num_matches
        if not matches:
            self.matches = [self._make_random_match() for _ in range(self.num_matches)]
        else:
            self.matches = matches

    @property
    def fitness(self):
        # number of unique matches
        unique_matches = set()
        players = []
        for g in self.matches:
            unique_matches.add(g)
            players.extend(g.to_list())
        return len(unique_matches) * (1 - variance(Counter(players).values()))

    def _make_random_match(self):
        num_players_for_match = 4
        players = random.sample(range(self.num_players), num_players_for_match)
        return Match(players[:num_players_for_match // 2], players[num_players_for_match // 2:])

    def mutate(self):
        for i in range(len(self.matches)):
            if random.random() < 0.9:
                # TODO: do I want mutated gene to be based on original?
                self.matches[i] = self._make_random_match()

    def __str__(self):
        return '\n'.join([str(g) for g in self.matches])


def crossover(a: 'Tournament', b: 'Tournament'):
    offspring_matches = []
    for left, right in zip(a.matches, b.matches):
        if random.random() < 0.5:
            offspring_matches.append(left)
        else:
            offspring_matches.append(right)
    return Tournament(a.num_players, a.num_matches, offspring_matches)


def initial_population(size, num_players, num_matches):
    population = []
    for _ in range(size):
        population.append(Tournament(num_players, num_matches))
    return population


def next_generation(current_generation):
    tournaments_by_fitness = sorted(current_generation, key=lambda g: g.fitness)
    # fittest from current generation will survive unmutated into future generation
    elite = tournaments_by_fitness[-1]
    # all genomes from current generation that are in the top 40% based on fitness will reproduce
    breeding_population = set(tournaments_by_fitness[int(len(current_generation) * 0.6):])

    children = [elite]
    while len(children) < len(current_generation):
        a = random.choice(list(breeding_population))
        b = random.choice(list(breeding_population - {a}))
        child = crossover(a, b)
        child.mutate()
        children.append(child)

    return children


def evolve_tournament(initial_population_size, generations_to_evolve, players_per_tournament, matches_per_tournament):
    start = time.time()
    population = initial_population(initial_population_size, players_per_tournament, matches_per_tournament)
    elite = sorted(population, key=lambda g: g.fitness)[-1]
    generation_elite_found = 0

    for i in range(generations_to_evolve):  # num generations
        population = next_generation(population)
        elite_in_generation = sorted(population, key=lambda g: g.fitness)[-1]
        if elite_in_generation.fitness > elite.fitness:
            print(f'Fitter tournament found in generation {i}.')
            elite = elite_in_generation
            generation_elite_found = i
    end = time.time()
    players = []
    for g in elite.matches:
        players.extend(g.to_list())

    print(f'Evolving for {generations_to_evolve} generations took {round(end-start)} seconds. The fittest tournament '
          f'was found in generation {generation_elite_found}. That tournament is: \n{elite}.')
    print(f'Match counts by player in that tournament are as follows: {Counter(players)}')


if __name__ == "__main__":
    evolve_tournament(100, 500, 13, 20)