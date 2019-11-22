from collections import Counter
import random
from statistics import variance
import time
from typing import List, Set, Tuple, Union


class Match:  # Gene
    players_per_match = 4

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

    @classmethod
    def make_random_match_from_n_players(cls, num_players: int) -> 'Match':
        """Generates a new random match by choosing ``players_per_match``
        players at random from players 0, 1, ... , num_players and splitting
        them into 2 teams.
        """
        players = random.sample(range(num_players), cls.players_per_match)
        return Match(players[:cls.players_per_match // 2], players[cls.players_per_match // 2:])


# For simplicity, let's start with the case of a single Gene (match) played
# simultaneously.

class Tournament:  # Genome
    def __init__(self, num_players: int, num_matches: int, matches: List['Match'] = None):
        self.num_players = num_players
        self.num_matches = num_matches
        if not matches:
            self.matches = [Match.make_random_match_from_n_players(self.num_players) for _ in range(self.num_matches)]
        else:
            self.matches = matches

    @property
    def fitness(self) -> float:
        """Calculates the current fitness of the tournament, defined as the
        number of unique matches in the tournament times 1 minus the variance
        of the number of games played by each player.
        """
        unique_matches = set()
        players = []
        for g in self.matches:
            unique_matches.add(g)
            players.extend(g.to_list())
        # ensure if a player plays 0 matches in the current tournament, we still include them for variance calc
        play_counts = Counter({p: 0 for p in range(self.num_players)}) + Counter(players)
        return len(unique_matches) * (1 - variance(play_counts.values()))

    def mutate(self):
        """Mutates the current genome (list of matches) by replacing each match
        with a new random match with a probability of 0.9.
        """
        probability_of_mutation = 0.9
        for i in range(len(self.matches)):
            if random.random() < probability_of_mutation:
                # TODO: do I want mutated gene to be based on original?
                self.matches[i] = Match.make_random_match_from_n_players(self.num_players)

    def __str__(self):
        return '\n'.join([str(g) for g in self.matches])


def crossover(a: Tournament, b: Tournament) -> Tournament:
    """Combines two 'parent' tournament (genomes) into an 'o'ffspring' tournament
    by mixing the component matches (genes) of each parent. For each gene in
    the genome, each parent's gene has equal probability of being chosen to be
    inherited by the offspring.
    """
    offspring_matches = []
    inheritance_from_a_probability = 0.5
    for left, right in zip(a.matches, b.matches):
        if random.random() < inheritance_from_a_probability:
            offspring_matches.append(left)
        else:
            offspring_matches.append(right)
    return Tournament(a.num_players, a.num_matches, offspring_matches)


def initial_population(size: int, num_players: int, num_matches: int) -> List[Tournament]:
    """Generate a new population of ``size`` potential tournaments, each with
    ``num_players`` players and ``num_matches`` matches. """
    population = []
    for _ in range(size):
        population.append(Tournament(num_players, num_matches))
    return population


def next_generation(current_generation: List['Tournament']) -> List[Tournament]:
    """Produce a new generation of tournament individuals based on the current
    generation's population. The fittest individual survives unchanged. The top
    40% of individuals by fitness are crossbred and the offspring mutated to
    produce the remaining individuals in the new generation. Preserves
    generation size (i.e., the new generation is guaranteed to be the same size
    as the current).
    """
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


def optimize_tournament(initial_population_size: int, generations_to_evolve: int, players_per_tournament: int,
                        matches_per_tournament: int):
    """Run the genetic algorithm for the given number of generations, with
    a starting population of the given size, tracking the fittest tournament
    found and the generation in which it was found to print at the end of the
    given number of generations.

    Parameters
    ----------
    initial_population_size
        The number of tournaments to generate for the starting population.
    generations_to_evolve
        The number of generations to run the algorithm for.
    players_per_tournament
        The number of players in each tournament.
    matches_per_tournament
        The number of matches each tournament should consist of.
    """
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
    optimize_tournament(100, 500, 13, 20)
