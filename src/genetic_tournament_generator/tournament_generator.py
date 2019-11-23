from collections import Counter, namedtuple
from pathlib import Path
import random
from statistics import variance
import time
from typing import List, Set, Tuple, Union
import warnings

import pandas as pd


class Match:
    players_per_match = 4

    def __init__(self, team_1: Union[List, Set, Tuple], team_2: Union[List, Set, Tuple]):
        assert not set(team_1).intersection(team_2), "No overlapping players between teams"
        self.team_1 = list(team_1)
        self.team_2 = list(team_2)

    @property
    def players(self):
        return self.team_1 + self.team_2

    def __eq__(self, other):
        return ((self.team_1 == other.team_1) and (self.team_2 == other.team_2)
                or (self.team_1 == other.team_2) and (self.team_2 == other.team_1))

    def __hash__(self):
        return hash(f"{sorted(self.team_1)}.{sorted(self.team_2)}")

    def __str__(self):
        return f'{self.team_1} - {self.team_2}'

    @classmethod
    def make_random_match_from_players(cls, players: Union[List, Set]) -> 'Match':
        """Generates a new random match by choosing ``players_per_match``
        players at random from players and splitting them into 2 teams.
        """
        players = random.sample(players, cls.players_per_match)
        return Match(players[:cls.players_per_match // 2], players[cls.players_per_match // 2:])


class Round:  # Gene
    def __init__(self, matches: List[Match]):
        # TODO: validate no repeat players across matches in a single round?
        self.matches = matches

    @property
    def players(self) -> List:
        players = []
        for m in self.matches:
            players.extend(m.players)
        return players

    def __str__(self):
        matches = [f'{i+1}: {str(m)}' for i, m in enumerate(self.matches)]
        return '\n'.join(matches)

    @classmethod
    def make_random_round_from_players(cls, matches_per_round: int, players: List) -> 'Round':
        """Generates a new round (set of ``matches_per_round`` matches) from the
        given set of players, ensuring that no single player plays twice in the
        round."""
        not_currently_playing = set(players)
        matches = []
        for _ in range(matches_per_round):
            match = Match.make_random_match_from_players(not_currently_playing)
            not_currently_playing -= set(match.players)
            matches.append(match)
        return Round(matches)


class Tournament:  # Genome
    def __init__(self, num_players: int, num_rounds: int, matches_per_round: int, rounds: List['Round'] = None):
        self.num_players = num_players
        self.num_rounds = num_rounds
        # ASSUMING validate_simultaneous_matches has already been called to update this if needed
        self.matches_per_round = matches_per_round
        if not rounds:
            self.rounds = [Round.make_random_round_from_players(self.matches_per_round, list(range(self.num_players)))
                           for _ in range(self.num_rounds)]
        else:
            self.rounds = rounds

    @property
    def fitness(self) -> float:
        """Calculates the current fitness of the tournament, defined as the
        number of unique matches in the tournament times 1 minus the variance
        of the number of games played by each player.
        """
        unique_matches = set()
        players = []
        for g in self.rounds:
            unique_matches.update(g.matches)
            players.extend(g.players)
        # ensure if a player plays 0 matches in the current tournament, we still include them for variance calc
        play_counts = Counter({p: 0 for p in range(self.num_players)}) + Counter(players)
        return len(unique_matches) * (1 - variance(play_counts.values()))

    def mutate(self):
        """Mutates the current genome (list of matches) by replacing each match
        with a new random match with a probability of 0.9.
        """
        probability_of_mutation = 0.9
        for i in range(len(self.rounds)):
            if random.random() < probability_of_mutation:
                # TODO: do I want mutated gene to be based on original?
                self.rounds[i] = Round.make_random_round_from_players(self.matches_per_round,
                                                                      list(range(self.num_players)))

    def __str__(self):
        return '\n'.join([f'ROUND {i}:\n' + str(g) for i, g in enumerate(self.rounds)])

    def record_metrics(self, out_file: Path):
        games_played = Counter({p: 0 for p in range(self.num_players)})
        unique_partners = {p: set() for p in range(self.num_players)}
        unique_opponents = {p: set() for p in range(self.num_players)}
        for r in self.rounds:
            for m in r.matches:
                games_played += Counter(m.players)
                for i in range(2):
                    team = m.team_1 if i == 0 else m.team_2
                    other = m.team_2 if i == 0 else m.team_1
                    unique_partners[team[0]].add(team[1])
                    unique_partners[team[1]].add(team[0])
                    unique_opponents[team[0]].update(other)
                    unique_opponents[team[1]].update(other)

        # format to work w/ existing analysis code
        games_played = pd.Series(games_played, name='gamesPlayed')
        unique_partners = pd.Series({k: len(v) for k, v in unique_partners.items()}, name='uniquePartners')
        unique_opponents = pd.Series({k: len(v) for k, v in unique_opponents.items()}, name='uniqueOpponents')
        (pd.concat([games_played, unique_partners, unique_opponents], axis=1).reset_index()
         .rename(columns={'index': 'player'}).to_csv(out_file, index=False))


def crossover(a: Tournament, b: Tournament) -> Tournament:
    """Combines two 'parent' tournament (genomes) into an 'offspring' tournament
    by mixing the component rounds (genes) of each parent. For each gene in
    the genome, each parent's gene has equal probability of being chosen to be
    inherited by the offspring.
    """
    offspring_rounds = []
    inheritance_from_a_probability = 0.5
    for left, right in zip(a.rounds, b.rounds):
        if random.random() < inheritance_from_a_probability:
            offspring_rounds.append(left)
        else:
            offspring_rounds.append(right)
    return Tournament(a.num_players, a.num_rounds, a.matches_per_round, offspring_rounds)


def initial_population(size: int, num_players: int, num_rounds: int, matches_per_round: int) -> List[Tournament]:
    """Generate a new population of ``size`` potential tournaments, each with
    ``num_players`` players and ``num_rounds`` rounds of ``matches_per_round``
    matches. """
    population = []
    for _ in range(size):
        population.append(Tournament(num_players, num_rounds, matches_per_round))
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


def validate_simultaneous_matches(matches_per_round: int, num_players: int) -> int:
    """Verify that the given number of matches can be played simultaneously with
    the given number of players (e.g., 2 matches can't be played with 6 players
    since each match requires 4 players). Asserts that at least a single match
    can be played with the given number of players and returns the given matches
    per round if it's possible, otherwise the maximum number of rounds that can
    be played with the given players.
    """
    actual_matches_to_play = min(num_players // Match.players_per_match, matches_per_round)
    assert actual_matches_to_play > 0, "Too few players to play even a single match."
    if actual_matches_to_play != matches_per_round:
        warnings.warn(f'You specified {matches_per_round} matches to be played simultaneously, but with {num_players} '
                      f'players, only {actual_matches_to_play} can be played. Finding tournament with '
                      f'{actual_matches_to_play} per round.')
    return actual_matches_to_play


def optimize_tournament(initial_population_size: int, generations_to_evolve: int, players_per_tournament: int,
                        rounds_per_tournament: int, matches_per_round: int) -> Tournament:
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
    rounds_per_tournament
        The number of rounds each tournament should consist of.
    matches_per_round
        The number of matches played simultaneously per tournament.
    """
    start = time.time()

    # let's up front fix the number of matches/round so we don't have to do it for each tournament
    matches_per_round = validate_simultaneous_matches(matches_per_round, players_per_tournament)

    population = initial_population(initial_population_size, players_per_tournament,
                                    rounds_per_tournament, matches_per_round)
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
    for g in elite.rounds:
        players.extend(g.players)

    print(f'Evolving for {generations_to_evolve} generations took {round(end-start)} seconds. The fittest tournament '
          f'was found in generation {generation_elite_found}. That tournament is: \n{elite}.')
    print(f'Match counts by player in that tournament are as follows: {Counter(players)}')

    return elite


if __name__ == "__main__":
    pop_size = 5
    generations = 10
    players = 13
    rounds = 20
    sim_matches = 2
    elite = optimize_tournament(pop_size, generations, players, rounds, sim_matches)
    elite.record_metrics(Path(f"C:\\output\\tournament\\genetic_algorithm\\{time.time()}.csv"))
