import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional, Union
import time
from enum import Enum

class Game:
    def __init__(self, number_of_periods: int, players_per_period: int):
        self.number_of_periods = number_of_periods
        self.players_per_period = players_per_period

    def to_dict(self):
        return self.__dict__
    
def mean(x):
    return cp.sum(x) / x.size

def variance(X: cp.Variable):
    return cp.sum_squares(X - mean(X)) # / scale

class Constraint(Enum):
    MAX_CONSECUTIVE_BENCH_TIME = 'max_consecutive_bench_time'
    MAX_CONSECUTIVE_PLAY_TIME = 'max_consecutive_play_time'
    MIN_PLAY_TIME = 'min_play_time'
    MAX_PLAY_TIME = 'max_play_time'



def define_constraints(variables: cp.Variable, parameters: Dict[str, int], enabled_constraints: Dict[str, bool]) -> List[cp.Expression]:
    '''Given parameters, for basketball line-up problem, return list of relevant constraints.'''

    NUMBER_OF_PERIODS = parameters['number_of_periods']
    SPOTS = parameters['spots']
    NUMBER_OF_CHILDREN = parameters['number_of_children']

    MAX_CONSECUTIVE_BENCH_PERIODS = parameters['max_consecutive_bench_periods']
    MAX_CONSECUTIVE_PLAYING_PERIODS = parameters['max_consecutive_playing_periods']

    MIN_PERIODS_A_CHILD_MUST_PLAY = parameters['min_periods_a_child_must_play']
    MAX_PERIODS_A_CHILD_MUST_PLAY = parameters['max_periods_a_child_must_play']

    constraints = []

    # Make sure there are SPOTS (= 5) players playing per period.
    for p in range(NUMBER_OF_PERIODS):
        constraints += [variables[p].sum() == SPOTS]


    if enabled_constraints.get(Constraint.MAX_CONSECUTIVE_BENCH_TIME, False):

        # For each child, make sure it never has more than 
        # MAX_CONSECUTIVE_BENCH_PERIODS in a row.
        for c in range(NUMBER_OF_CHILDREN):
            for i in range(NUMBER_OF_PERIODS - MAX_CONSECUTIVE_BENCH_PERIODS):
                # Make sure for every start 
                # every sequence of length 
                # MAX_CONSECUTIVE_BENCH_PERIOD + 1
                # has more than 1 in it.
                total = 0
                for j in range(i, i + MAX_CONSECUTIVE_BENCH_PERIODS + 1):
                    total += variables[j][c]
                constraints += [total >= 1]

    # Make sure each child plays a minimum number of periods.
    # Make sure each child does not play more than a maximum number of periods.
    for c in range(NUMBER_OF_CHILDREN):

        if enabled_constraints.get(Constraint.MIN_PLAY_TIME, False):
            constraints += [variables[:, c].sum() >= MIN_PERIODS_A_CHILD_MUST_PLAY]

        if enabled_constraints.get(Constraint.MAX_PLAY_TIME, False):
            constraints += [variables[:, c].sum() <= MAX_PERIODS_A_CHILD_MUST_PLAY]

    # Make sure no kids plays for more than MAX_CONSECUTIVE_PLAYING_PERIODS
    # otherwise they get tired :(
    if enabled_constraints.get(Constraint.MAX_CONSECUTIVE_PLAY_TIME, False):
        for c in range(NUMBER_OF_CHILDREN):
            for i in range(0, NUMBER_OF_PERIODS - (MAX_CONSECUTIVE_PLAYING_PERIODS + 1)):
                constraints += [variables[i:i + MAX_CONSECUTIVE_PLAYING_PERIODS + 1, c].sum() \
                                <= MAX_CONSECUTIVE_PLAYING_PERIODS]
    
    return constraints

def frame_problem(number_of_periods: int, skill_levels: np.ndarray):
    '''Frames the problem as period x child graph problem.'''
    number_of_children = len(skill_levels)
    independent_variables = cp.Variable((number_of_periods, number_of_children), boolean = True)
    objective = cp.Minimize(variance(independent_variables @ skill_levels))
    return independent_variables, objective


def generate_assignment(names, skills, game_config: Game, initial_line_up: Optional[Union[List[str], Literal['auto']]] = None, enabled_constraints = None):
    NUMBER_OF_PERIODS = game_config.number_of_periods
    
    MAX_CONSECUTIVE_BENCH_PERIODS = 1
    SPOTS = game_config.players_per_period

    NUMBER_OF_CHILDREN = len(skills)

    total_spots = SPOTS * NUMBER_OF_PERIODS

    # To ensure fairness, max periods_a_child_must_play is greater than min_periods_a_child_must_play
    # but not by much!

    MIN_PERIODS_A_CHILD_MUST_PLAY = total_spots // NUMBER_OF_CHILDREN
    MAX_PERIODS_A_CHILD_MUST_PLAY = MIN_PERIODS_A_CHILD_MUST_PLAY + 1

    variables, objective = frame_problem(NUMBER_OF_PERIODS, skills)

    parameters = {
        'number_of_periods' : NUMBER_OF_PERIODS,
        'spots': SPOTS,
        'number_of_children': NUMBER_OF_CHILDREN,
        'max_consecutive_bench_periods': MAX_CONSECUTIVE_BENCH_PERIODS,
        'max_consecutive_playing_periods': 3,
        'min_periods_a_child_must_play': MIN_PERIODS_A_CHILD_MUST_PLAY,
        'max_periods_a_child_must_play': MAX_PERIODS_A_CHILD_MUST_PLAY
    }

    constraints = define_constraints(variables, parameters, enabled_constraints)

    print(enabled_constraints)

    if initial_line_up is not None:

        if initial_line_up == 'auto':
            initial_line_up = np.random.choice(names, SPOTS, replace = False)
            print(f"Initial Lineup set to {initial_line_up}")

        name_to_index = {name: i for i, name in enumerate(names)}

        first_period_lineup = np.zeros(NUMBER_OF_CHILDREN)
        for child in initial_line_up:
            idx = name_to_index[child]
            first_period_lineup[idx] = 1
        
        constraints += [variables[0] == first_period_lineup]

    problem = cp.Problem(objective, constraints)
    out = problem.solve(solver = cp.SCIP)

    active_constraints = [c.value for c, isActive in enabled_constraints.items() if isActive]
    if problem.status == 'infeasible':
        return pd.DataFrame(), problem.status, active_constraints

    # Post Processing.
    df = pd.DataFrame(variables.value, columns = names)

    cleaned_up = []
    for _, row in df.iterrows():
        playing = [name for item, name in zip(row, names) if item > 0.99]
        cleaned_up.append(playing)

    df = pd.DataFrame(cleaned_up)
    df.index = [f"Period {p}" for p in range(1, NUMBER_OF_PERIODS + 1)]
    df.columns = [f"Player {i}" for i in range(1, SPOTS + 1)]
    # END Post Processing.
    return df, problem.status, active_constraints

def main():

    game = Game(6, 5)

    first_period_lineup = ["Jameson", "Beckham", "Caleb", "Dominic", "Leo"]
    names = ["Jack", "Jameson", "Caleb", "Beckham", "Dillon", "Brady", "Dominic", "Leo"]
    skills = np.array([3, 3, 3, 2, 1, 1, 1, 1])
    start = time.time()
    df, _ = generate_assignment(names, skills, game, first_period_lineup)
    df = df.T
    end = time.time()
    print(end - start)
    df.to_csv('assignments.csv')

    print(df)

if __name__ == '__main__':
    main()
