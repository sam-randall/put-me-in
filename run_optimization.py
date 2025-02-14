import cvxpy as cp
import numpy as np
import pandas as pd

def main():
    NUMBER_OF_PERIODS = 6
    NUMBER_OF_CHILDREN = 10
    MAX_CONSECUTIVE_BENCH_PERIODS = 1
    N_ALL_STAR_PERIODS = 1
    SPOTS = 5

    skills = np.array([2, 2, 2, 3, 3, 3, 4, 4, 5, 7])

    best_player_indices = np.argsort(skills)[-SPOTS:]

    # We can specify a matrix here.
    variables = cp.Variable((NUMBER_OF_PERIODS, NUMBER_OF_CHILDREN), boolean = True)

    # TODO: Minimize variance. 
    objective = cp.Minimize(1)

    constraints = []

    # For each period, add a constraint where the 
    # total number of children
    # playing is the number allowed.
    for p in range(NUMBER_OF_PERIODS):
        constraints += [variables[p].sum() == SPOTS]

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

    period_totals = cp.Variable(NUMBER_OF_PERIODS)
    max_skill = skills[best_player_indices].sum()

    indicator = cp.Variable(NUMBER_OF_PERIODS, boolean=True)

    # Need to make sure the count of periods that have total skill 
    # level of max_skill == N_ALL_STAR_PERIODS.

    M = 1000  # Large enough to handle constraints but not too large

    constraints += [period_totals == variables @ skills - max_skill]
    # Constraints
    constraints_all_star = [
        period_totals <= M * (1 - indicator),  # Upper bound when indicator=1
        period_totals >= -M * (1 - indicator),  # Lower bound when indicator=1
        cp.sum(indicator) == N_ALL_STAR_PERIODS  # Ensure exactly N_ALL_STAR_PERIODS are max_skill
    ]
    # TODO: Indicator doesn't have enough forced ones. 

    constraints += constraints_all_star

    problem = cp.Problem(objective, constraints)
    problem.solve()

    print("Assignments.")
    df = pd.DataFrame(variables.value, columns = [f'player_{i}' for i in range(NUMBER_OF_CHILDREN)])
    df.index = [f"Period {p}" for p in range(NUMBER_OF_PERIODS)]
    print(df)
    print()


    print("Derived Variables.")

    print("Period Totals", period_totals.value)
    print("Indicator", indicator.value)
    print()

    print("Constraints.")

    print(period_totals.value, '<=', M * (1 - indicator.value))
    print(period_totals.value, '>=',-M * (1 - indicator.value))


    # Current Problem: 

if __name__ == '__main__':
    main()
