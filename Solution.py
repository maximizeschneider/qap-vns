import numpy as np
import json
from InputData import InputData
from typing import Literal
from helper import *

class Solution:
    """
    A solution for the Quadratic Assignment Problem.
    It contains the permutation, the move that was made to get to this solution and the cost of the solution.
    """
    def __init__(self, permutation: np.ndarray, input_data: InputData, search_neighborhood: str = "None", shaking_neighborhood: str = "None") -> None:
        self.permutation = permutation
        self.cost = self.calculate_cost(permutation, input_data)
        # for analysis
        # set in Neighborhood.local_search
        self.search_neighborhood = search_neighborhood
        # set in VariableNeighborhoodSearch.local_search_procedure
        self.shaking_neighborhood = shaking_neighborhood
        
        
        
    def __str__(self) -> str:
        return "The permutation " + str([int(i) for i in self.permutation]) + " results in a cost of " + str(self.cost)

    @staticmethod
    def calculate_cost(permutation: np.ndarray, input_data: InputData, method: Literal["numpy", "python-loop"] = "numpy", penalty: bool = False) -> float:
        """
        Calculate the cost of a given permutation for the Quadratic Assignment Problem.
        If the minimum distance is considered, the cost is set to infinity if the minimum distance is not met or if penalty is True, the cost is increased by the penalty factor and the values of the distance matrix that are below the minimum distance.
        The method can be either "numpy" or "python-loop".
        The numpy method is faster than the python-loop method because it is vectorized.
        If the permutation has unassigned facilities the cost is calculated for assigned facilities only (used in constructive heuristic: increasing_dof_greedy, python-loop is missing this partial calculation)
        """
        F = input_data.F
        D = input_data.D
        M = input_data.M

        penalty_factor = 1_000_000
        
        if method == "numpy":
            placed = np.where(permutation >= 0)[0]
            if placed.size == 0:
                return 0
            cost = 0
            F_subset= F[np.ix_(placed, placed)]
            D_subset = D[np.ix_(permutation[placed], permutation[placed])] # distance per facility
            if input_data.consider_minimum_distance:
                M_subset = M[np.ix_(placed, placed)]
                violation = D_subset < M_subset
                if np.any(violation):
                    if penalty:
                        cost += np.sum((M_subset[violation] - D_subset[violation]))*penalty_factor
                    else:
                        return np.inf
            cost += np.sum(F_subset * D_subset)
            return cost
        
        elif method == "python-loop":
            cost = 0
            for facility_i, location_i in enumerate(permutation):
                for facility_j, location_j in enumerate(permutation):
                    distance = D[location_i, location_j]
                    if input_data.consider_minimum_distance:
                        if distance < M[facility_i, facility_j]:
                            return np.inf
                    cost += F[facility_i, facility_j] * distance

            return cost
        
    def is_better_than(self, previous_solution: "Solution", alpha: float = 0) -> bool:
        """Checks if the solution is better than the previous solution. If the previous solution is None, then the new solution is better.
        The alpha parameter is used to control the influence of the hamming distance on the decision. Only used for skewed objectives and often set to 0, which implies new_distance < previous_distance.
        """
        if previous_solution is None:
            return True
        return self.cost - previous_solution.cost < alpha * hamming_distance(self.permutation, previous_solution.permutation)


    def check_optimal_solution(self, input_data: InputData, print_diff: bool = True) -> float:
        """Checks if the solution is optimal. If the solution is optimal, the cost is 0. returns the relative difference in %
        """
        filename = f"./BKS/{input_data.name}_Lsg_{"m" if input_data.consider_minimum_distance else "o"}MA.json"
        with open(filename, "r") as f:
            optimal_solution = json.load(f)
        diff = (self.cost - optimal_solution['ObjVal']) / optimal_solution['ObjVal'] * 100
        if print_diff:
            print(f"Optimal solution: {optimal_solution['ObjVal']}, Solution: {self.cost}")
            print(f"Relative difference in %: {diff:.2f}%")
        return float(diff)