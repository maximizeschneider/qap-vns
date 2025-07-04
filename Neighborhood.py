from Solution import *
from helper import *
import numpy as np
import itertools
from timeit import default_timer
from InputData import InputData
from Timer import Timer
from Logger import Logger

class Neighborhood:
    def __init__(self, input_data: InputData, rng: np.random.Generator, logger: Logger, timer: Timer, type: str) -> None:
        self.input_data = input_data
        self.rng = rng
        self.logger = logger
        self.timer = timer
        self.type = type
        self.k = int(type.split("-")[0])

        start = default_timer()
        self.combinations = self._get_all_combinations(self.input_data.n, self.k)
        self.time_taken = default_timer() - start

    def _get_all_combinations(self, n: int, k: int) -> list[tuple]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def _make_move(self, initial_permutation: np.ndarray, combination: tuple) -> Solution:
        raise NotImplementedError("Subclasses must implement this method")


    def discover_and_evaluate_moves(self, evaluation_strategy: str, solution: Solution) -> Solution:
        """
        Looks for moves and evaluates them based on the evaluation strategy.
        Returns the best solution or the starting solution if no improvement is found.
        2 different evaluation strategies:
        - FirstImprovement: Stop after finding the first improvement but shuffle the combinations randomly to avoid focus on the first combinations
        - BestImprovement: Continue until all moves are evaluated and the best is returned as solution. Implementation is without an array to store the moves because you would need to sort which adds time complexity
        """

        best_solution = solution

        # shuffle combinations
        if evaluation_strategy == "FirstImprovement":
            self.rng.shuffle(self.combinations)

        for combination in self.combinations:
            if self.timer.is_time_up():
                return best_solution
            new_solution = self._make_move(solution.permutation, combination)
            if evaluation_strategy == "BestImprovement":
                if new_solution.is_better_than(best_solution):   
                    best_solution = new_solution
            elif evaluation_strategy == "FirstImprovement":
                if new_solution.is_better_than(best_solution):
                    best_solution = new_solution
                    break
            else:
                error_message = f"Invalid evaluation strategy: {evaluation_strategy}"
                self.logger.log(error_message, self.timer.get_time(), "error")
                raise ValueError(error_message)
        return best_solution
    

    def shake(self, solution: Solution) -> Solution:
        """
        Shakes the solution by applying a random move from the neighborhood to escape local optima. 
        This is done by selecting a random combination from the neighborhood and applying the move.
        Returns the new solution.
        """

        random_idx = self.rng.integers(len(self.combinations))
        combination = self.combinations[random_idx]
        new_solution = self._make_move(solution.permutation, combination)
        self.logger.log(f"Shaked solution with {self.type} neighborhood and combination {combination} at index {random_idx}.", self.timer.get_time(), "debug")
        return new_solution

       
    def local_search(self, neighborhood_evaluation_strategy: str, solution: Solution) -> Solution:
        while True:
            new_solution = self.discover_and_evaluate_moves(neighborhood_evaluation_strategy, solution)
            
            if new_solution.is_better_than(solution):
                solution = new_solution
            else:
                self.logger.log(f"Reached local optimum of {self.type} neighborhood. Stop local search.", self.timer.get_time(), "debug")
                break

            if self.timer.is_time_up():
                self.logger.log(f"Time is up.", self.timer.get_time())
                self.logger.log(f"Local search in {self.type} neighborhood stopped.", self.timer.get_time(), "debug")
                break

        return solution
        

class K_SwapNeighborhood(Neighborhood):
    """
    Contains all k-swap moves for a given permutation.
    A swap exchanges the elements at two distinct indices.
    A combination is a tupel of tupels. The k inner tupels represent the 1-Swap moves. 
    For k-swap k swaps are applied after each other. The indices of the swaps should be disjoint.
    """

    def __init__(self, input_data: InputData, rng: np.random.Generator, logger: Logger, timer: Timer, type: str) -> None:
        super().__init__(input_data, rng, logger, timer, type)

    @staticmethod
    def _is_disjoint(combination: tuple[tuple[int, int], ...]) -> bool:
        """Return True if index was not used before in the combination."""

        used: set[int] = set()
        for i, j in combination:
            if i in used or j in used:
                return False
            used.add(i)
            used.add(j)
        return True

    @staticmethod
    def _get_all_combinations(n: int, k: int) -> list[tuple[tuple[int, int], ...]]:
        """Return list of all (possibly disjoint) k-swap combinations. Could be done faster for disjoint swaps but itertools is in C (hard to beat)."""

        # get all index pairs (i, j) with i < j
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] # pairs = list(itertools.permutations(range(n), 2))
        # generate the k-length combinations of those pairs and filter out non disjoint combinations
        combinations = [x for x in itertools.combinations(pairs, k) if K_SwapNeighborhood._is_disjoint(x)]
        return combinations


    def _make_move(self, initial_permutation: np.ndarray, combination: tuple[tuple[int, int], ...]) -> Solution:
        """Apply the sequence of swaps described by combination."""

        permutation = initial_permutation.copy()
        for i, j in combination:
            permutation[i], permutation[j] = permutation[j], permutation[i]

        return Solution(permutation, self.input_data, self.type)


class K_InsertionNeighborhood(Neighborhood):
    """
    Contains all k-insertion moves for a given permutation.
    A single insertion removes the element at index i and reinserts it at index j. I and j cannot be the same.
    For k-insertion k such moves are applied after each other. 
    A combination is a tupel of tupels. The k inner tupels represent the 1-Insertion moves.
    Duplicates are allowed, therefore combinations are generated with replacement. 
    ((1,4), (1,4)) are fine beacause they dont lead to the same solution.
    ((1,3),(2,3)) is not the same as ((2,3),(1,3))
    Could lead to multiple insertions leading to the same solution. Filter out adjacent or same insertions. ((1,8),(8,1)) are the same and ((1,2),(1,2)) would lead to the same solution.
    For k>2 there could be combinations that lead to the same solution cause they cancel each other out. ((1,2),(2,4),(4,1)) would lead to the same starting solution -> ignored cause k = 3 is not sufficient anyway (too large)
    Ideally you should filter out one non-interfering combinations from ((1,2),(3,5)) and ((3,5),(1,2)) cause they lead to the same solution. (not done here) 
    """

    def __init__(self, input_data: InputData, rng: np.random.Generator, logger: Logger, timer: Timer, type: str) -> None:
        super().__init__(input_data, rng, logger, timer, type)

    @staticmethod   
    def _is_valid_combination(combination: tuple[tuple[int, int], ...]) -> bool:
        """Invalid if adjacent or same insertions."""
        seen_pairs = set()
        for i, j in combination:
            # check for reverse insertions (i,j) and (j,i)
            if (j, i) in seen_pairs:
                return False
            # check for adjacent insertions (i,j) and (i,j+1) or (i,j-1) that already in seen_pairs
            if (i, j) in seen_pairs:
                if i == j - 1 or i == j + 1:
                    return False
            seen_pairs.add((i, j))
        
        return True

    @staticmethod
    def _get_all_combinations(n: int, k: int) -> list[tuple[tuple[int, int], ...]]:
        # get all ordered pairs (i, j) with i != j
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j] # pairs = list(itertools.permutations(range(n), 2))
        # allow repetitions and different order of pairs and filter out adjacent or same insertions, cause order is important
        combinations = [x for x in itertools.product(pairs, repeat=k) if K_InsertionNeighborhood._is_valid_combination(x)]
        return combinations
    
    def _make_move(self, initial_permutation: np.ndarray, combination: tuple[tuple[int, int], ...]) -> Solution:
        permutation = initial_permutation.copy()
        
        for i, j in combination:
            value_to_insert = permutation[i]
            permutation = np.delete(permutation, i)
            permutation = np.insert(permutation, j, value_to_insert)

        return Solution(permutation, self.input_data, self.type)


class K_CutNeighborhood(Neighborhood):
    """
    Return all ways to choose k distinct cut‐points (1 through n-1).
    Reorders splits and concatenates them again.
    A cut combination is a tupel of two tupels. The first tupel represents the indices of the cuts. The second tupel represents the reordering of the resultingsplits.
    The cut‐points must be strictly increasing and they must not have indices that are either 0 or n to avoid empty splits.
    Filters out reordering that would lead to the same solution.
    """

    def __init__(self, input_data: InputData, rng: np.random.Generator, logger: Logger, timer: Timer, type: str) -> None:
        super().__init__(input_data, rng, logger, timer, type)  

    @staticmethod
    def _get_all_combinations(n: int, k: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        # neither 0 nor n are valid cut points
        combo_indices_cut_points = itertools.combinations(range(1, n), k)
        # reordering of the splits, k plus 1 cause k cuts make k+1 splits
        reorderings = [item for item in itertools.permutations(range(k+1), k+1) if list(item) != list(range(k+1))]
        
        # create all combinations of index selections with reorderings
        combinations = [(indices, reorder) for indices in combo_indices_cut_points for reorder in reorderings]
        return combinations

    def _make_move(self, initial_permutation: np.ndarray, combination: tuple[tuple[int, ...], tuple[int, ...]]) -> Solution:
        permutation = initial_permutation.copy()
        indices, reorder = combination
        
        # split the permutation at the indices (should return indices + 1 arrays)
        splits = np.split(permutation, indices)
        reordered_splits = [splits[i] for i in reorder]
        permutation = np.concatenate(reordered_splits)

        return Solution(permutation, self.input_data, self.type)


class K_PermuteNeighborhood(Neighborhood):
    """
    A k-permute selects k distinct indices and reorders the corresponding values using any permutation different from the identity.
    A permute combination is a tupel of two tupels. The first tupel represents the indices of the values to permute. The second tupel represents the reordering of the values.
    """

    def __init__(self, input_data: InputData, rng: np.random.Generator, logger: Logger, timer: Timer, type: str) -> None:
        super().__init__(input_data, rng, logger, timer, type)

    @staticmethod
    def _get_all_combinations(n: int, k: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        # Get all possible k-length combinations of indices 
        combo_indices_permute = list(itertools.combinations(range(n), k))
        # Get all permutations except the identity permutation
        reorderings = [item for item in itertools.permutations(range(k), k) if list(item) != list(range(k))]

        # Create all combinations of index selections with reorderings
        combinations = [(indices, reorder) for indices in combo_indices_permute for reorder in reorderings] # combinations = list(itertools.product(combo_indices_permute, reorderings))

        return combinations


    def _make_move(self, initial_permutation: np.ndarray, combination: tuple[tuple[int, ...], tuple[int, ...]]) -> Solution:
        permutation = initial_permutation.copy()
        
        indices, reorder = combination
        indices_arr = np.array(indices)
        reorder_arr = np.array(reorder)
        # reorder_arr of (2,0,1) means value at index 2 moves to index 0 etc 
        permutation[indices_arr] = permutation[indices_arr][reorder_arr]

        return Solution(permutation, self.input_data, self.type)   


class K_BlockNeighborhood(Neighborhood):
    """
    A block move extracts the contiguous subsequence starting at index_a with length k and reinserts it starting at index_b.
    Index_b is not the index where the block should be inserted from the permutation thats left after the block is extracted.
    K in this case is the length of the block.
    """

    def __init__(self, input_data: InputData, rng: np.random.Generator, logger: Logger, timer: Timer, type: str) -> None:
        super().__init__(input_data, rng, logger, timer, type)

    @staticmethod
    def _get_all_combinations(n: int, k: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(n-k+1) for j in range(n-k+1) if j != i and j != i - k] #(k == 1 or j != i - k)] # last condition is only if k is not equal to 1 (code from lecture is missing this to be correct for block length of 1)
        # edit: the k==1 is not correct. the condition should apply to all k. (1,5) and (1,6) would be the same move, but result in less then n*(n-1) combinations (same is true for insertion)
    

    def _make_move(self, initial_permutation: np.ndarray, combination: tuple[int, int]) -> Solution:
        index_a, index_b = combination

        permutation = initial_permutation.copy()
        start, block, end = np.split(permutation, (index_a, index_a + self.k))

        permutation = np.concatenate([start, end])
        permutation = np.insert(permutation, index_b, block)

        return Solution(permutation, self.input_data, self.type)
