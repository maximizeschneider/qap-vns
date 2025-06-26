from Neighborhood import *
from helper import *
from InputData import InputData
from Timer import Timer
from Logger import Logger
from typing import Callable
import numpy as np

class VariableNeighborhoodSearch:
    def __init__(self, input_data: InputData, rng: np.random.Generator, timer: Timer, logger: Logger) -> None:
        self.input_data = input_data
        self.rng = rng
        self.timer = timer
        self.logger = logger

    def initialize(self, params: dict) -> None: 
        self.neighborhood_evaluation_strategy = params["neighborhood_evaluation_strategy"]
        self.neighborhood_change_procedure = params["neighborhood_change_procedure"]
        self.search_procedure = params["search_procedure"]
        if "alpha_neighborhood_change" in params:
            self.alpha_neighborhood_change = params["alpha_neighborhood_change"]
        else:
            self.alpha_neighborhood_change = 0

        # vns bookkeeping for neighborhoods
        self.neighborhood_types = params["neighborhood_types"]
        self.neighborhoods = {}
        self.current_neighborhood_type_idx = 0

        # bookkeping only needed for general vns (search procedure is not local search)
        if params["search_procedure"] not in ["LocalSearch", "Skip"]:
            self.vnd_neighborhoods = {}
            self.current_vnd_neighborhood_type_idx = 0

            # if vnd_neighborhood_types is not set then use the same neighborhood types as the vns‚
            if "vnd_neighborhood_types" in params:
                self.vnd_neighborhood_types = params["vnd_neighborhood_types"]
            else:
                self.vnd_neighborhood_types = self.neighborhood_types

        else:
            self.vnd_neighborhood_types = None
            self.vnd_neighborhoods = None
            self.current_vnd_neighborhood_type_idx = None

    def create_neighborhood(self, neighborhood_type: str) -> Neighborhood:
        if "Swap" in neighborhood_type:
            neighborhood = K_SwapNeighborhood(self.input_data, self.rng, self.logger, self.timer, neighborhood_type)
        elif "Insertion" in neighborhood_type:
            neighborhood = K_InsertionNeighborhood(self.input_data, self.rng, self.logger, self.timer, neighborhood_type)
        elif "Cut" in neighborhood_type:
            neighborhood = K_CutNeighborhood(self.input_data, self.rng, self.logger, self.timer, neighborhood_type)
        elif "Permute" in neighborhood_type:
            neighborhood = K_PermuteNeighborhood(self.input_data, self.rng, self.logger, self.timer, neighborhood_type)
        elif "Block" in neighborhood_type:
            neighborhood = K_BlockNeighborhood(self.input_data, self.rng, self.logger, self.timer, neighborhood_type)
        else:
            error_message = f"Neighborhood type {neighborhood_type} not defined."
            self.logger.log(error_message, self.timer.get_time(), "error")
            raise ValueError(error_message)
        return neighborhood

    def initialize_neighborhoods(self) -> None:
        """First initializes all the neighborhoods for the VNS. If vnd is used then the neighborhoods are initialized for the VND"""

        for neighborhood_type in self.neighborhood_types:
            if self.timer.is_time_up():
                self.logger.log(f"Time is up. Neighborhoods for VNS not created.", self.timer.get_time())
                return
            self.logger.log(f"Creating neighborhood {neighborhood_type} for VNS.", self.timer.get_time(), "debug")
            neighborhood = self.create_neighborhood(neighborhood_type)
            self.neighborhoods[neighborhood_type] = neighborhood
            self.logger.log(f"Neighborhood {neighborhood_type} for VNS created.", self.timer.get_time(), "debug")
        
        # if vnd is not used then no installation of vnd neighborhoods needed
        if self.vnd_neighborhood_types is None:
            return

        for neighborhood_type in self.vnd_neighborhood_types:
            if self.timer.is_time_up():
                self.logger.log(f"Time is up. Neighborhoods for VNS not created.", self.timer.get_time())
                return
            # if neighborhood got already initialized for vns then use it for vnd
            if neighborhood_type in self.neighborhoods:
                self.logger.log(f"Neighborhood {neighborhood_type} reused for VND.", self.timer.get_time(), "debug")
                self.vnd_neighborhoods[neighborhood_type] = self.neighborhoods[neighborhood_type]
            else:
                self.logger.log(f"Creating neighborhood {neighborhood_type} for VND.", self.timer.get_time(), "debug")
                neighborhood = self.create_neighborhood(neighborhood_type)
                self.vnd_neighborhoods[neighborhood_type] = neighborhood
                self.logger.log(f"Neighborhood {neighborhood_type} for VND created.", self.timer.get_time(), "debug")
    
    def run(self, solution: Solution, update_solution: Callable[[Solution], bool]) -> None:
        self.initialize_neighborhoods()
        i = 1

        while self.timer.is_time_left(): 
            # only shake if not first iteration and time is left
            solution_shake = self.shake(solution) if i > 1 and self.timer.is_time_left() else solution 
            self.logger.log(f"Shaked solution: {solution_shake}", self.timer.get_time(), "debug")
            # only local search if time is left
            solution_search = self.local_search_procedure(solution_shake) if self.timer.is_time_left() else solution_shake 
            self.logger.log(f"Local searched solution: {solution_search}", self.timer.get_time(), "debug")
            # only neighborhood change if time is left
            new_solution = self.neighborhood_change(self.neighborhood_change_procedure, solution_search, solution, self.alpha_neighborhood_change) if self.timer.is_time_left() else solution_search 
            updated = update_solution(new_solution) # only updates if new solution is better than solution (new solution could be just the current solution)
            if updated:
                self.logger.log(f"Solution updated in VNS.", self.timer.get_time(), "debug")
                self.logger.log(f"New solution: {new_solution}", self.timer.get_time())
                solution = new_solution

            i += 1 
        self.logger.log(f"VNS finished after {i} iterations.", self.timer.get_time()) # last one is not completed because time is up
    
    def local_search_procedure(self, solution: Solution) -> Solution:
        """either local search or variable neighborhood descent (B-VND, P-VND, C-VND, U-VND)"""

        self.logger.log(f"Starting {self.search_procedure} procedure.", self.timer.get_time(), "debug")
        
        if self.search_procedure == "LocalSearch":
            current = self.neighborhoods[self.neighborhood_types[self.current_neighborhood_type_idx]]
            solution = current.local_search(self.neighborhood_evaluation_strategy, solution)
            solution.shaking_neighborhood = current.type # the same as search neighborhood for Base VNS

        elif self.search_procedure in ["B-VND", "C-VND", "P-VND"]:
            current_solution = solution
            while True:
                if self.timer.is_time_up():
                    self.logger.log(f"Time is up. {self.search_procedure} procedure stopped.", self.timer.get_time())
                    break

                current = self.vnd_neighborhoods[self.vnd_neighborhood_types[self.current_vnd_neighborhood_type_idx]]
                solution_search = current.local_search(self.neighborhood_evaluation_strategy, current_solution)

                # get the neighborhood that got used for shaking 
                solution_search.shaking_neighborhood = self.neighborhood_types[self.current_neighborhood_type_idx]

                # do the right neighborhood change procedure for the search procedure
                if self.search_procedure == "B-VND":
                    neighborhood_change_procedure = "Sequential"
                elif self.search_procedure == "C-VND":
                    neighborhood_change_procedure = "Cyclic"
                elif self.search_procedure == "P-VND":
                    neighborhood_change_procedure = "Pipe"
                self.neighborhood_change(neighborhood_change_procedure, solution_search, current_solution, for_vnd=True)

                if solution_search.is_better_than(current_solution):
                        current_solution = solution_search
                else:
                    # VND should not iterate over and over the neighborhoods unlike VNS
                    # only stop if all neighborhoods have been tried and no improvement was found
                    # B-VND resets to first neighborhood if no improvement was found but then its solution is better so it will not execute this
                    if self.current_vnd_neighborhood_type_idx == 0:
                        solution = current_solution
                        break

        elif self.search_procedure == "U-VND":
            # U-VND for First Improvement iterates over all neighborhoods in sequence and not random between the moves of all neighborhoods
            initial_solution = solution
            for neighborhood_type in self.vnd_neighborhood_types:
                current = self.vnd_neighborhoods[neighborhood_type]
                solution_search = current.local_search(self.neighborhood_evaluation_strategy, initial_solution)
                if solution_search.is_better_than(solution):
                    solution = solution_search
            # set the shaking neighborhood for the solution
            solution.shaking_neighborhood = self.neighborhood_types[self.current_neighborhood_type_idx]
        elif self.search_procedure == "Skip":
            # set the shaking neighborhood for the solution
            solution.shaking_neighborhood = self.neighborhood_types[self.current_neighborhood_type_idx]
            # do no search, just return the initial solution (Reduced VNS)
            self.logger.log(f"Skipping search procedure (Reduced VNS).", self.timer.get_time(), "debug")
            pass
        else:
            error_message = f"Invalid search procedure: {self.search_procedure}"
            self.logger.log(error_message, self.timer.get_time(), "error")
            raise ValueError(error_message)
    
        return solution

    
    def shake(self, current_solution: Solution) -> Solution:
        current_neighborhood = self.neighborhoods[self.neighborhood_types[self.current_neighborhood_type_idx]]
        return current_neighborhood.shake(current_solution)

    def _change_neighborhood_index(self, for_vnd: bool = False) -> None:
        """Increments the neighborhood index by 1 if inside the length of the neighborhood types list, otherwise resets to 0. 
        Either uses neighborhood bookkeeping for vns or vnd."""
        if for_vnd:
            self.current_vnd_neighborhood_type_idx = (self.current_vnd_neighborhood_type_idx + 1) % len(self.vnd_neighborhood_types)
        else:
            self.current_neighborhood_type_idx = (self.current_neighborhood_type_idx + 1) % len(self.neighborhood_types)

    def _reset_neighborhood_index(self, for_vnd: bool = False) -> None:
        """Resets the neighborhood index to 0."""
        if for_vnd:
            self.current_vnd_neighborhood_type_idx = 0
        else:
            self.current_neighborhood_type_idx = 0


    def neighborhood_change(self, neighborhood_change_procedure: str, solution_search: Solution, current_solution: Solution, alpha: float = 0, for_vnd: bool = False) -> Solution:
        """
        Makes a decision on:
        1. which neighborhood will be explored as the next
        2. whether some solution will be accepted as a new incumbent solution or not.

        Different change procedures:
        - Sequential: Reset to first neighborhood if better solution found, otherwise cycle to next
        - Cyclic: Always cycle to next neighborhood, update solution if better
        - Pipe: Keep same neighborhood if better solution found, otherwise cycle to next
        - Skewed: If alpha is not set to 0. Allows exploration of valleys far from the incumbent solution

        If for_vnd is True then the neighborhood change procedure is used for the VND.
        """
        self.logger.log(f"Starting {neighborhood_change_procedure} neighborhood change procedure for {"VND" if for_vnd else "VNS"}", self.timer.get_time(), "debug")
        is_better = solution_search.is_better_than(current_solution, alpha)

        if neighborhood_change_procedure == "Sequential":
            if is_better:
                current_solution = solution_search
                self._reset_neighborhood_index(for_vnd)
                self.logger.log(f"Solution search is better than current solution. Reset {"VND" if for_vnd else "VNS"} neighborhood type to first.", self.timer.get_time(), "debug")
            
            else:
                self._change_neighborhood_index(for_vnd)

        elif neighborhood_change_procedure == "Cyclic":
            self._change_neighborhood_index(for_vnd)
            if is_better:
                current_solution = solution_search
                self.logger.log(f"Solution search is better than current solution. Continue with next {"VND" if for_vnd else "VNS"} neighborhood.", self.timer.get_time(), "debug")

        elif neighborhood_change_procedure == "Pipe":
            if is_better:
                current_solution = solution_search
                self.logger.log(f"Solution search is better than current solution. Continue with the same {"VND" if for_vnd else "VNS"} neighborhood.", self.timer.get_time(), "debug")
            else:
                self._change_neighborhood_index(for_vnd)
            
        else:
            error_message = f"Invalid neighborhood change procedure: {neighborhood_change_procedure}"
            self.logger.log(error_message, self.timer.get_time(), "error")
            raise ValueError(error_message)

        return current_solution


