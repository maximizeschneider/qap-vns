import numpy as np
from Solution import Solution
from helper import *
from typing import Callable
from InputData import InputData
from Timer import Timer
from Logger import Logger

class ConstructiveHeuristic:
    def __init__(self, input_data: InputData, rng: np.random.Generator, timer: Timer, logger: Logger) -> None:
        self.input_data = input_data
        self.rng = rng
        self.timer = timer
        self.logger = logger
        self.disable_special_order = False

    def initialize(self, params: dict) -> None:
        self.constructive_method = params["constructive_method"]
        self.disable_special_order = params.get("disable_special_order", False)
    
    def random(self, times: int = 1) -> Solution:
        """Generate multiple random permutations and put them in a Solution object. Returns the best solution."""
        best_solution = None
        for i in range(times):
            if self.timer.is_time_up():
                self.logger.log(f"Time is up at iteration {i} in random constructive heuristic.", self.timer.get_time())
                break
            permutation = self.rng.permutation(self.input_data.n)
            solution = Solution(permutation, self.input_data)
            if solution.is_better_than(best_solution):
                best_solution = solution
        return best_solution
    
    # a facility that has big flows to other facilities should be placed at a location that has the smallest distance to other locations
    def ordered_greedy(self) -> Solution:
        """Greedy construction with dynamic ordering (Müller-Merbach Optimale Reihenfolge)"""
        
        # calculate total flow for each facility and total distance for each location, matrix and its transpose to get flow i to j and j to i, axis 1 is the sum over the rows, should be the same as axis 0
        total_flow = np.sum(self.input_data.F + self.input_data.F.T, axis=1)
        total_distance = np.sum(self.input_data.D + self.input_data.D.T, axis=1)
        
        # order facilities by descending total flow and locations by ascending total distance
        facility_order = np.argsort(-total_flow) 
        location_order = np.argsort(total_distance)
        
        # assign each facility to a location
        permutation = -np.ones(self.input_data.n, dtype=int)
        for facility, location in zip(facility_order, location_order):
            permutation[facility] = location

        return Solution(permutation, self.input_data)


    def dynamic_ordered_greedy(self) -> Solution:
        """
        Greedy construction with dynamic ordering (Müller-Merbach Optimale Reihenfolge).
        """

        n = self.input_data.n

        # make matrices symmetric 
        D = self.input_data.D+self.input_data.D.T
        F = self.input_data.F+self.input_data.F.T

        # bookkeeping with int arrays for indexing
        used_locations = np.array([], dtype=int)
        used_facilities = np.array([], dtype=int)
        free_locations = np.arange(n, dtype=int)
        free_facilities = np.arange(n, dtype=int)
        permutation = -np.ones(n, dtype=int) 
       
        # get the facility with the highest flow and the location with the lowest distance
        total_flow = np.sum(F, axis=1)
        total_distance = np.sum(D, axis=1)
        first_facility = np.argmax(total_flow)
        first_location = np.argmin(total_distance)
        permutation[first_facility] = first_location

        used_locations = np.append(used_locations, first_location)
        used_facilities = np.append(used_facilities, first_facility)
    
        # remove first location and facility from free locations and facilities
        free_locations = free_locations[free_locations != first_location]
        free_facilities = free_facilities[free_facilities != first_facility]

        # place the next facility with the highest flow and the next location with the lowest distance relative to the already used facilities and locations until all locations are used
        # flow/distance between used and unused facilities/locations
        # matrices are symmetric, so we need only sum for rows of used and columns of unused locations/facilities (otherwise we would need the vice versa too)
        while len(used_locations) < n:  
            # get a subset of the matrices with the rows of used locations/facilities and columns of free locations/facilities
            D_subset = D[np.ix_(used_locations, free_locations)]
            F_subset = F[np.ix_(used_facilities, free_facilities)]
            
            # sum over the rows (add values with same column index) to get values from used location to free locations (and back cause of transpose earlier)
            total_distance = np.sum(D_subset, axis=0)
            total_flow= np.sum(F_subset, axis=0)
            
            # index of free locations/facilities to get the next one
            next_location = free_locations[np.argmin(total_distance)]
            next_facility = free_facilities[np.argmax(total_flow)]
            permutation[next_facility] = next_location

            used_locations = np.append(used_locations, next_location)
            used_facilities = np.append(used_facilities, next_facility)
            free_locations = free_locations[free_locations != next_location]
            free_facilities = free_facilities[free_facilities != next_facility]

        return Solution(permutation, self.input_data)
    

   # all others have decreasing degree of freedom
    def increasing_dof(self, times: int = 1, facility_order: np.ndarray | None = None) -> Solution:
        """
        Greedy construction with increasing degree of freedom (Müller-Merbach Optimale Reihenfolgen 1970).
        First the degrees of freedom (possible moves per facility) are increasing, but decrease again at some point. (k+1)*(n-k) for k facilities already placed.
        Calculates (n^3 + 3n^2 - 4n)/6 possibilities. For n=25 this is 2900.
        Either provide a facility order or let the algorithm choose a random order. Can be executed multiple times and returns the best solution.
        If times and facility_order are provided, the provided facility order is used for the first iteration and then the random order is used for the remaining iterations.
        If minimum distance is considered, a special facility order is added to the first iteration to try to avoid inf solutions. (in this case provided facility_order won't be used).
        The special facility order is the order of facilities with the highest sum of constraints to have them placed first.
        """

        n = self.input_data.n
        best_solution = None

        # get a facility order that places facilities with highest sum of constraints first, they should have influence on all the placing afterwards and avoid inf solutions
        # one execution is reserved for this specific facility order
        if self.input_data.consider_minimum_distance and not self.disable_special_order:
            facility_order = np.argsort(-np.sum(self.input_data.M + self.input_data.M.T, axis = 1))
            
        for i in range(times):
            if self.timer.is_time_up():
                self.logger.log(f"Time is up at iteration {i} in increasing degree of freedom greedy constructive heuristic.", self.timer.get_time())
                break

            # order of facilities to assign
            # if facility_order exists and it's first iteration, use that order otherwise use random permutation
            facilities = facility_order if (facility_order is not None and i == 0) else self.rng.permutation(n)
            permutation = -np.ones(n, dtype=int)

            # place facilities in the order of the random permutation
            for i, facility in enumerate(facilities): 
                # bookkeeping
                used_locations = set(permutation[permutation >= 0]) # filter out -1
                free_locations = [location for location in range(n) if location not in used_locations]
                placed_facilities = facilities[:i] 

                best_cost = np.inf
                best_move = None

                # check all moves which place the facility in a free location
                for free_location in free_locations:
                    # place first facility in first free location cause it has always 0 total distance anyway 
                    if i == 0:
                        best_move = ("place", free_location)
                        break
                    
                    tryout_permutation = permutation.copy()
                    tryout_permutation[facility] = free_location
                    cost = Solution.calculate_cost(tryout_permutation, self.input_data) 
                    if cost < best_cost:
                        best_cost = cost
                        best_move = ("place", free_location)

                # check all moves which swap a facility with a free location and place the displaced facility in a free location 
                # combine all facilities which are already placed and all locations which are free to moves 
                # iterate over placed facilities instead of used locations because the placed facility is needed to change the permutation 
                for placed_facility in placed_facilities:
                    location_of_placed_facility = permutation[placed_facility]
                    for free_location in free_locations:
                        tryout_permutation = permutation.copy()
                        # swap the facility with the free location
                        tryout_permutation[facility] = location_of_placed_facility 
                        # place the displaced facility in a free location
                        tryout_permutation[placed_facility] = free_location 
                        cost = Solution.calculate_cost(tryout_permutation, self.input_data) 
                        if cost < best_cost:
                            best_cost = cost
                            best_move = ("swap", placed_facility, location_of_placed_facility, free_location)
                        

                # if no best move is found, place the facility in the first free location (if minimum distance is considered, all delta objectives could be inf)
                if best_move is None:
                    permutation[facility] = free_locations[0]
                else:
                    # make the best move (change the actual permutation)
                    if best_move[0] == "place":
                        _, free_location = best_move
                        permutation[facility] = free_location
                    elif best_move[0] == "swap":
                        _, placed_facility, location_of_placed_facility, free_location = best_move
                        permutation[facility] = location_of_placed_facility
                        permutation[placed_facility] = free_location

            solution = Solution(permutation, self.input_data)
            if solution.is_better_than(best_solution):
                best_solution = solution
        
        return best_solution


        
    def run(self, update_solution: Callable[[Solution], bool]) -> None:
        self.logger.log(f"Generating an initial solution according to {self.constructive_method}.", self.timer.get_time())

        solution = None
        method = self.constructive_method
        if method == "random":
            solution = self.random()
        elif method == "ordered_greedy":
            solution = self.ordered_greedy()
        elif method == "dynamic_ordered_greedy":
            solution = self.dynamic_ordered_greedy()
        elif  method == "increasing_dof":
            solution = self.increasing_dof()
        elif  "-random" in method: 
            times = int(method.split("-")[0])
            solution = self.random(times)
        elif  "-increasing_dof" in method:
            times = int(method.split("-")[0])
            solution = self.increasing_dof(times)
        else:
            error_message = f"Invalid solution method: {method}"
            self.logger.log(error_message, self.timer.get_time(), "error")
            raise ValueError(error_message)
       
        updated = update_solution(solution)

        if updated:
            self.logger.log(f"Solution updated in constructive phase.", self.timer.get_time())

       
            
       
            
