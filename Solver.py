import numpy as np
import json
from InputData import *
from Solution import *
from ConstructiveHeuristic import *
from VariableNeighborhoodSearch import *    
from Timer import *
from Logger import *
import os

class Solver:
    def __init__(self, input_data_path: str, seed: int, consider_minimum_distance: bool, constructive_params: dict, vns_params: dict, time_limit: int = 5) -> None:
        self.timer = Timer(time_limit)
    
        self.input_data = InputData(input_data_path, consider_minimum_distance)
        self.rng = np.random.default_rng(seed)
        
        self.consider_minimum_distance = consider_minimum_distance
        self.solution = None
        self.solution_history = []
        self.constructive_params = constructive_params
        self.vns_params = vns_params

    def update_solution(self, solution: Solution) -> bool:
        # only update if solution is better or if it is the first solution
        if solution.is_better_than(self.solution):
            self.solution = solution
            self.solution_history.append({"time": self.timer.get_time(),"cost":solution.cost, "search_neighborhood": solution.search_neighborhood, "shaking_neighborhood": solution.shaking_neighborhood, "permutation": solution.permutation.tolist()})
            return True
        return False

    def _construction_phase(self) -> None:
        self.logger.log(f"Starting construction phase.", self.timer.get_time())

        constructive_heuristic = ConstructiveHeuristic(self.input_data, self.rng, self.timer, self.logger)  
        constructive_heuristic.initialize(self.constructive_params)
        constructive_heuristic.run(self.update_solution)

        self.logger.log(f"Constructive solution found.\n{self.solution}", self.timer.get_time())

    def _variable_neighborhood_search_phase(self) -> None:
        self.logger.log(f"Starting variable neighborhood search phase.", self.timer.get_time())

        vns = VariableNeighborhoodSearch(self.input_data, self.rng, self.timer, self.logger)
        vns.initialize(self.vns_params)
        vns.run(self.solution, self.update_solution)

        self.logger.log(f"Best found Solution.\n{self.solution}", self.timer.get_time())

    def run_search(self) -> None:
        # set end time at the start of the search
        self.timer.start_timer()
        # instantiate logger here to close file when search is finished
        self.logger = Logger()
        self.logger.log(f"Starting search.", self.timer.get_time())

        self._construction_phase()
        self._variable_neighborhood_search_phase()

    def save_solution_and_logs(self) -> None:
        self.logger.log(f"Saving solution and logs to file.", self.timer.get_time())
        title = f"{self.input_data.name}_Lsg_{"m" if self.consider_minimum_distance else "o"}MA"
        json_file = f"./Solutions/{title}.json"
        
        # Check if file exists and only save if current solution is better
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                existing_solution = json.load(f)
                if existing_solution["ObjVal"] <= int(self.solution.cost):
                    self.logger.log(f"Solution is not better than existing solution. Skipping save.", self.timer.get_time())
                    return
                
        # Save solution if it's better or file doesn't exist
        with open(json_file, "w") as f:
            solution_dict = {"ObjVal": int(self.solution.cost), "Permutation": self.solution.permutation.tolist()}
            json.dump(solution_dict, f)

        # save log file to file inside the SolutionLogs folder
        log_file = f"./SolutionLogs/{title}.log"
        with open(log_file, "w") as f:
            f.write(self.logger.content)



        

        



