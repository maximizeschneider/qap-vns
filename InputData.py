import json
import numpy as np

class InputData:
    def __init__(self, path: str, consider_minimum_distance: bool = False) -> None:
        self.path = path
        self.name = path.split("/")[-1].split(".")[0]
        self.consider_minimum_distance = consider_minimum_distance
        self.data_load()
        

    def data_load(self) -> None:

        with open(self.path, "r") as input_file:
            input_data = json.load(input_file)
        
        self.n = input_data["Size"]
    
        self.F = np.array(input_data["A"])
        self.D = np.array(input_data["D"])
        self.M = np.array(input_data["M"])
