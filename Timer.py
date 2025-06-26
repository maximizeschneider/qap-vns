from timeit import default_timer

# helper class
class Timer:
    def __init__(self, time_limit: int) -> None:
        self.time_limit = time_limit
    
    def is_time_left(self) -> bool:
        return default_timer() < self.end_time
    
    def is_time_up(self) -> bool:
        return default_timer() > self.end_time
    
    def start_timer(self) -> None:
        self.start_time = default_timer()
        self.end_time = default_timer() + self.time_limit
    
    def get_time(self) -> float:
        return default_timer() - self.start_time