from datetime import datetime
from typing import Literal

class Logger:
    def __init__(self, log_file_path: str = "./VNS.log") -> None:
        self.log_file = open(log_file_path, "a")
        # delete all content of the file
        self.log_file.truncate(0)
        self.content = ""

    def log(self, message: str, time: str, level: Literal["info", "debug", "error"] = "info") -> None:
        """info prints to console and writes to file, debug/error only write to file"""
        if level == "info":
            print(message)
        log_message = f"[{time}] {level.upper()}: {message}\n"
        self.content += log_message
        self.log_file.write(log_message)
        self.log_file.flush() # immediately write to file for live updates

    def __del__(self) -> None:
        # close file when logger doesn't exist anymore
        self.log_file.close()
