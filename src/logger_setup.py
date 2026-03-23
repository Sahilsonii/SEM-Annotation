import sys
import os

class LoggerWriter:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logging(log_filename="app.log"):
    # Check if we are already logging to avoid recursion or double logging if executed multiple times
    if isinstance(sys.stdout, LoggerWriter):
        return

    # Open the log file in append mode
    log_file = open(log_filename, "a", encoding="utf-8")
    
    # Redirect stdout and stderr
    sys.stdout = LoggerWriter(sys.stdout, log_file)
    sys.stderr = LoggerWriter(sys.stderr, log_file)
    
    print(f"--- Logging started to {os.path.abspath(log_filename)} ---")
