#!/usr/bin/env python3
"""
Logging utilities for consistent, beautiful output across all scripts.

This module provides formatted logging functions that make it easy to see
what's happening during pipeline execution.
"""

import logging
import sys
from typing import Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section_header(title: str, char: str = "=", width: int = 100):
    """
    Print a prominent section header.
    
    Args:
        title (str): Section title
        char (str): Character to use for border (default: "=")
        width (int): Width of the header (default: 100)
    """
    print("\n" + char * width)
    print(f"{Colors.BOLD}{Colors.OKBLUE}  {title}{Colors.ENDC}")
    print(char * width + "\n")


def print_step(step_num: int, step_name: str, description: str = ""):
    """
    Print a step indicator.
    
    Args:
        step_num (int): Step number
        step_name (str): Name of the step
        description (str): Optional description
    """
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[STEP {step_num}] {step_name}{Colors.ENDC}")
    if description:
        print(f"  → {description}")
    print("-" * 100)


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str, indent: int = 0):
    """
    Print an info message.
    
    Args:
        message (str): Message to print
        indent (int): Number of spaces to indent (default: 0)
    """
    indent_str = " " * indent
    print(f"{indent_str}  {message}")


def print_progress(current: int, total: int, item_name: str = "item"):
    """
    Print progress indicator.
    
    Args:
        current (int): Current item number
        total (int): Total number of items
        item_name (str): Name of the item being processed
    """
    percentage = (current / total) * 100
    print(f"  [{current}/{total}] ({percentage:.1f}%) Processing {item_name}...")


def print_results_summary(results: dict, title: str = "Results Summary"):
    """
    Print a formatted results summary.
    
    Args:
        results (dict): Dictionary of results to display
        title (str): Title for the summary
    """
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 80)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:<30} {value:>10.3f}")
        elif isinstance(value, int):
            print(f"  {key:<30} {value:>10}")
        else:
            print(f"  {key:<30} {str(value):>10}")
    print("-" * 80)


def setup_script_logging(script_name: str, log_file: Optional[str] = None):
    """
    Set up logging for a script with consistent formatting.
    
    Args:
        script_name (str): Name of the script
        log_file (str, optional): Path to log file
    """
    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    # Print script header
    print_section_header(f"Running: {script_name}")


def print_file_info(file_path: str, file_type: str = "file"):
    """
    Print information about a file being processed.
    
    Args:
        file_path (str): Path to the file
        file_type (str): Type of file (e.g., "input", "output", "config")
    """
    print(f"  📄 {file_type.capitalize()}: {file_path}")


def print_dataset_info(dataset_name: str, n_samples: int, n_features: Optional[int] = None):
    """
    Print dataset information.
    
    Args:
        dataset_name (str): Name of the dataset
        n_samples (int): Number of samples
        n_features (int, optional): Number of features
    """
    info = f"  📊 Dataset: {dataset_name} | Samples: {n_samples}"
    if n_features:
        info += f" | Features: {n_features}"
    print(info)


def print_completion(script_name: str, output_files: list = None):
    """
    Print completion message with output files.
    
    Args:
        script_name (str): Name of the script
        output_files (list, optional): List of output file paths
    """
    print("\n" + "="*100)
    print(f"{Colors.BOLD}{Colors.OKGREEN}✓ {script_name} completed successfully!{Colors.ENDC}")
    
    if output_files:
        print(f"\n{Colors.BOLD}Output files:{Colors.ENDC}")
        for i, file_path in enumerate(output_files, 1):
            print(f"  {i}. {file_path}")
    
    print("="*100 + "\n")


class ProgressLogger:
    """Context manager for logging progress through a list of items."""
    
    def __init__(self, items: list, item_name: str = "item"):
        """
        Initialize progress logger.
        
        Args:
            items (list): List of items to process
            item_name (str): Name of the item type
        """
        self.items = items
        self.item_name = item_name
        self.total = len(items)
        self.current = 0
    
    def __enter__(self):
        print(f"\n  Processing {self.total} {self.item_name}(s)...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print_success(f"Completed processing {self.total} {self.item_name}(s)")
        return False
    
    def log_item(self, item_name: str):
        """Log progress for current item."""
        self.current += 1
        print_progress(self.current, self.total, f"{self.item_name}: {item_name}")


# Example usage
if __name__ == "__main__":
    # Demonstrate the logging utilities
    setup_script_logging("Example Script")
    
    print_step(1, "Data Loading", "Loading input data files")
    print_file_info("/path/to/input.csv", "input")
    print_dataset_info("HCP-Dev", 1000, 246)
    print_success("Data loaded successfully")
    
    print_step(2, "Processing", "Running analysis")
    
    with ProgressLogger(["dataset1", "dataset2", "dataset3"], "dataset") as progress:
        for dataset in ["dataset1", "dataset2", "dataset3"]:
            progress.log_item(dataset)
    
    print_step(3, "Results", "Saving output")
    results = {
        "R²": 0.856,
        "MAE": 2.34,
        "N": 1000
    }
    print_results_summary(results)
    
    print_completion("Example Script", [
        "/path/to/output1.png",
        "/path/to/output2.csv"
    ])

