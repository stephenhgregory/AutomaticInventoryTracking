"""
Contains logging and testing functions for:
    - Benchmarking performance of various steps of the pipeline
    - Logging debugging information to the console
"""
import os
import re
from functools import wraps
from typing import List
import logging


def time_logger(orig_func):
    """Logs the total execution time of functions to a file"""
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time()
        with open(f'time_logs/{orig_func.__name__}.log', 'a') as file:
            file.write(f'{orig_func.__name__} Ran in {t2 - t1} seconds.\n')
        return result

    return wrapper


def average(my_list: List[float]) -> float:
    """Returns the average of a list of floats"""
    return sum(my_list) / len(my_list)


def log_average_run_times(logfile_path: str) -> None:
    """Logs the average run time from a log file of run times"""

    # Open the file for reading and appending
    with open(logfile_path, 'r+') as file:
        # Read the entire file, all lines, at once
        text = file.read()
        # Read all floating point numbers from the text file (each is seconds elapsed)
        all_seconds_elapsed = re.findall(r"\d*\.\d+e?[+-]?\d*|\d+", text)
        # Convert the list of strings to a list of floats
        all_seconds_elapsed = [float(seconds_elapsed) for seconds_elapsed in all_seconds_elapsed]
        # Get the average seconds of function elapse
        average_seconds = average(all_seconds_elapsed)
        # Write the average seconds to the same log file
        file.write(f'Average run time: {average_seconds:.5f} seconds')


def log_all_average_run_times(logdir_path: str):
    """Logs average run times for all log files in a log directory"""
    for filename in os.listdir(logdir_path):
        if filename.endswith(".log"):
            log_average_run_times(os.path.join(logdir_path, filename))


def log_num_text_strings(img_name: str = '', num_text_strings: int = 0,
                         log_filename: str = 'barcode_scanning_results.log') -> None:
    """
    Logs the number of text strings found for an image

    Parameters
    ----------
    img_name: The filename of the image
    num_text_strings: The number of text strings found in the image
    log_filename:

    Returns
    -------
    None

    """
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    logging.debug(f'{num_text_strings} text strings found in {img_name}')


def log_barcodes(barcodes: any) -> None:
    """
    Logs barcode detections to a logger file
    Parameters
    ----------
    barcodes: contains the detected barcodes from "image"

    Returns
    -------
    None

    """
    for barcode in barcodes:
        # print the barcode type and data to the terminal
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        print(f'[INFO] Found {barcode_type} barcode: {barcode_data}')


if __name__ == "__main__":
    log_all_average_run_times("./time_logs")
    # log_average_run_times("./logs/detect_text_boxes.logs")
