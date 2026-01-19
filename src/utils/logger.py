#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Logger Utility

Provides logging functionality for the PDB processor.
"""

from typing import Optional
from datetime import datetime


class Logger:
    """
    Logger class for handling debug information logging with colored output.
    
    Attributes:
        is_debug (bool): Whether debug mode is enabled
        log_file (Optional[str]): Path to log file
        module_name (str): Name of the module using this logger
    """
    def __init__(self, debug: bool = False, log_file: Optional[str] = None, module_name: str = ""):
        self.is_debug = debug
        self.log_file = log_file
        self.module_name = module_name
        
        # ANSI color codes for different log levels
        self.colors = {
            "INFO": "\033[94m",    # Blue
            "DEBUG": "\033[90m",   # Gray
            "WARNING": "\033[93m", # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"      # Reset color
        }

    def log(self, message: str, level: str = "INFO", indent: int = 0) -> None:
        """
        Log a message with proper formatting and color.
        
        Args:
            message (str): The message to log
            level (str): Log level (INFO, DEBUG, WARNING, ERROR)
            indent (int): Number of spaces to indent the message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        indent_str = " " * indent
        
        # Add module name if specified
        module_part = f"[{self.module_name}] " if self.module_name else ""
        
        # Format log message
        log_message = f"[{timestamp}] [{level}] {module_part}{indent_str}{message}"
        
        # Format console output with color
        colored_log = f"{self.colors.get(level, self.colors['RESET'])}{log_message}{self.colors['RESET']}"
        
        # Always print INFO, WARNING, and ERROR messages
        # Print DEBUG messages only in debug mode
        if level in ["INFO", "WARNING", "ERROR"] or self.is_debug:
            print(colored_log)
        
        # Write to log file if specified (without colors)
        if self.log_file:
            import os
            # Ensure log directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Only write important messages to log file (INFO, WARNING, ERROR)
            # Skip DEBUG messages to keep log file small, unless in debug mode
            if level in ["INFO", "WARNING", "ERROR"] or self.is_debug:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_message + "\n")

    def info(self, message: str, indent: int = 0) -> None:
        """
        Log an info message.
        
        Args:
            message (str): The message to log
            indent (int): Number of spaces to indent the message
        """
        self.log(message, "INFO", indent)

    def debug(self, message: str, indent: int = 0) -> None:
        """
        Log a debug message.
        
        Args:
            message (str): The message to log
            indent (int): Number of spaces to indent the message
        """
        self.log(message, "DEBUG", indent)

    def warning(self, message: str, indent: int = 0) -> None:
        """
        Log a warning message.
        
        Args:
            message (str): The message to log
            indent (int): Number of spaces to indent the message
        """
        self.log(message, "WARNING", indent)

    def error(self, message: str, indent: int = 0) -> None:
        """
        Log an error message.
        
        Args:
            message (str): The message to log
            indent (int): Number of spaces to indent the message
        """
        self.log(message, "ERROR", indent)

    def section(self, title: str) -> None:
        """
        Log a section title.
        
        Args:
            title (str): The section title
        """
        self.info(f"=== {title} ===")

    def table(self, headers: list, rows: list, indent: int = 0) -> None:
        """
        Log tabular data.
        
        Args:
            headers (list): Table headers
            rows (list): Table rows
            indent (int): Number of spaces to indent the table
        """
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, col in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(col)))
        
        # Format headers
        header_line = " | ".join([h.ljust(w) for h, w in zip(headers, widths)])
        separator_line = "-" * (len(header_line) + 2)
        
        # Log table
        self.info(header_line, indent)
        self.info(separator_line, indent)
        for row in rows:
            row_line = " | ".join([str(col).ljust(w) for col, w in zip(row, widths)])
            self.info(row_line, indent)
