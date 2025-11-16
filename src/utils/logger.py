"""
Logging configuration with fancy formatting
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        # Format timestamp
        record.asctime = self.formatTime(record, self.datefmt)
        
        return super().format(record)


def setup_logger(
    name: str = 'smadex', 
    log_file: str = 'logs/training.log',
    use_colors: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        use_colors: Whether to use colored output in console
        
    Returns:
        Configured logger
    """
    
    # Create logs directory
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if use_colors:
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (no colors)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def log_section(logger: logging.Logger, title: str, width: int = 80):
    """Log a fancy section header"""
    logger.info("=" * width)
    logger.info(f"{title:^{width}}")
    logger.info("=" * width)


def log_subsection(logger: logging.Logger, title: str, width: int = 80):
    """Log a fancy subsection header"""
    logger.info("-" * width)
    logger.info(f"  {title}")
    logger.info("-" * width)


def log_metric(logger: logging.Logger, name: str, value: float, format_str: str = ".6f"):
    """Log a metric with fancy formatting"""
    logger.info(f"  ✓ {name:.<50} {value:{format_str}}")


def log_progress(logger: logging.Logger, step: int, total: int, message: str = ""):
    """Log progress with percentage"""
    percentage = (step / total) * 100
    bar_length = 40
    filled = int(bar_length * step / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    logger.info(f"  [{bar}] {percentage:5.1f}% - {message}")
