"""
Safe printing utility to handle Unicode characters on Windows
"""

import sys


def safe_print(message: str):
    """
    Safely print Unicode messages, falling back to ASCII if needed
    
    Args:
        message: The message to print
    """
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)


def safe_print_error(message: str):
    """
    Safely print error messages to stderr
    
    Args:
        message: The error message to print
    """
    try:
        print(message, file=sys.stderr)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message, file=sys.stderr)