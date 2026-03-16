"""
Base class providing logging functionality and common utilities.

This module contains the Base class that serves as a foundation for other classes
in the ifc-hydro system, providing centralized logging capabilities and resource
path management.
"""

from datetime import datetime as time
import sys
import os


class Base:
    """
    Base class providing logging functionality and common utilities.

    This class serves as a foundation for other classes in the IfcHydro system,
    providing centralized logging capabilities and resource path management.

    Attributes:
        _log (str): Name of the log file (class variable)
        _counter (int): Instance counter for tracking object creation (class variable)
    """

    _log     = "ifc-hydro"      # name of log file
    _counter = 0                # instance counter

    def __init__(self, log: str = ""):
        """
        Initialize the Base class instance.

        Args:
            log (str, optional): Custom log file name. If empty, uses default log file.
        """

        if log != "":
            Base._log = log

        Base._counter += 1

    def configure_log(cls, log_dir: str = "", log_name: str = "") -> str:
        """
        Configure the log file location and name for the current run.

        Args:
            log_dir (str, optional): Directory where the log file should be stored.
                Defaults to the current working directory.
            log_name (str, optional): Log file name. Defaults to "ifc-hydro.log".

        Returns:
            str: The full path to the configured log file.
        """

        if log_name == "":
            log_name = "ifc-hydro"

        if log_dir == "":
            log_dir = os.getcwd()

        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        cls._log = os.path.join(log_dir, log_name+".log")

        Base.append_log(Base, f">>> Project {log_name}:")
        Base.append_log(Base, f">>> New run started at {time.now().strftime('%d/%m/%Y %H:%M:%S')}.")
        Base.append_log(Base, f"{'-'*100}")

        return cls._log

    def append_log(self, text: str):
        """
        Append a timestamped message to the log file and print to console.

        Args:
            text (str): The message to log
        """
        t = time.now()
        tstamp = "%2.2d.%2.2d.%2.2d " % (t.hour, t.minute, t.second)

        otext = tstamp + text
        with open(Base._log, "a") as f:
            f.write(otext + "\n")
        print(otext)

    def resource_path(self, relative_path: str) -> str:
        """
        Retrieve the absolute path of resources used in the application.

        This method handles both development and PyInstaller bundled environments.

        Args:
            relative_path (str): The relative path to the resource

        Returns:
            str: The absolute path to the resource
        """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)
