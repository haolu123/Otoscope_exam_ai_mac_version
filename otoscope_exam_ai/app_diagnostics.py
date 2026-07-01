import faulthandler
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "otoscope_exam_ai"
_crash_file = None


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_path = Path(sys.executable).resolve()
        if (
            sys.platform == "darwin"
            and executable_path.parent.name == "MacOS"
            and executable_path.parent.parent.name == "Contents"
        ):
            return executable_path.parents[3]
        return executable_path.parent
    return Path(__file__).resolve().parent


def setup_diagnostics(root: Path | None = None) -> logging.Logger:
    global _crash_file

    result_path = (root or runtime_root()) / "result"
    result_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            result_path / "application.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
        )
        logger.addHandler(handler)

    if _crash_file is None:
        _crash_file = (result_path / "native_crash.log").open("a", encoding="utf-8")
        faulthandler.enable(file=_crash_file, all_threads=True)

    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception

    if hasattr(threading, "excepthook"):
        def handle_thread_exception(args):
            logger.critical(
                "Uncaught thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = handle_thread_exception

    logger.info("Application diagnostics initialized")
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
