import logging
import sys


def init_logger(name='simple_example', filename=None):
    """ Initialise a logger

    Parameters
    ----------
    name : str
        The name of the logger
    filename : str
        The filename to save the logger to

    Returns
    -------
    logger : logging.Logger
        The logger
    """
    # create logger
    logger = logging.getLogger(name)

    if filename is None:
        # create console handler and set level to debug
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # add formatter to stdout_handler
        stdout_handler.setFormatter(formatter)

        # add stdout_handler to logger
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
    else:
        logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info(f'Started logger named: {name}')

    return logger


def print_time(duration):
    """Print elapsed time in a user-friendly format HH:MM:SS

    Parameters
    ----------
    duration : float
        Duration to which convert

    Returns
    -------
    string : str
        String of duration converted into format DD day(s) HH:MM:SS
    """
    d = int(duration // (60 * 60 * 24))
    duration -= d * (60 * 60 * 24)
    h = int(duration // (60 * 60))
    duration -= h * (60 * 60)
    m = int(duration // 60)
    duration -= m * 60
    s = int(duration)
    duration -= s
    ms = f"{duration:.2f}".split('.')[-1]

    return f"{str(d).zfill(2)} day(s) {str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2):}.{ms}"
