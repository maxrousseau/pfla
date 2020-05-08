import logging

class Logger:
    """Logging functionality

    Parameters
    ----------
    VERBOSE : boolean
        Value from the verbose argument
    """
    def __init__(self, VERBOSE):
        self.verbose = VERBOSE

    def info(self, MSG, LEVEL):
        """Log message based on level

        Parameters
        ----------
        MSG : string
            Message to be logged
        LEVEL : int
            Level of the message: (0 = info, 1 = error)
        """
        msg = MSG
        level = LEVEL

        if self.verbose:
            logging.basicConfig(level=logging.DEBUG)

        else:
            logging.basicConfig(level=logging.WARNING)

        if level == 0:
            logging.info(msg)

        elif level == 1:
            logging.error(msg)

        else:
            None
