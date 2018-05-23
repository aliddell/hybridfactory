# Copyright (C) 2018 Vidrio Technologies. All rights reserved.


def log(msg, stdout, in_progress=False):
    """

    Parameters
    ----------
    msg : str
        Message to log.
    stdout : bool
        Print to stdout if True.
    in_progress : bool, optional
        Print newline if and only if True.

    """

    end = " ... " if in_progress else "\n"

    if stdout:
        print(msg, end=end)
