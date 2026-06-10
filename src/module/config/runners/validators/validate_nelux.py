import nelux


def validate_nelux_loglevel(loglevel: str) -> str:
    """Ensure `loglevel` follows NeLux's loglevel specification.

    Parameters
    ----------
    loglevel : str
        The loglevel, e.g. "DEBUG".

    Returns
    -------
    str
        The loglevel, if valid.

    Raises
    ------
    ValueError
        The loglevel is invalid.
    """
    loglevel = loglevel.lower()
    if loglevel not in nelux.LogLevel._member_names_:
        err_msg = (
            f"Invalid log level '{loglevel}'. "
            + f"Expected one of '{nelux.LogLevel._member_names_}'"
        )
        raise ValueError(err_msg)
    return loglevel
