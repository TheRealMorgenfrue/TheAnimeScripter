import math


def compute_resolution_padding(frame_axis: int, multiplier: int) -> int:
    """Returns the frame size, e.g., height, with padding.

    Parameters
    ----------
    frame_size : int
        The size of one axis of the frame in pixels, e.g. 720.
    multiplier : int
        The model multiplier, e.g. 64.

    Returns
    -------
    int
        The padded frame size, e.g. 768.
    """
    return math.ceil(frame_axis / multiplier) * multiplier
