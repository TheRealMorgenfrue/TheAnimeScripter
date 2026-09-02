import math
from fractions import Fraction


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


def compute_timesteps(vfi_factor: int | float, current_index: int) -> list[float]:
    """Computes the timesteps between two frames where interpolated
    frames should be inserted.

    The algorithm works for all vfi_factor >= 1.

    Parameters
    ----------
    vfi_factor : int | float
        The frame interpolation factor.
    current_index : int
        The current frame index to compute timesteps for.

    Examples
    --------
    NOTE: The two given frames are denoted as frame 0 and frame 1.

    - Given `vfi_factor=2`, the timestep is `[0.5]` (the middle of frame 0 and frame 1).
    - Given `vfi_factor=2.5`, the timesteps alternates between `[0.4]` for frame 0 and `[0.2, 0.6]` for frame 1.
    - Given `vfi_factor=3`, the timesteps are `[0.33, 0.66]`.

    Returns
    -------
    list[float]
        A list of timesteps.
    """
    if isinstance(vfi_factor, float):
        factor = Fraction(vfi_factor).limit_denominator(100)
        vfi_factor_numerator = factor.numerator
        vfi_factor_denominator = factor.denominator
        next_index = current_index + 1
        output_start = (current_index * vfi_factor_numerator) // vfi_factor_denominator
        output_end = (next_index * vfi_factor_numerator) // vfi_factor_denominator
        return [
            (output_start + i * vfi_factor_denominator % vfi_factor_numerator)
            / vfi_factor_numerator
            for i in range(1, output_end - output_start)
        ]
    else:
        return [i / vfi_factor for i in range(1, vfi_factor)]
