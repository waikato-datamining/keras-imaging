import enum


class Interpolation(enum.Enum):
    """
    The types of interpolations that can be performed to calculate pixel values from
    the (non-grid-aligned) offset pixels after applying the diffeomorphism transform.
    """
    BILINEAR = 0,
    GAUSSIAN = 1
