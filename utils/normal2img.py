import numpy as np


class Normal2Img(object):
    """Convert gray image to rgb representation for plotting

    Args:
        img: Normalized image
    Returns:
        img: Denormalized image
    """
    def __init__(self):
        """
        """

    def __call__(self, norm):
        """
        """
        invalid_mask = norm > 254
        norm = (norm + 1.0) / 2.0
        norm[invalid_mask] = 0  # invalid regions in gt normal
        return norm
