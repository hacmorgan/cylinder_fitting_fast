import numpy as np

from . import geometry


def fitting_rmsd(w_fit, C_fit, r_fit, Xs):
    """Calculate the RMSD of fitting."""
    return np.sqrt(
        sum((geometry.point_line_distance(p, C_fit, w_fit) - r_fit) ** 2 for p in Xs)
        / len(Xs)
    )


def distance_to_cylinder(w_fit, C_fit, r_fit, Xs):
    """Calculate the distance of each point to the cylinder."""
    return np.array(
        [(geometry.point_line_distance(p, C_fit, w_fit) - r_fit) ** 2 for p in Xs]
    )
