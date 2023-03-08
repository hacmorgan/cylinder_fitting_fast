import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from .geometry import rotation_matrix_from_axis_and_angle
from . import fitting
from .analysis import distance_to_cylinder

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def show_G_distribution(data):
    """Show the distribution of the G function."""
    Xs, t = fitting.preprocess_data(data)

    Theta, Phi = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50))
    G = []

    for i in range(len(Theta)):
        G.append([])
        for j in range(len(Theta[i])):
            w = fitting.direction(Theta[i][j], Phi[i][j])
            G[-1].append(fitting.G(w, Xs))

    plt.imshow(G, extent=[0, np.pi, 0, 2 * np.pi], origin="lower")
    plt.show()


def show_fit(w_fit, C_fit, r_fit, Xs):
    """Plot the fitting given the fitted axis direction, the fitted
    center, the fitted radius and the data points.
    """
    data = np.array(Xs)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the data points

    ax.scatter([X[0] for X in Xs], [X[1] for X in Xs], [X[2] for X in Xs])

    # Get the transformation matrix

    theta = np.arccos(np.dot(w_fit, np.array([0, 0, 1])))
    phi = np.arctan2(w_fit[1], w_fit[0])

    M = np.dot(
        rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
        rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta),
    )

    xd = np.max(data[:, 0]) - np.min(data[:, 0])
    yd = np.max(data[:, 1]) - np.min(data[:, 1])
    zd = np.max(data[:, 2]) - np.min(data[:, 2])
    c = np.sqrt(xd**2 + zd**2 + yd**2)

    # Plot the cylinder surface
    delta = np.linspace(-np.pi, np.pi, 20)
    z = np.linspace(-c / 2, c / 2, 20)

    Delta, Z = np.meshgrid(delta, z)
    X = r_fit * np.cos(Delta)
    Y = r_fit * np.sin(Delta)

    for i in range(len(X)):
        for j in range(len(X[i])):
            p = np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]])) + C_fit

            X[i][j] = p[0]
            Y[i][j] = p[1]
            Z[i][j] = p[2]

    ax.plot_surface(X, Y, Z, alpha=0.5, color="green")

    # Plot the center and direction
    ax.quiver(
        C_fit[0],
        C_fit[1],
        C_fit[2],
        r_fit * w_fit[0],
        r_fit * w_fit[1],
        r_fit * w_fit[2],
        color="red",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plt.axis('equal')
    ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
    ax.set_proj_type(
        "ortho"
    )  # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax)  # IMPORTANT - this is also required
    plt.show()
