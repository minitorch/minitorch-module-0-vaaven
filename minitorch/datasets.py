import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N):
    """Generate N random 2D points in the unit square [0,1] x [0,1].

    Args:
        N (int): Number of points to generate

    Returns:
        List[Tuple[float, float]]: List of (x1, x2) coordinate pairs

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Dataset container for 2D point classification problems.

    Attributes:
        N (int): Number of data points
        X (List[Tuple[float, float]]): List of (x1, x2) coordinate pairs
        y (List[int]): List of binary class labels (0 or 1)

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N):
    """Simple vertical split dataset.

    Creates a binary classification dataset where points are classified
    based on their x1 coordinate. Points with x1 < 0.5 are class 1,
    points with x1 >= 0.5 are class 0. This creates a vertical
    decision boundary at x1 = 0.5.

    Args:
        N (int): Number of data points to generate

    Returns:
        Graph: Dataset with N points and binary labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N):
    """Diagonal split dataset.

    Creates a binary classification dataset where points are classified
    based on their sum x1 + x2. Points with x1 + x2 < 0.5 are class 1,
    points with x1 + x2 >= 0.5 are class 0. This creates a diagonal
    decision boundary at x1 + x2 = 0.5.

    Args:
        N (int): Number of data points to generate

    Returns:
        Graph: Dataset with N points and binary labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N):
    """Split dataset with two vertical bands.

    Creates a binary classification dataset where points are classified
    based on their x1 coordinate. Points with x1 < 0.2 OR x1 > 0.8 are
    class 1, points with 0.2 <= x1 <= 0.8 are class 0. This creates
    two vertical decision boundaries at x1 = 0.2 and x1 = 0.8.

    Args:
        N (int): Number of data points to generate

    Returns:
        Graph: Dataset with N points and binary labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N):
    """XOR dataset.

    Creates a binary classification dataset implementing the XOR function.
    Points are class 1 if they are in the top-left quadrant (x1 < 0.5, x2 > 0.5)
    OR bottom-right quadrant (x1 > 0.5, x2 < 0.5). Points in the other
    two quadrants are class 0. This creates a non-linearly separable dataset.

    Args:
        N (int): Number of data points to generate

    Returns:
        Graph: Dataset with N points and binary labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N):
    """Circular dataset.

    Creates a binary classification dataset where points are classified
    based on their distance from the center (0.5, 0.5). Points with
    distance > sqrt(0.1) from center are class 1, points closer to center
    are class 0. This creates a circular decision boundary.

    Args:
        N (int): Number of data points to generate

    Returns:
        Graph: Dataset with N points and binary labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N):
    """Spiral dataset.

    Creates a binary classification dataset with two interleaving spirals.
    Points are generated along two spiral curves - one clockwise and one
    counter-clockwise. The first spiral (class 0) starts from center and
    spirals outward clockwise. The second spiral (class 1) starts from center
    and spirals outward counter-clockwise. This creates a highly non-linear
    decision boundary.

    Args:
        N (int): Number of data points to generate (should be even)

    Returns:
        Graph: Dataset with N points and binary labels

    """

    def x(t):
        return t * math.cos(t) / 20.0

    def y(t):
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


# Dictionary mapping dataset names to their generator functions
datasets = {
    "Simple": simple,  # Vertical split at x1 = 0.5
    "Diag": diag,  # Diagonal split at x1 + x2 = 0.5
    "Split": split,  # Two vertical bands (x1 < 0.2 or x1 > 0.8)
    "Xor": xor,  # XOR pattern (non-linearly separable)
    "Circle": circle,  # Circular boundary around center
    "Spiral": spiral,  # Two interleaving spirals
}
