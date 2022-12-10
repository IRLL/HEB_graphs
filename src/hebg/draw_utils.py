import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def PointsInCircum(eachPoint, r, n=100):
    return [
        (
            eachPoint[0] + math.cos(2 * math.pi / n * x) * r,
            eachPoint[1] + math.sin(2 * math.pi / n * x) * r,
        )
        for x in range(0, n + 1)
    ]


def bufferPoints(inPoints, stretch, n):
    newPoints = []
    for eachPoint in inPoints:
        newPoints += PointsInCircum(eachPoint, stretch, n)
    newPoints = np.array(newPoints)
    newBuffer = ConvexHull(newPoints)

    return newPoints[newBuffer.vertices]


def draw_convex_hull(points, ax: "Axes", stretch=0.3, n_points=30, **kwargs):
    points = np.array(points)
    convh = ConvexHull(points)  # Get the first convexHull (speeds up the next process)
    points = bufferPoints(points[convh.vertices], stretch=stretch, n=n_points)

    hull = ConvexHull(points)
    hull_cycle = np.concatenate((hull.vertices, hull.vertices[:1]))
    ax.plot(points[hull_cycle, 0], points[hull_cycle, 1], **kwargs)
