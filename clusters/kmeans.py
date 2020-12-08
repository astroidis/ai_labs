import numpy as np
import matplotlib.pyplot as plt
from random import choice


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def points_mean(cluster):
    x = sum([p[0] for p in cluster]) / len(cluster)
    y = sum([p[1] for p in cluster]) / len(cluster)
    return (x, y)


def centers(points, k):
    cnts = []
    cnts.append(choice(points))
    for i in range(k-1):
        mean = points_mean(cnts)
        ds = np.array([])
        for p in points:
            ds = np.append(ds, distance(mean, p))
        cnts.append(points[ds.argmax()])
    return cnts


def assign_point(clusters, centers, point):
    ds = np.array([distance(point, cnt) for cnt in centers])
    clusters[ds.argmin()].append(point)
    return clusters


def kmean(points, k):
    clusters = [[] for _ in range(k)]
    cnts = centers(points, k)
    for p in points:
        clusters = assign_point(clusters, cnts, p)
        for i, _ in enumerate(cnts):
            if len(clusters[i]) > 0:
                cnts[i] = points_mean(clusters[i])
    return clusters


points = [
    (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 1), (3, 2), (2, 5), (2, 6), (3, 6),
    (3, 7), (4, 5), (4, 6), (5, 3), (5, 4), (6, 2), (6, 4), (7, 2), (7, 3), (7, 4)
]
xs = [p[0] for p in points]
ys = [p[1] for p in points]
plt.scatter(xs, ys)
plt.grid(True)
plt.show()

clusters = kmean(points, 3)
# print(clusters)

fig, ax = plt.subplots()
for clust in clusters:
    xs = [p[0] for p in clust]
    ys = [p[1] for p in clust]
    ax.scatter(xs, ys)

plt.show()
