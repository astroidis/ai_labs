import matplotlib.pyplot as plt
import csv
import numpy as np
from kmeans import kmeans
from trees import make_clusters, Graph
from sklearn.datasets import make_blobs


def test_kmeans():
    dots = []
    with open("a1.txt", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for _ in range(1500):
            row = next(reader)
            dots.append(np.asarray(row, dtype=np.float32))

    colors = ["#FF6347", "#CD5C5C", "#FF8C00", "#FFD700", "#DAA520", "#BDB76B",
              "#9ACD32", "#32CD32", "#90EE90", "#00FA9A", "#66CDAA", "#20B2AA",
              "#5F9EA0", "#6495ED", "#7B68EE", "#DB7093", "#F4A460", "#B0C4DE"]
    clusters, centers = kmeans(dots, 10)

    fig, ax = plt.subplots()
    for clust, col in zip(clusters, colors):
        xs = [p[0] for p in clust]
        ys = [p[1] for p in clust]
        ax.scatter(xs, ys, s=5, c=col)

    ax.plot(*zip(*centers), "r+")

    plt.show()


def test_trees():
    dots = []
    with open("spiral.txt", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", skipinitialspace=True)
        for row in reader:
            dots.append(np.asarray(row[:2], dtype=np.float32))

    vertices = [i for i in range(len(dots))]
    edges = [(i, j) for i in range(len(dots)) for j in range(len(dots))]
    distances = np.zeros( shape=(len(vertices), len(vertices)) )
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            d = np.linalg.norm(dots[i] - dots[j])
            distances[i, j] = d
            distances[j, i] = d
    graph = Graph(vertices, edges, distances)

    clusters = make_clusters(graph, 3)

    colors = ["#FF6347", "#6495ED", "#00FA9A", "#CD5C5C", "#FF8C00", "#FFD700"]
    for clust, col in zip(clusters, colors):
        for p in clust:
            plt.scatter(dots[p][0], dots[p][1], s=8, c=col)

    plt.show()


if __name__ == "__main__":
    test_kmeans()
    # test_trees()
