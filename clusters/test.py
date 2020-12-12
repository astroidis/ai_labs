import matplotlib.pyplot as plt
import csv
import numpy as np
from kmeans import kmeans


dots = []
with open("s1.txt", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
    for row in reader:
        dots.append(np.asarray(row, dtype=np.float32))


# plt.scatter(*zip(*dots), s=5)
# plt.show()

clusters, centers = kmeans(dots, 13)

fig, ax = plt.subplots()
for clust in clusters:
    xs = [p[0] for p in clust]
    ys = [p[1] for p in clust]
    ax.scatter(xs, ys, s=5)

ax.plot(*zip(*centers), "r^")

plt.show()
