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

colors = ["#FF6347", "#CD5C5C", "#FF8C00", "#FFD700", "#DAA520", "#BDB76B",
          "#9ACD32", "#32CD32", "#90EE90", "#00FA9A", "#66CDAA", "#20B2AA",
          "#5F9EA0", "#6495ED", "#7B68EE", "#DB7093", "#F4A460", "#B0C4DE"]
clusters, centers = kmeans(dots, 15)

fig, ax = plt.subplots()
for clust, col in zip(clusters, colors):
    xs = [p[0] for p in clust]
    ys = [p[1] for p in clust]
    ax.scatter(xs, ys, s=5, c=col)

ax.plot(*zip(*centers), "r^")

plt.show()
