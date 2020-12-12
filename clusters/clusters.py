class Graph:
    def __init__(self, vertices, edges, distances):
        self.vertices = vertices
        self.edges = edges
        self.distances = distances
        self.roots = {vert: (vert, 0) for vert in self.vertices}

    def __find(self, vert):
        while vert != self.roots[vert][0]:
            vert = self.roots[vert][0]
        return vert

    def __union(self, x, y):
        rx = self.roots[x]
        ry = self.roots[y]
        if rx[0] == ry[0]:
            return
        if rx[1] > ry[1]:
            self.roots[y] = rx
        else:
            if rx[1] == ry[1]:
                ry = (ry[0], rx[1] + 1)
            self.roots[x] = ry

    def get_mst(self, mode):
        if mode == "kruskal":
            return self.__kruskal()
        elif mode == "prim":
            return self.__prim()

    def __kruskal(self):
        mst = set()
        self.edges.sort(key=lambda e: self.distances[e])
        for edge in self.edges:
            # print(f"CUR EDGE {edge} len {self.distances[edge]}")
            if self.__find(edge[0]) != self.__find(edge[1]):
                # print("Added")
                mst.add((edge[0], edge[1]))
                self.__union(edge[0], edge[1])
        return mst

    def __min_distance(self, selected, complement):
        d = []
        for v1 in selected:
            for v2 in complement:
                if (v1, v2) in self.distances:
                    d.append( (self.distances[(v1, v2)], (v1, v2)) )
                elif (v2, v1) in self.distances:
                    d.append( (self.distances[(v2, v1)], (v2, v1)) )
        m = min(d, key=lambda v: v[0])
        return m[1]

    def __prim(self):
        mst = set()
        selected = set()
        selected.add(self.vertices[3])
        complement = set(self.vertices) - selected
        while len(mst) < len(self.vertices) - 1:
            min_edge = self.__min_distance(selected, complement)
            selected.add(min_edge[1] if min_edge[1] not in selected else min_edge[0])
            complement.remove(min_edge[1] if min_edge[1] in complement else min_edge[0])
            mst.add(min_edge)
            # print("ADDED EDGE", min_edge)
        return mst


def union(clusters):
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if clusters[i].intersection(clusters[j]):
                clusters[j].union(clusters[i])
                del clusters[i]


def make_clusters(graph):
    mst = graph.get_mst("prim")
    clusters = [set([vert]) for vert in graph.vertices]
    # print("MST start", mst)
    # print("CLUSTER start", clusters)
    while len(mst) > 0:
        m = min(mst, key=lambda e: graph.distances[e])
        # print("MIN edge", m)
        new = None
        for c in clusters:
            if (m[0] in c):
                c.add(m[1])
            elif m[1] in c:
                c.add(m[0])
        union(clusters)
        mst.remove(m)
        # print("MST", mst)
        # print("CLUSTERS", clusters)
        # input()
    return clusters


def main():
    vertices = [1, 2, 3, 4, 5]
    distances = {
        (1, 2): 11, (1, 3): 9, (1, 4): 7, (1, 5): 8,
        (2, 3): 15, (2, 4): 14, (2, 5): 15,
        (3, 4): 12, (3, 5): 14,
        (4, 5): 6
    }

    graph = Graph(vertices, list(distances.keys()), distances)
    # mst_k = graph.kruskal()
    # print(mst_k)

    # mst_p = graph.prim()
    # print(mst_p)
    print(make_clusters(graph))


if __name__ == "__main__":
    main()
