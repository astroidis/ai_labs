import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class GraphColoring:
    def __init__(self, graph, hardpenalty):
        self.graph = graph
        self.hard_penalty = hardpenalty
        self.nodes = list(self.graph.nodes)
        self.ajdmatrix = nx.adjacency_matrix(graph).todense()

    def __len__(self):
        return nx.number_of_nodes(self.graph)

    def cost(self, color_list):
        return self.hard_penalty * self.violations(color_list) + self.number_of_colors(color_list)

    def violations(self, color_list):
        violations = 0
        for i in range(len(color_list)):
            for j in range(i+1, color_list):
                if self.ajdmatrix[i][j]:
                    if color_list[i] == color_list[j]:
                        violations += 1
        return violations

    def number_of_colors(self, color_list):
        return len(set(color_list))

    def plot(self, color_list):
        used_colors = list(set(color_list))
        # convert color numbers to actual colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(used_colors)))
        colormap = []
        for i in range(self.__len__()):
            color = colors[used_colors.index(color_list[i])]
            colormap.append(color)

        nx.draw_kamada_kawai(self.graph, node_color=colormap, with_labels=True)
        return plt


if __name__ == "__main__":
    pass
