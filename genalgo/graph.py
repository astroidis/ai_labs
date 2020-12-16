import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms


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
        for i in range(len(color_list)-1):
            for j in range(i+1, len(color_list)):
                if self.ajdmatrix[i, j]:
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
    HC_PENALTY = 10
    GENERATIONS = 50
    POPULATION_SIZE = 100
    MUT_PROB = 0.2
    CX_PROB = 0.2
    MAX_COLORS = 10

    toolbox = base.Toolbox()
    gcp = GraphColoring(nx.petersen_graph(), HC_PENALTY)
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox.register("randint", np.random.randint, 0, MAX_COLORS)
    toolbox.register("create_individual", tools.initRepeat, creator.Individual,
                     toolbox.randint, len(gcp))
    toolbox.register("create_population", tools.initRepeat, list,
                     toolbox.create_individual, POPULATION_SIZE)

    f = lambda x: (gcp.cost(x),)
    toolbox.register("evaluate", f)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=MAX_COLORS-1, indpb=MUT_PROB)
    toolbox.register("mate", tools.cxTwoPoint)

    stat = tools.Statistics(key=lambda ind: ind.fitness.values)
    stat.register("min", np.min)
    stat.register("avg", np.mean)

    population = toolbox.create_population()
    hof = tools.HallOfFame(1)

    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=GENERATIONS,
        stats=stat, halloffame=hof, verbose=False
    )

    best = hof[0]
    print("Best individual:", best)
    print("Best fitness:", best.fitness.values[0])
    print("Number of colors:", gcp.number_of_colors(best))
    print("Violations count:", gcp.violations(best))
    print("Cost:", gcp.cost(best))

    plt.figure(1)
    gcp.plot(best)

    minfit, avgfit = logbook.select("min", "avg")
    plt.figure(2)
    plt.plot(minfit, label="Min")
    plt.plot(avgfit, label="Avg")
    plt.xlabel("generations")
    plt.ylabel("fitness")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
