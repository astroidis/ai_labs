import csv
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random


class TravelingSalesman:
    def __init__(self, fname):
        self.fname = fname
        self.locations = []
        self.distances = []
        self.ncities = 0

        self.__read_data()

    def __read_data(self):
        with open(self.fname, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
            for row in reader:
                if row[0] in ("DISPLAY_DATA_SECTION", "NODE_COORD_SECTION"):
                    break

            for row in reader:
                if row[0] != "EOF":
                    del row[0]
                    self.locations.append(np.asarray(row, dtype=np.float32))
                else:
                    break

        self.ncities = len(self.locations)
        self.distances = [[0]*self.ncities for _ in range(self.ncities)]
        for i in range(self.ncities):
            for j in range(i+1, self.ncities):
                d = np.linalg.norm(self.locations[i] - self.locations[j])
                self.distances[i][j], self.distances[j][i] = d, d

    def total_distance(self, path):
        d = self.distances[path[0]][path[-1]]
        for i in range(len(path) - 1):
            d += self.distances[path[i]][path[i+1]]
        return d

    def plot(self, path):
        plt.scatter(*zip(*self.locations))
        locs = [self.locations[loc] for loc in path]
        locs.append(locs[0])
        plt.plot(*zip(*locs))
        plt.show()


if __name__ == "__main__":
    POPULATION_SIZE = 250
    GENERATIONS = 150
    MUT_PROB = 0.2
    CROSS_PROB = 0.8

    tsp = TravelingSalesman("bayg29.tsp")
    toolbox = base.Toolbox()

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox.register("random_order", random.sample, range(tsp.ncities), tsp.ncities)
    toolbox.register("create_individual", tools.initIterate, creator.Individual,
                     toolbox.random_order)
    toolbox.register("create_population", tools.initRepeat, list,
                     toolbox.create_individual, POPULATION_SIZE)
    f = lambda ind: (tsp.total_distance(ind),)
    toolbox.register("evaluate", f)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)

    stat = tools.Statistics(key=lambda ind: ind.fitness.values)
    stat.register("min", np.min)
    stat.register("avg", np.mean)

    population = toolbox.create_population()
    hof = tools.HallOfFame(1)

    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=CROSS_PROB, mutpb=MUT_PROB, ngen=GENERATIONS,
        stats=stat, halloffame=hof, verbose=False
    )

    best = hof[0]
    print("Best individual:", best)
    print("Best fitness:", best.fitness.values[0])

    tsp.plot(best)

    minpath, avgpath = logbook.select("min", "avg")
    plt.plot(minpath, label="Min")
    plt.plot(avgpath, label="Avg")
    plt.xlabel("generations")
    plt.ylabel("length of path")
    plt.title("path over generations")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()
