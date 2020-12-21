import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class Individual:
    LOW = -100
    HI = 100
    MUT_PROBA = 0.3

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def fitness(self):
        return (-1) * -np.cos(self.x1)*np.cos(self.x2)*np.exp(-(self.x1-np.pi)**2 - (self.x2-np.pi)**2)

    def mutate(self):
        if np.random.random() < Individual.MUT_PROBA:
            self.x1 = np.random.normal(0.0, 1.0)
            self.x2 = np.random.normal(0.0, 1.0)


class Population:
    SIZE = 200
    ALPHA = 1 + np.random.random()
    BETA = 2 - ALPHA
    CX_PROB = 0.5

    def __init__(self, initpop):
        self.population = initpop

    def __getitem__(self, key):
        return self.population[key]

    def selection(self, size, method):
        if method == "rank":
            return self.__rank()
        elif method == "tournament":
            return self.__tournament()

    def __rank(self):
        self.population = sorted(self.population, key=lambda x: x.fitness(),
                                 reverse=True)
        t = lambda i: ((Population.ALPHA - Population.BETA) *
                       ((i - 1) / (Population.SIZE - 1)))
        ranks = [1/Population.SIZE * (Population.ALPHA - t(i))
                 for i, _ in enumerate(self.population)]
        next = []
        for _ in range(Population.SIZE):
            for r, item in zip(ranks, self.population):
                if np.random.random() < r:
                    next.append(item)
        return next

    def __tournament(self):
        next = []
        for _ in range(Population.SIZE):
            contestants = np.random.choice(self.population, 5)
            contestants = sorted(contestants, key=lambda x: x.fitness(),
                                 reverse=True)
            next.append(contestants[0])
        return next

    @staticmethod
    def crossover(par1, par2):
        cross_proba = np.random.random()
        c1x1 = cross_proba * par1.x1 + (1 - cross_proba) * par2.x2
        c1x2 = cross_proba * par1.x2 + (1 - cross_proba) * par2.x1

        c2x1 = cross_proba * par2.x1 + (1 - cross_proba) * par1.x2
        c2x2 = cross_proba * par2.x2 + (1 - cross_proba) * par2.x1
        return (Individual(c1x1, c1x2), Individual(c2x1, c2x2))

    def extend(self, new):
        self.population.extend(new)

    def reduce(self):
        self.population = sorted(self.population,
                                 key=lambda indiv: indiv.fitness(),
                                 reverse=True)[:Population.SIZE]

    def best_fitness(self):
        fit = [indiv.fitness() for indiv in self.population]
        return max(fit)

    def avg_fitness(self):
        fit = [indiv.fitness() for indiv in self.population]
        return sum(fit) / len(self.population)


def main():
    GENERATIONS = 80

    init_gen = []
    for _ in range(Population.SIZE):
        x1, x2 = np.random.randint(Individual.LOW, Individual.HI, size=2)
        init_gen.append(Individual(x1, x2))

    population = Population(init_gen)

    f = lambda x1, x2: -np.cos(x1)*np.cos(x2)*np.exp(-(x1-np.pi)**2 - (x2-np.pi)**2)
    x = np.linspace(0, 6, 50)
    y = np.linspace(0, 6, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    # ax = plt.subplot(111, projection="3d")
    # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()

    maxfit = []
    avgfit = []
    for generation in range(GENERATIONS):
        maxfit.append(population.best_fitness())
        avgfit.append(population.avg_fitness())

        nextgen = []
        parents = population.selection(Population.SIZE // 2, "tournament")
        for i in range(Population.SIZE // 2):
            if np.random.random() < Population.CX_PROB:
                p1, p2 = parents[i], parents[-i-1]
                nextgen.extend(Population.crossover(p1, p2))

        for indiv in nextgen:
            indiv.mutate()

        population.extend(nextgen)
        population.reduce()

        print("Best individual ({:.5f}, {:.5f}), f={:.5f}".format(
            population.population[0].x1,
            population.population[0].x2,
            population.population[0].fitness()
        ))

    g = list(range(GENERATIONS))
    plt.plot(g, avgfit, label="Average")
    plt.plot(g, maxfit, label="Best")
    plt.legend()
    plt.xlabel("generations")
    plt.ylabel("fitness")
    plt.grid(True)
    plt.title("Tournament selection")
    # plt.title("Rank selection")
    plt.show()


if __name__ == '__main__':
    main()
