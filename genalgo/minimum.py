import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class Individual:
    LOW = -100
    HI = 100
    MUT_PROBA = 0.35

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def fitness(self):
        # return -np.cos(self.x1) * np.cos(self.x2) * \
        #        np.exp(-(self.x1 - np.pi)**2 - (self.x2 - np.pi)**2)
        X = self.x1
        Y = self.x2
        return -(X**2 + 2*Y**2 - 0.3*np.cos(3*np.pi*X) - 0.4*np.cos(4*np.pi*Y)+0.7)
        # a = np.sin(np.sqrt(self.x1**2 + self.x2**2))**2 - 0.5
        # b = (1 + 0.001*(self.x1**2 + self.x2**2))**2
        # return (-1) * (0.5 + a / b)

    def mutate(self):
        if np.random.random() < Individual.MUT_PROBA:
            self.x1 = np.random.normal(0.0, 1.0)
            self.x2 = np.random.normal(0.0, 1.0)


class Population:
    SIZE = 100
    ALPHA = 1 + np.random.random()
    BETA = 2 - ALPHA

    def __init__(self, initpop):
        self.population = initpop

    def selection(self, size, method):
        if method == "rank":
            return self.rank_selection(size)
        elif method == "tournament":
            return self.tournament_selection(size)

    def rank_selection(self, size):
        self.population = sorted(self.population, key=lambda x: x.fitness(),
                                 reverse=True)
        t = lambda i: ((Population.ALPHA - Population.BETA) *
                       ((i - 1) / (Population.SIZE - 1)))
        ranks = [1/Population.SIZE * (Population.ALPHA - t(i))
                 for i, _ in enumerate(self.population)]
        next = []
        while size > 0:
            for r, item in zip(ranks, self.population):
                if np.random.random() < r:
                    next.append(item)
                    size -= 1
        return next

    def tournament_selection(self, size):
        next = []
        while size > 0:
            contestants = np.random.choice(self.population, 5)
            contestants = sorted(contestants, key=lambda x: x.fitness(),
                                 reverse=True)
            next.append(contestants[0])
            size -= 1
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
    GENERATIONS = 40

    init_gen = []
    for _ in range(Population.SIZE):
        x1, x2 = np.random.randint(Individual.LOW, Individual.HI, size=2)
        init_gen.append(Individual(x1, x2))

    population = Population(init_gen)

    # f = lambda X, Y: -np.cos(X) * np.cos(Y) * \
    #     np.exp(-(X - np.pi)**2 - (Y - np.pi)**2)  # 5
    f = lambda X, Y: X**2 + 2*Y**2 - 0.3*np.cos(3*np.pi*X) - \
                     0.4*np.cos(4*np.pi*Y)+0.7  # 6 (0)
    # f = lambda x1, x2: (0.5 + np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5 /
    #                     (1 + 0.001*(x1**2 + x2**2))**2)  # 2 (0)
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    # # ax = plt.subplot(111, projection="3d")
    # # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()

    maxfit = []
    avgfit = []
    for generation in range(GENERATIONS):
        # print("\nGeneration", generation)
        # print("Max Fitness", population.best_fitness())
        # print("Average Fitness", population.avg_fitness())
        maxfit.append(population.best_fitness())
        avgfit.append(population.avg_fitness())

        nextgen = []
        parents = population.selection(Population.SIZE // 2, "rank")
        for i in range(len(parents) // 2):
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
    # plt.title("Tournament selection")
    plt.title("Rank selection")
    plt.show()


if __name__ == '__main__':
    main()
