import matplotlib.pyplot as plt
import numpy as np

class Individual:
    LO = 0
    HI = 10
    NGENES = 13
    LEN = 2 ** NGENES
    MUT_PROBA = 0.2

    def __init__(self, value):
        self.value = value
        self.geno = Individual.pheno_geno(self.value)

    @staticmethod
    def pheno_geno(pheno):
        x = int((Individual.LEN - 1) / (Individual.HI - Individual.LO) *
                (pheno - Individual.LO))
        s = ""
        while x > 0:
            s += str(x % 2)
            x //= 2

        for _ in range(Individual.NGENES - len(s)):
            s = '0' + s
        return s

    @staticmethod
    def geno_pheno(geno):
        return Individual.LO + int(geno, base=2) * \
               (Individual.HI - Individual.LO) / (Individual.LEN - 1)

    def fitness(self):
        return (-1) * (2.5*self.value + 1.7) * np.sin(1.1*np.pi*self.value + 0.7)

    def mutate(self):
        geno = list(self.geno)
        for g in range(Individual.NGENES):
            if np.random.random() < Individual.MUT_PROBA:
                geno[g] = "1" if geno[g] == "0" else "0"

        self.geno = "".join(geno)
        self.value = Individual.geno_pheno(self.geno)


class Population:
    SIZE = 100
    CX_PROB = 0.6

    def __init__(self, initpop):
        self.population = initpop

    def __getitem__(self, key):
        return self.population[key]

    def selection(self, size):
        return sorted(self.population, key=lambda item: item.fitness(), reverse=True)[:size]

    @staticmethod
    def crossover(par1, par2):
        cut = np.random.randint(0, par1.NGENES)
        child1 = par1.geno[:cut] + par2.geno[cut:]
        child2 = par2.geno[:cut] + par1.geno[cut:]
        return (Individual(Individual.geno_pheno(child1)),
                Individual(Individual.geno_pheno(child2)))

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
    init_gen = [Individual(np.random.randint(Individual.LO, Individual.HI))
                for _ in range(Population.SIZE)]

    population = Population(init_gen)
    GENERATIONS = 50
    f = lambda x: (2.5*x + 1.7) * np.sin(1.1*np.pi*x + 0.7)

    stats_best = []
    stats_avg = []
    for generation in range(GENERATIONS):
        stats_best.append(population.best_fitness())
        stats_avg.append(population.avg_fitness())

        nextgen = []
        for i in range(Population.SIZE // 2):
            if np.random.random() < Population.CX_PROB:
                p1, p2 = population[i], population[-i-1]
                nextgen.extend(Population.crossover(p1, p2))

        for indiv in nextgen:
            indiv.mutate()

        population.extend(nextgen)
        population.reduce()

        if (generation % 5 == 0) or (generation == GENERATIONS - 1):
            fx = np.linspace(Individual.LO, Individual.HI, 100)
            fy = f(fx)
            plt.plot(fx, fy)

            x = [item.value for item in population.population]
            d = [f(item.value) for item in population.population]
            plt.scatter(x, d, c="red", s=8)
            plt.title(f"Generation {generation}")
            plt.show()

    plt.plot(list(range(GENERATIONS)), stats_best, label="Best")
    plt.plot(list(range(GENERATIONS)), stats_avg, label="Avg")
    plt.xlabel("generations")
    plt.ylabel("fitness")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
