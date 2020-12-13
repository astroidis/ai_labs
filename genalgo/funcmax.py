from random import randint, random
import matplotlib.pyplot as plt
import numpy as np

class Individual:
    LO = -9 #-6  # -9
    HI = 9  #6  # 9
    NGENES = 10
    LEN = 2 ** NGENES
    MUT_PROBA = 0.25

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
        # return (1.1 * self.value - 1.7) * np.cos(np.pi * self.value + 1.5)
        # return (1.3*self.value + 1.9) * np.cos(1.1*np.pi*self.value - 1.5)
        return (0.5*self.value - 1.4) * np.cos(0.5*np.pi*self.value + 1.1)

    def mutate(self):
        if random() < Individual.MUT_PROBA:
            gen_id = randint(0, Individual.NGENES-1)
            geno = list(self.geno)
            geno[gen_id] = "1" if geno[gen_id] == "0" else "0"
            self.geno = "".join(geno)
            self.value = Individual.geno_pheno(self.geno)


class Population:
    SIZE = 70

    def __init__(self, initpop):
        self.population = initpop

    def selection(self, size):
        return sorted(self.population, key=lambda item: item.fitness(), reverse=True)[:size]

    @staticmethod
    def crossover(par1, par2):
        cut = randint(0, par1.NGENES)
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
        return sum(fit) / len(self.population) # Population.SIZE


def main():
    init_gen = [Individual(randint(Individual.LO, Individual.HI))
                for _ in range(Population.SIZE)]

    population = Population(init_gen)
    GENERATIONS = 20

    for generation in range(GENERATIONS):
        print("\nGeneration", generation)
        print("Max Fitness", population.best_fitness())
        print("Average Fitess", population.avg_fitness())

        nextgen = []
        parents = population.selection(Population.SIZE // 2)
        for i in range(len(parents) // 2):
            p1, p2 = parents[i], parents[-i-1]
            nextgen.extend(Population.crossover(p1, p2))

        for indiv in nextgen:
            indiv.mutate()

        population.extend(nextgen)
        population.reduce()

        if (generation % 5 == 0) or (generation == GENERATIONS - 1):
            # f = lambda x: (1.1 * x - 1.7) * np.cos(np.pi * x + 1.5)  # 4
            # f = lambda x: (1.3*x + 1.9) * np.cos(1.1*np.pi*x - 1.5)  # 2
            f = lambda x: (0.5*x - 1.4) * np.cos(0.5*np.pi*x + 1.1)
            fx = np.linspace(Individual.LO, Individual.HI, 100)
            fy = f(fx)
            plt.plot(fx, fy)

            x = [item.value for item in population.population]
            d = [f(item.value) for item in population.population]
            plt.scatter(x, d, c="red", s=8)
            plt.title(f"Generation {generation}")
            plt.show()


if __name__ == '__main__':
    main()
