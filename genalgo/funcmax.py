# find max f(t) = (1.1t - 1.7)cos(pi*t + 1.5) -9 <= t <= 9
# inputs:
#   size of population
#   max individual size (number of genes)
#   crossover probability
#   mutation probability
#   max number of generations
# outputs:
#   best ang average fitness in population
#   value of best individual

from random import randint, random
import matplotlib.pyplot as plt
from numpy import linspace, cos, pi

class Individual:
    LBOUND = -9
    HBOUND = 9
    NGENES = 10
    LEN = 2 ** NGENES
    MUT_PROBA = 0.25

    def __init__(self, value):
        self.value = value
        self.geno = Individual.pheno_geno(self.value)

    @staticmethod
    def pheno_geno(pheno):
        x = int((Individual.LEN - 1) / (Individual.HBOUND - Individual.LBOUND) *
                (pheno - Individual.LBOUND))
        s = ""
        while x > 0:
            s += str(x % 2)
            x //= 2

        return s

    @staticmethod
    def geno_pheno(geno):
        return int(geno, base=2)

    def fitness(self):
        # f(t) = (1.1t - 1.7)cos(pi*t + 1.5)
        return (1.1 * self.value - 1.7) * cos(pi * self.value + 1.5)

    def mutate(self):
        if random() < Individual.MUT_PROBA:
            gen_id = randint(0, Individual.NGENES)
            geno = list(self.geno)
            geno[gen_id] = "1" if geno[gen_id] == "0" else "0"
            self.geno = "".join(geno)


class Population:
    SIZE = 50

    def __init__(self, initpop):
        self.population = initpop

    def selection(self, size):
        return sorted(self.population, key=lambda item: item.fitness())[:size]

    @staticmethod
    def crossover(par1, par2):
        cut = randint(0, par1.NGENES)
        child1 = par1.geno[:cut] + par2.geno[cut:]
        child2 = par1.geno[cut:] + par2.geno[:cut]
        return (Individual(Individual.geno_pheno(child1)),
                Individual(Individual.geno_pheno(child2)))

    def extend(self, new):
        self.population.extend(new)

    def reduce(self):
        self.population.sort(key=lambda indiv: indiv.fitness())
        return self.population[:Population.SIZE]

    def best_fitness(self):
        fit = [indiv.fitness() for indiv in self.population]
        return max(fit)

    def avg_fitness(self):
        fit = [indiv.fitness() for indiv in self.population]
        return sum(fit) / Population.SIZE


init_gen = [Individual(randint(Individual.LBOUND, Individual.HBOUND))
            for _ in range(Population.SIZE)]

population = Population(init_gen)
GENERATIONS = 30

f = lambda x: (1.1 * x - 1.7) * cos(pi * x + 1.5)
x = linspace(-9, 9, 100)
y = (1.1 * x - 1.7) * cos(pi * x + 1.5)
plt.plot(x, y)
plt.show()

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
    input()
