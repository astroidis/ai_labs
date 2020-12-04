import numpy as np
from deap import creator, base, tools, algorithms
from random import randint
import matplotlib.pyplot as plt


class Knapsack:
    DATA = [  # (item, weight, value)
        ("map", 9, 150), ("compass", 13, 35),
        ("water", 153, 200), ("sandwich", 50, 160),
        ("glucose", 15, 60), ("tin", 68, 45),
        ("banana", 27, 60), ("apple", 39, 40),
        ("cheese", 23, 30), ("beer", 52, 10),
        ("suntan cream", 11, 70), ("camera", 32, 30),
        ("t-shirt", 24, 15), ("trousers", 48, 10),
        ("umbrella", 73, 40), ("waterproof trousers", 42, 70),
        ("waterproof overclothes", 43, 75), ("note-case", 22, 80),
        ("sunglasses", 7, 20), ("towel", 18, 12),
        ("socks", 4, 50), ("book", 30, 10)
    ]

    def __init__(self, capacity):
        self.capacity = capacity

    def get_value(self, items):
        accvalue = 0
        accweight = 0
        for included, item in zip(items, Knapsack.DATA):
            if included:
                if accweight + item[1] <= self.capacity:
                    accweight += item[1]
                    accvalue += item[2]
        return accvalue, accweight

    def __len__(self):
        return len(self.items)

    def print_items(self, items):
        accvalue = 0
        accweight = 0
        for included, item in zip(items, Knapsack.DATA):
            if included:
                if accweight + item[1] <= self.capacity:
                    accweight += item[1]
                    accvalue += item[2]
                    print(f"{item[0]} (weight={item[1]} value={item[2]})")

        print(f"Total weight = {accweight} Total value = {accvalue}")


def main():
    POPULATION_SIZE = 50
    GENERATIONS = 50
    MUT_PROBA = 0.1
    CROSS_PROBA = 0.1

    sack = Knapsack(400)
    toolbox = base.Toolbox()

    creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMaxMin)

    toolbox.register("zero_or_one", randint, 0, 1)

    toolbox.register("create_individual", tools.initRepeat, creator.Individual,
                     toolbox.zero_or_one, len(Knapsack.DATA))
    toolbox.register("create_population", tools.initRepeat, list,
                     toolbox.create_individual, POPULATION_SIZE)

    fitness = lambda individual: sack.get_value(individual)
    toolbox.register("evaluate", fitness)
    # toolbox.register("select", tools.selRoulette)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selBest)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population = toolbox.create_population()
    hof = tools.HallOfFame(1)

    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=CROSS_PROBA, mutpb=MUT_PROBA, ngen=GENERATIONS,
        stats=stats, verbose=False, halloffame=hof
    )

    maxfit, avgfit = logbook.select("max", "avg")

    best = hof[0]
    print("Best:", best)
    print("Best fitness:", best.fitness.values[0])
    print("Best pack:")
    sack.print_items(best)

    plt.plot(maxfit, color="red", label="Max")
    plt.plot(avgfit, color="green", label="Avg")
    plt.xlabel("generations")
    plt.ylabel("fitness")
    plt.grid(True)
    plt.legend(loc="best")
    # plt.title("fitness over generations (roulette)")
    plt.title("fitness over generations (tournament)")
    # plt.title("fitness over generations (best)")
    plt.show()


if __name__ == "__main__":
    main()
