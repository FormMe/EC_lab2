import numpy as np
import operator
import random


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, p):
        return np.sqrt(abs(self.x - p.x) ** 2 + abs(self.y - p.y) ** 2)


def fitness_func(route):
    print("route", route)
    pathDistance = 0.0
    for i, fromCity in enumerate(route):
        toCity = route[(i + 1) % len(route)]
        pathDistance += fromCity.distance(toCity)
    return 1 / pathDistance


def create_individ(cities):
    return random.sample(cities, len(cities))


def create_population(pop_size, cities):
    return [create_individ(cities) for _ in range(pop_size)]


def rank_individs(pop):
    fitness = {i: fitness_func(individ)
               for i, individ in enumerate(pop)}
    return np.array(sorted(fitness.items(),
                           key=operator.itemgetter(1),
                           reverse=False))


class TSPGeneticAlgo:
    def __init__(self, cities, pop_size=100, elite_size=20, mutation_rate=0.5):
        self.cities = cities
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.pop = create_population(pop_size, cities)
        self.pop_ranked = rank_individs(self.pop)
        self.history = []

    def selection(self):
        selected = self.pop_ranked[:self.elite_size, 0]
        proba = self.pop_ranked[:, 1] / sum(self.pop_ranked[:, 1])
        selected = np.concatenate((selected,
                                   np.random.choice(self.pop_ranked[:, 0],
                                                    len(self.pop_ranked) - self.elite_size,
                                                    p=proba)))
        return selected

    def cross(self, parent1, parent2):
        gene_a = int(random.random() * len(parent1))
        gene_b = int(random.random() * len(parent1))
        if gene_b == gene_a:
            gene_a = (gene_b + 1) % len(parent1)

        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)

        child1 = [parent1[i] for i in range(start_gene, end_gene)]
        child2 = [item for item in parent2 if item not in child1]
        return child1 + child2

    def cross_population(self, parents):
        pop = parents[:self.elite_size]
        for _ in range(self.pop_size - self.elite_size):
            parent1 = np.random.randint(0, len(parents))
            parent2 = np.random.randint(0, len(parents))
            if parent2 == parent1:
                parent2 = (parent2 + 1) % len(parents)
            child = self.cross(self.pop[parent2], self.pop[parent1])
            np.append(pop, child)
        return pop

    def mutate(self, individual):
        for swapped in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                swap_with = int(np.random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swap_with]

                individual[swapped] = city2
                individual[swap_with] = city1
        return individual

    # def mutate(self, pop):
    #     tmp = []
    #     for i, individ in enumerate(pop):
    #         if i < self.elite_size // 2:
    #             if np.random.random() < 0.0:
    #                 city1 = np.random.randint(0, len(individ))
    #                 city2 = np.random.randint(0, len(individ))
    #
    #                 individ[city1], individ[city2] = individ[city2], individ[city1]
    #
    #         elif i > self.elite_size // 2 and i < self.elite_size:
    #             if np.random.random() < self.mutation_rate / 2:
    #                 city1 = np.random.randint(0, len(individ))
    #                 city2 = np.random.randint(0, len(individ))
    #
    #                 individ[city1], individ[city2] = individ[city2], individ[city1]
    #
    #         elif np.random.random() < self.mutation_rate:
    #             city1 = np.random.randint(0, len(individ))
    #             city2 = np.random.randint(0, len(individ))
    #
    #             individ[city1], individ[city2] = individ[city2], individ[city1]
    #
    #         tmp.append((individ))
    #
    #     return tmp

    def eval(self, generations=1):
        for i in range(generations):
            selected = self.selection()
            new_pop = self.cross_population(selected)
            new_pop_mut = self.mutate(new_pop)
            print(new_pop_mut)
            self.pop = new_pop_mut
            self.pop_ranked = rank_individs(self.pop)
            self.history.append(self.pop_ranked[0][1])

            print('Best result now: {}'.format(self.pop_ranked[0][1]))

    def draw_path(self, individ):
        xs = [i.x for i in self.pop[individ]]
        ys = [i.y for i in self.pop[individ]]
        plt.plot(xs, ys, '-o')
