import random
import numpy as np
import json
from typing import Iterable, List
from numpy._typing import NDArray


class IteratedLocalSearch:
    # maximum number of iterations (Stopping criterion of ILS)
    MAX_ITER = 100

    # number of iterations without improvement of the objective function (Stopping criterion of ILS)
    MAX_ITER_NI = 10
    # maximum number of iterations of the local search operator (Outer loop)
    MAX_ITER_LS = 10000
    # maximum number of swaps of the local search operator (Inner loop)
    MAX_SWAPS = 1
    NB_SWAPS = 1       # number of swaps in the perturbation operator
    ALPHA = 0.05

    round_with_no_improvement = 0

    @staticmethod
    def intensify(initial_solution, neighborhood_generator, fitness_function):
        """
        This method tries to improve a given initial_solution by exploring its neighborhood.
        The neighborhood_generator is a generator that return a generator over the neighbors of the initial_solution. (It calls neighborhood_generator(initial_solution))
        Neighbors are compared using the fitness function.
        """
        best_sol, best_fitness = initial_solution, fitness_function(
            initial_solution)
        for sol in neighborhood_generator(initial_solution):
            fitness = fitness_function(sol)
            if fitness < best_fitness:
                best_sol, best_fitness = sol, fitness
        return best_fitness, best_sol

    def local_search(self, solution, neighborhood_generator, fitness_function):
        """
        Local search iteratively intensify a solution using the given neighborhood_generator.
        It stops after MAX_ITER_LS number of iteration, or after MAX_ITER_NI iterations without any improvement over the fitness_function.
        """
        self.round_with_no_improvement = 0
        best_fitness, best_sol = fitness_function(solution), solution
        for i in range(IteratedLocalSearch.MAX_ITER_LS):
            fitness, sol = IteratedLocalSearch.intensify(
                best_sol, neighborhood_generator, fitness_function)
            if best_fitness <= fitness:
                self.round_with_no_improvement += 1
                if self.round_with_no_improvement >= IteratedLocalSearch.MAX_ITER_NI:
                    print(
                        f"No improvement after {self.round_with_no_improvement} iterations (total # iteratio: {i})")
                    break
            elif best_fitness >= fitness:
                best_fitness, best_sol = fitness, sol
        return best_fitness, best_sol

    def solve(self,
              solution,
              neighborhood_generator,
              perturbator,
              fitness_function):
        """
        This method iteratively use local search for finding the best solution, starting from solution.
        Once the local search is done, it tries to escape the local optimum using the perturbator function.
        """
        best_fitness, best_sol = fitness_function(solution), solution
        current_fitness, current_sol = best_fitness, best_sol
        for _ in range(IteratedLocalSearch.MAX_ITER):
            print(f"Intensifting {current_fitness}")
            fitness, sol = self.local_search(
                current_sol, neighborhood_generator, fitness_function)
            print(f"Intensified until {fitness} (vs. {best_fitness})")
            # Keep best found solution
            if best_fitness > fitness:
                best_fitness, best_sol = fitness, sol
            current_sol = perturbator(current_sol)
            current_fitness = fitness_function(current_sol)
            print(current_sol)
        return best_fitness, best_sol



    

class Network:

    # parameters shared by all instances 
    n: int               = None
    f: list[float]       = None # hub allocation cost (fixed cost)
    c: list[list[float]] = None # cost per flow unit (variable cost)
    C: list[float]       = None # hub capacity
    alpha: float         = None # discount factor
   
    w: list[list[float]]  = None  # flow from each to each node
    flow_in: list[float]  = None # total flow targeted at each node
    flow_out: list[float] = None # total flow outgoing from each node

    # network structure (per instance)
    mat: list[list[bool]]
    # if nodes i and j are connected then mat[i][j] = True
    # if node i is a hub then mat[i][i] = True
    ### mat[i][j] = False for i < j (as edges are not directed)

    @classmethod
    def load_from_json(cls, filename: str):
        with open(filename, "r") as file:
            data = json.load(file)

            cls.n = int(data["NodeNum"])
            cls.f = data["fixedCost"]
            cls.c = data["varCost"]
            cls.C = data["Cap"]
            cls.alpha = data["alpha"]

            cls.w = data["flow"]
            cls.flow_in = []
            cls.flow_out = []
            for i in range(cls.n):
                cls.flow_in.append(sum(cls.w[j][i] for j in range(cls.n)))
                cls.flow_out.append(sum(cls.w[i][j] for j in range(cls.n)))

    @classmethod
    def random(cls, h: int):
        network = Network([[False for _ in range(cls.n)] for _ in range(cls.n)])

        # random hubs
        nodes = [i for i in range(cls.n)]
        random.shuffle(nodes)
        for i in (hubs := nodes[:h]):
            network[i, i] = True

        # random tree (prufer tree)
        prufer = [random.randint(0, h-1) for _ in range(h-2)]
        degrees = [1 for _ in range(h)]
        for i in prufer:
            degrees[i] += 1

        for i in prufer:
            for j in range(h):
                if degrees[j] == 1:
                    network[hubs[i], hubs[j]] = True
                    degrees[i] += -1
                    degrees[j] += -1
                    break
        i = degrees.index(1)
        j = degrees.index(1, i+1)
        network[hubs[i], hubs[j]] = True

        # connect non hub to random hub
        for i in range(cls.n):
            if not network.is_hub(i):
                network[i, hubs[random.randint(0, h-1)]] = True

        return network

    def __init__(self, mat: list[list[bool]]):
        self.mat = mat

    def __repr__(self):
        s = "  "
        for i in range(self.n):
            s += f"{i} "
        s += "\n"
        for i in range(self.n):
            s += f"{i} "
            for j in range(i):
                if self.mat[i][j]:
                    if self.is_hub(i) and self.is_hub(j):
                        s += "\033[95mx\033[0m"
                    else:
                        s += "x"
                else:
                    s += "."
                s += " "
            s += "\033[93mx\033[0m" if self.is_hub(i) else "."
            s += " \n"
        return s

    def __getitem__(self, key):
        i, j = key
        if i > j:
            return self.mat[i][j]
        return self.mat[j][i]

    def __setitem__(self, key, value):
        i, j = key
        if i > j:
            self.mat[i][j] = value
        else:
            self.mat[j][i] = value

    def is_hub(self, i: int):
        return self.mat[i][i]

    def hub_number(self):
        return sum([self.is_hub(i) for i in range(self.n)])

    def copy(self):
        return Network([l.copy() for l in self.mat])

    def fitness(self):
        fixed_cost = sum([self.is_hub(i) * self.f[i] for i in range(self.n)])
        var_cost = 0.
        for i in range(self.n):
            for j in range(self.n):
                if self.mat[i][j]:
                    var_cost += self.alpha if self.is_hub(i) and self.is_hub(j) else 1. \
                                * (self.c[j][i] * self.flow_in[i] \
                                +  self.c[i][j] * self.flow_out[i])
        return fixed_cost + var_cost


# intensifiers

# relink spoke to adjacent hub of its hub
def relink_adj(n: Network) -> Iterable[Network]:
    for i in range(n.n):
        if not n.is_hub(i):
            hub = [n[i, j] for j in range(n.n)].index(True)
            for j in range(n.n):
                if j != hub and n.is_hub(j) and n[hub, j]:
                    new_n = n.copy()
                    new_n[i, hub] = False
                    new_n[i, j] = True
                    yield new_n

# rebase hub to adjacent spoke
def rebase_adj(n: Network) -> Iterable[Network]:
    for i in range(n.n):
        if n.is_hub(i):
            for j in range(n.n):
                if not n.ishub(j) and n[i, j]:
                    new_n = n.copy()
                    new_n[j, j] = False
                    new_n[i, i] = True
                    yield new_n


# perturbators


def perturbator(n: Network) -> Network:
    return Network.random(n.hub_number())
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="ILS TSP")
    parser.add_argument('--max_iter_ls', type=int,
                        default=IteratedLocalSearch.MAX_ITER_LS)
    parser.add_argument('--max_iter', type=int,
                        default=IteratedLocalSearch.MAX_ITER)
    parser.add_argument('--initial_solution')
    parser.add_argument('--seed', default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    IteratedLocalSearch.MAX_ITER_LS = args.max_iter_ls
    IteratedLocalSearch.MAX_ITER = args.max_iter

    # print(IteratedLocalSearch().solve(CityTour.random(n),
    #                                   intensifier,
    #                                   perturbator,
    #                                   lambda tour: tour.fitness(distances)))

    Network.load_from_json("InputDataHubSmallInstance.json")

    # network = Network.random(4)
    # print(network)
    # print(Network.random(network.hub_number()))
    # for n in perturbator(network):
        # print(n)

    print(IteratedLocalSearch().solve(Network.random(4),
                                   relink_adj,
                                   perturbator,
                                   lambda n: n.fitness()))

