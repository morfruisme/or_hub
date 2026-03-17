import random
import json
from typing import Iterable, List


class IteratedLocalSearch:
    MAX_ITER = 100
    MAX_ITER_NI = 10
    MAX_ITER_LS = 10000
    MAX_SWAPS = 1
    NB_SWAPS = 1
    ALPHA = 0.05

    round_with_no_improvement = 0

    @staticmethod
    def intensify(initial_solution, neighborhood_generator, fitness_function):
        """
        This method tries to improve a given initial_solution by exploring its neighborhood.
        The neighborhood_generator is a generator that return a generator over the neighbors of the initial_solution. (It calls neighborhood_generator(initial_solution))
        Neighbors are compared using the fitness function.
        """
        best_sol, best_fitness = initial_solution, fitness_function(initial_solution)
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
                        f"No improvement after {self.round_with_no_improvement} iterations (total # iteration : {i})")
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
            print(f"Intensifying from {current_fitness}")
            fitness, sol = self.local_search(
                current_sol, neighborhood_generator, fitness_function)
            print(f"Intensified to {fitness} (vs. {best_fitness})")
            # Keep best found solution
            if best_fitness > fitness:
                best_fitness, best_sol = fitness, sol
            current_sol = perturbator(current_sol)
            current_fitness = fitness_function(current_sol)
        return best_fitness, best_sol


class Network:
    # structure[i][j] :
    # - for all i: assignment Z_ij (node i assigned to hub j) -> one 1 per row on hub columns
    # - for hub rows only: off-diagonal Y_ij links in the hub tree (symmetric)
    structure: List[List[int]]
    length: int

    def __init__(self, structure: List[List[int]], length: int):
        self.structure = structure
        self.length = length

    def __repr__(self):
        return f"Network(n={self.length})"

    @staticmethod
    def _deepcopy_structure(structure: List[List[int]]) -> List[List[int]]:
        return [row[:] for row in structure]

    @staticmethod
    def _hubs_from_structure(structure: List[List[int]]) -> List[int]:
        n = len(structure)
        return [k for k in range(n) if structure[k][k] == 1]

    @staticmethod
    def _clear_hub_links(structure: List[List[int]], hubs: List[int]) -> None:
        for a in hubs:
            for b in hubs:
                if a != b:
                    structure[a][b] = 0

    @staticmethod
    def _build_star_tree(structure: List[List[int]], hubs: List[int]) -> None:
        if len(hubs) <= 1:
            return
        root = hubs[0]
        for h in hubs[1:]:
            structure[root][h] = 1
            structure[h][root] = 1

    @staticmethod
    def _assign_all_to_nearest_hub(structure: List[List[int]], hubs: List[int], C: List[List[float]]) -> None:
        n = len(structure)
        if not hubs:
            return

        for i in range(n):
            # clear assignment entries on hub columns
            for h in hubs:
                structure[i][h] = 0

            if i in hubs:
                structure[i][i] = 1
            else:
                h_best = min(hubs, key=lambda h: C[i][h])
                structure[i][h_best] = 1

    @staticmethod
    def initial_structure(n: int, hub_indices: List[int], C: List[List[float]]) -> "Network":
        adj = [[0 for _ in range(n)] for _ in range(n)]

        # open hubs
        for h in hub_indices:
            adj[h][h] = 1

        # assign all nodes to closest hub
        Network._assign_all_to_nearest_hub(adj, hub_indices, C)

        # add hub tree links
        Network._clear_hub_links(adj, hub_indices)
        Network._build_star_tree(adj, hub_indices)

        return Network(adj, n)

    @staticmethod
    def greedy(data: dict, nb_hubs: int = 3) -> "Network":
        n = int(data["NodeNum"])
        C = data["varCost(cij)"]
        f_costs = data["fixCost(fk)"]
        caps = data["Cap(ckmax)"]

        nb_hubs = max(1, min(nb_hubs, n))
        # score simple: fixed cost / capacity (lower is better)
        scored = sorted(range(n), key=lambda k: f_costs[k] / max(caps[k], 1.0))
        hubs = scored[:nb_hubs]

        return Network.initial_structure(n, hubs, C)

    def _assigned_hub(self, i: int, hubs: List[int]) -> int:
        assigned = [h for h in hubs if self.structure[i][h] == 1]
        if len(assigned) != 1:
            return -1
        return assigned[0]

    def fitness(self, data: dict) -> float:
        n = self.length
        N = range(n)

        W = data["flow(wij)"]
        C = data["varCost(cij)"]
        alpha = data["alpha"]
        f_costs = data["fixCost(fk)"]
        Cap = data["Cap(ckmax)"]

        hubs = [k for k in N if self.structure[k][k] == 1]
        num_hubs = len(hubs)
        if num_hubs == 0:
            return float("inf")

        # Feasibility penalties
        penalty = 0.0

        # each node must be assigned to exactly one opened hub
        assign = {}
        for i in N:
            k = self._assigned_hub(i, hubs)
            if k == -1:
                penalty += 1_000_000.0
            else:
                assign[i] = k

        # hub must be assigned to itself
        for h in hubs:
            if assign.get(h, None) != h:
                penalty += 1_000_000.0

        # fixed costs
        total_cost = sum(f_costs[k] for k in hubs)

        # transportation costs
        for i in N:
            if i not in assign:
                continue
            k = assign[i]
            for j in N:
                if j not in assign:
                    continue
                l = assign[j]
                route_cost = C[i][k] + alpha * C[k][l] + C[l][j]
                total_cost += W[i][j] * route_cost

        # capacity penalties (simplified)
        hub_loads = {h: 0.0 for h in hubs}
        O = [sum(W[i][j] for j in N) for i in N]

        for i in N:
            if i not in assign:
                continue
            k = assign[i]
            hub_loads[k] += O[i]

            for j in N:
                if j not in assign:
                    continue
                l = assign[j]
                if k != l:
                    hub_loads[l] += W[i][j]

        for h in hubs:
            if hub_loads[h] > Cap[h]:
                penalty += (hub_loads[h] - Cap[h]) * 1000.0

        # tree structure penalty
        active_links = 0
        asym_penalty = 0.0
        for a in hubs:
            for b in hubs:
                if a < b and self.structure[a][b] == 1:
                    active_links += 1
                if self.structure[a][b] != self.structure[b][a]:
                    asym_penalty += 100_000.0

        if active_links != (num_hubs - 1):
            penalty += abs(active_links - (num_hubs - 1)) * 1_000_000.0

        return total_cost + penalty + asym_penalty


def intensifier(network: Network, data: dict) -> Iterable[Network]:
    n = network.length
    hubs = Network._hubs_from_structure(network.structure)

    for i in range(n):
        if i in hubs:
            continue

        current_h = network._assigned_hub(i, hubs)
        if current_h == -1:
            continue

        for h in hubs:
            if h == current_h:
                continue
            new_struct = Network._deepcopy_structure(network.structure)
            new_struct[i][current_h] = 0
            new_struct[i][h] = 1
            yield Network(new_struct, n)


def perturbator(network: Network, data: dict) -> Network:
    n = network.length
    C = data["varCost(cij)"]

    new_struct = Network._deepcopy_structure(network.structure)
    hubs = Network._hubs_from_structure(new_struct)
    non_hubs = [i for i in range(n) if i not in hubs]

    if random.random() < 0.6 and non_hubs:
        # open or swap hub
        add_h = random.choice(non_hubs)

        if len(hubs) <= 1:
            new_struct[add_h][add_h] = 1
            hubs = Network._hubs_from_structure(new_struct)
        else:
            remove_h = random.choice(hubs)
            if remove_h == add_h:
                pass
            else:
                new_struct[remove_h][remove_h] = 0
                new_struct[add_h][add_h] = 1
                hubs = Network._hubs_from_structure(new_struct)

        Network._assign_all_to_nearest_hub(new_struct, hubs, C)
        Network._clear_hub_links(new_struct, hubs)
        Network._build_star_tree(new_struct, hubs)

    else:
        # random reassignment on existing hubs
        if hubs:
            spokes = [i for i in range(n) if i not in hubs]
            for _ in range(min(IteratedLocalSearch.NB_SWAPS, len(spokes))):
                i = random.choice(spokes)
                h_cur = network._assigned_hub(i, hubs)
                h_new = random.choice(hubs)
                if h_cur != -1:
                    new_struct[i][h_cur] = 0
                new_struct[i][h_new] = 1

    return Network(new_struct, n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="ILS Hubs")
    parser.add_argument("--max_iter_ls", type=int, default=IteratedLocalSearch.MAX_ITER_LS)
    parser.add_argument("--max_iter", type=int, default=IteratedLocalSearch.MAX_ITER)
    parser.add_argument("--nb_hubs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    with open("InputDataHubLargeInstance.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    IteratedLocalSearch.MAX_ITER_LS = args.max_iter_ls
    IteratedLocalSearch.MAX_ITER = args.max_iter

    initial = Network.greedy(data, nb_hubs=args.nb_hubs)

    best_fitness, best_sol = IteratedLocalSearch().solve(
        initial,
        lambda net: intensifier(net, data),
        lambda net: perturbator(net, data),
        lambda net: net.fitness(data),
    )

    print(f"Best fitness: {best_fitness:.2f}")
    print(best_sol.structure)