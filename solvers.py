# solvers.py
import heapq
from time import perf_counter
from typing import Dict, List, Tuple, Any

State = Tuple[int, ...]
GOAL: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def get_neighbours_with_direction(state: State) -> List[Tuple[State, str]]:
    neighbours = []
    blank = state.index(0)
    r, c = blank // 3, blank % 3
    moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]

    for dr, dc, move_name in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            res = list(state)
            res[blank], res[nr * 3 + nc] = res[nr * 3 + nc], res[blank]
            neighbours.append((tuple(res), move_name))
    return neighbours

def manhattan(state: State, goal: State = GOAL) -> int:
    # small state size => goal.index(tile) is fine
    dist = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        gi = goal.index(tile)
        dist += abs(i // 3 - gi // 3) + abs(i % 3 - gi % 3)
    return dist

def apply_move(state: State, direction: str) -> State:
    idx0 = state.index(0)
    r, c = divmod(idx0, 3)
    drdc = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
    dr, dc = drdc[direction]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < 3 and 0 <= nc < 3):
        return state

    lst = list(state)
    nidx = nr * 3 + nc
    lst[idx0], lst[nidx] = lst[nidx], lst[idx0]
    return tuple(lst)

def path_to_states(start: State, path: List[str]) -> List[State]:
    states = [start]
    s = start
    for mv in path:
        s = apply_move(s, mv)
        states.append(s)
    return states

def a_star_baseline_metrics(start: State, goal: State = GOAL) -> Dict[str, Any]:
    """
    A* (Manhattan). Returns:
      steps, time_ms, nodes_expanded, max_frontier, visited_count,
      states (path states),
      frontier_sizes + expanded_counts (for frontier growth plot)
    """
    t0 = perf_counter()

    frontier = [(manhattan(start, goal), 0, start, [])]  # (f, g, state, path)
    visited = {start: 0}

    expanded = 0
    max_frontier = 1
    frontier_sizes: List[int] = []
    expanded_counts: List[int] = []

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        frontier_sizes.append(len(frontier))
        expanded_counts.append(expanded)

        f, g, curr, path = heapq.heappop(frontier)
        expanded += 1

        if curr == goal:
            t1 = perf_counter()
            return {
                "steps": len(path),
                "time_ms": (t1 - t0) * 1000.0,
                "nodes_expanded": expanded,
                "max_frontier": max_frontier,
                "visited_count": len(visited),
                "path": path,
                "states": path_to_states(start, path),
                "frontier_sizes": frontier_sizes,
                "expanded_counts": expanded_counts,
            }

        for n, direction in get_neighbours_with_direction(curr):
            ng = g + 1
            if n not in visited or ng < visited[n]:
                visited[n] = ng
                heapq.heappush(frontier, (ng + manhattan(n, goal), ng, n, path + [direction]))

    t1 = perf_counter()
    return {
        "steps": 0,
        "time_ms": (t1 - t0) * 1000.0,
        "nodes_expanded": expanded,
        "max_frontier": max_frontier,
        "visited_count": len(visited),
        "path": [],
        "states": [start],
        "frontier_sizes": frontier_sizes,
        "expanded_counts": expanded_counts,
    }

class NFSACOSolver:
    """
    NFS-ACO: A* structure with pheromone weighting + thermal depth density.
    Returns same metrics as A* plus thermal_limit and evaporation.
    """
    def __init__(self, goal: State, initial_manhattan_distance: int, thermal_limit: float = 50, evaporation: float = 0.95):
        self.goal = goal

        if initial_manhattan_distance > 10:
            self.thermal_limit = min(50 + (initial_manhattan_distance - 10) * (100 / 15), 150)
            self.evaporation = max(0.95 - (initial_manhattan_distance - 10) * (0.10 / 15), 0.85)
        else:
            self.thermal_limit = thermal_limit
            self.evaporation = evaporation

        self.Pheromone: Dict[State, float] = {}

    def solve_metrics(self, start: State) -> Dict[str, Any]:
        t0 = perf_counter()

        frontier = [(manhattan(start, self.goal), 0, start, [])]  # (hybrid_f, g, state, path)
        visited = {start: 0}
        depth_density: Dict[int, int] = {}

        expanded = 0
        max_frontier = 1
        frontier_sizes: List[int] = []
        expanded_counts: List[int] = []

        while frontier:
            max_frontier = max(max_frontier, len(frontier))
            frontier_sizes.append(len(frontier))
            expanded_counts.append(expanded)

            f, g, curr, path = heapq.heappop(frontier)

            # thermal depth density gating
            depth_density[g] = depth_density.get(g, 0) + 1
            if depth_density[g] > self.thermal_limit:
                continue

            expanded += 1
            if curr == self.goal:
                t1 = perf_counter()
                return {
                    "steps": len(path),
                    "time_ms": (t1 - t0) * 1000.0,
                    "nodes_expanded": expanded,
                    "max_frontier": max_frontier,
                    "visited_count": len(visited),
                    "path": path,
                    "states": path_to_states(start, path),
                    "thermal_limit": self.thermal_limit,
                    "evaporation": self.evaporation,
                    "frontier_sizes": frontier_sizes,
                    "expanded_counts": expanded_counts,
                }

            for n, direction in get_neighbours_with_direction(curr):
                h = manhattan(n, self.goal)

                scent = 1.0 / (h + 1)  # heuristic-aligned pheromone deposit
                self.Pheromone[n] = self.Pheromone.get(n, 1.0) * self.evaporation + scent

                smell_multiplier = 1.0 / (self.Pheromone[n])
                hybrid_f = (g + 1 + h) * smell_multiplier

                ng = g + 1
                if n not in visited or ng < visited[n]:
                    visited[n] = ng
                    heapq.heappush(frontier, (hybrid_f, ng, n, path + [direction]))

        t1 = perf_counter()
        return {
            "steps": 0,
            "time_ms": (t1 - t0) * 1000.0,
            "nodes_expanded": expanded,
            "max_frontier": max_frontier,
            "visited_count": len(visited),
            "path": [],
            "states": [start],
            "thermal_limit": self.thermal_limit,
            "evaporation": self.evaporation,
            "frontier_sizes": frontier_sizes,
            "expanded_counts": expanded_counts,
        }
