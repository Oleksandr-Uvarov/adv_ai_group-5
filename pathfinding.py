import numpy as np
from collections import deque

# Tile types (shared by every module that reads the grid).
FLOOR = 0
WALL = 1

# The four orthogonal moves, and the named single-axis directions.
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_VECTORS = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}


def floor_cells(grid):
    """Every ``(row, col)`` floor cell in the grid."""
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == FLOOR]


def wall_neighbor_count(grid, cell):
    """How many of a cell's four orthogonal neighbours are walls."""
    r, c = cell
    return sum(1 for dr, dc in _DIRS if grid[r + dr, c + dc] == WALL)


def adjacent_floor_pos(grid, anchor, occupied=()):
    """First floor cell orthogonally adjacent to ``anchor`` that is not already
    taken, returned as a ``[row, col]`` list, or None if every side is a wall or
    occupied. ``occupied`` is a container of ``(row, col)`` tuples."""
    r, c = anchor
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if grid[nr, nc] == FLOOR and (nr, nc) not in occupied:
            return [nr, nc]
    return None


def is_reachable(grid, start, goal):
    """True if ``goal`` can be reached from ``start`` walking only on floor."""
    start = tuple(start)
    goal = tuple(goal)
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) == goal:
                return True
            if (nr, nc) not in visited and grid[nr, nc] != WALL:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def is_reachable_avoiding(grid, start, goal, blocked):
    """Reachability that treats every cell in ``blocked`` (a set of ``(row, col)``
    tuples) as impassable on top of the walls. Used to confirm a route to an
    objective exists that never steps on a spike."""
    start = tuple(start)
    goal = tuple(goal)
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) == goal:
                return True
            if ((nr, nc) not in visited and grid[nr, nc] != WALL
                    and (nr, nc) not in blocked):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def is_reachable_unidirectional(grid, start, goal, direction, forbidden=None):
    """Whether ``goal`` is reachable from ``start`` moving only along a single
    fixed ``direction`` ("left"/"right"/"up"/"down"), optionally treating the
    ``forbidden`` cell as impassable. Used to check the guard can be approached
    from one side without stepping onto the key (which would wake it)."""
    start = tuple(start)
    goal = tuple(goal)
    forbidden = tuple(forbidden) if forbidden is not None else None
    dr, dc = _DIR_VECTORS[direction]
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        nr, nc = r + dr, c + dc
        if (r, c) == goal:
            return True
        if ((nr, nc) not in visited and grid[nr, nc] != WALL
                and (forbidden is None or (nr, nc) != forbidden)):
            visited.add((nr, nc))
            queue.append((nr, nc))
    return False


def is_reachable_unidirectional_any(grid, start, goal, forbidden=None):
    """True if ``goal`` is reachable from ``start`` along any one of the four
    directions (see :func:`is_reachable_unidirectional`)."""
    return any(
        is_reachable_unidirectional(grid, start, goal, direction, forbidden)
        for direction in ("left", "right", "up", "down")
    )


def bfs_distance(grid, start, goal):
    """Shortest floor-path distance from ``start`` to ``goal`` as an int, or
    ``float('inf')`` if the goal is unreachable."""
    start = tuple(start)
    goal = tuple(goal)
    if start == goal:
        return 0
    visited = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and grid[nr, nc] != WALL:
                visited[(nr, nc)] = visited[(r, c)] + 1
                if (nr, nc) == goal:
                    return visited[(nr, nc)]
                queue.append((nr, nc))
    return float("inf")


def bfs_step_toward(grid, start, goal, obstacles):
    """First cell on a shortest path from ``start`` to ``goal`` avoiding walls and
    every cell in ``obstacles`` (other entities), returned as a ``(row, col)``
    tuple, or None if ``goal`` is unreachable under those constraints. ``goal``
    itself is always allowed even if it appears in ``obstacles``."""
    start = tuple(start)
    goal = tuple(goal)
    if start == goal:
        return start
    visited = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dr, dc in _DIRS:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if (neighbor not in visited and grid[nr, nc] != WALL
                    and (neighbor == goal or neighbor not in obstacles)):
                visited[neighbor] = current
                queue.append(neighbor)

    if goal not in visited:
        return None
    # Trace back from the goal to the first step out of start.
    current = goal
    while visited[current] != start:
        current = visited[current]
    return current


def distance_map(grid, goal):
    """A ``grid``-shaped float array of normalised distances to ``goal``.

    BFS fills reachable floor cells with their distance to the goal; walls and
    floor cells with no path to the goal keep the -1 sentinel. Reachable cells are
    normalised into [0, 1] (0 = on the goal, 1 = the farthest reachable tile) and
    the sentinel cells are then set to 1.0 as well - an unreachable tile is, in
    effect, infinitely far. This keeps the whole map inside the [0, 1] observation
    bounds and stops an unreachable tile from reading as ~0 (as if it sat on the
    goal), which is what a plain ``-1 / max_d`` normalisation would produce."""
    dist = np.full(grid.shape, -1.0, dtype=np.float32)
    goal = tuple(goal)
    queue = deque([goal])
    dist[goal[0], goal[1]] = 0.0

    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if dist[nr, nc] < 0 and grid[nr, nc] != WALL:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    unreachable = dist < 0
    max_d = dist.max()
    if max_d > 0:
        dist = dist / max_d
    dist[unreachable] = 1.0
    return dist
