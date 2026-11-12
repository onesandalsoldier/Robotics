"""
ROB-UY 3303 - Lab 04: Tree Search Motion Planning (A*)
STARTER CODE - complete every function marked STUDENT TODO.
Run the unit tests with:   python lab_4.py
The GUI (gui_4.py) imports this file and uses your planner and trackers on
the robot. map_4.py (same folder) holds the maps.

Write your code only where it says STUDENT TODO. Sections marked
PROVIDED are given to you: do not change them.

Units: meters, seconds, radians.
Grid: 200 x 200 cells of 1 cm (see map_4.py). A cell is a list [i, j] of
two ints: i = column (along x), j = row (along y). OCCUPANCY_GRID[i][j] is
1 if the CENTRE of the robot cannot be in cell [i, j].
"""
import math
import time

import numpy as np

import map_4

# =====================================================================
# PART 0 - STUDENT TODO: COPY YOUR LAB 3 CODE (lab_3.py)
# =====================================================================
# Copy your robot constants and these three functions from lab_3.py
# (same names, inputs and outputs). Copy any helper function they use too.
WHEEL_RADIUS_M =     0.0335       # m
WHEELBASE_M =     0.175         # distance between the wheels = 2L (m)

def get_wheel_velocities(v, w):
    L = WHEELBASE_M / 2.0
    phi_dot_l = (v - w * L) / WHEEL_RADIUS_M
    phi_dot_r = (v + w * L) / WHEEL_RADIUS_M
    return phi_dot_l, phi_dot_r

def point_tracking_controller(X_des, X):
    k_rho, k_alpha, k_beta = 1.0, 3.0, -1.0
    dx = X_des[0] - X[0]
    dy = X_des[1] - X[1]
    rho = math.hypot(dx, dy)
    
    if rho < 0.001:
        return get_wheel_velocities(0, k_beta * wrap_angle(X_des[2] - X[2]))

    alpha = wrap_angle(math.atan2(dy, dx) - X[2])
    beta = wrap_angle(X_des[2] - X[2] - alpha)
    
    if alpha > math.pi / 2:
        alpha = wrap_angle(alpha - math.pi)
        beta = wrap_angle(beta - math.pi)
        v = -k_rho * rho
    elif alpha < -math.pi / 2:
        alpha = wrap_angle(alpha + math.pi)
        beta = wrap_angle(beta + math.pi)
        v = -k_rho * rho
    else:
        v = k_rho * rho

    # drive forward only as much as we are facing the aim point:
    # at 90 degrees off, v = 0, so the robot turns in place first
    v = v * max(0.0, math.cos(alpha))
    V_MIN = 0.08
    if 0 < v < V_MIN and rho > 0.03:
        v = V_MIN
    w = k_alpha * alpha + k_beta * beta
    return get_wheel_velocities(v, w)

def get_closest_point_on_segment(A, B, P):
    abx, aby = B[0] - A[0], B[1] - A[1]
    apx, apy = P[0] - A[0], P[1] - A[1]
    len_sq = abx**2 + aby**2
    if len_sq == 0: return A
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / len_sq))
    return [A[0] + t * abx, A[1] + t * aby, B[2]]

def path_tracking_controller(P, X):
    lookahead = 0.2
    min_dist = float('inf')
    closest_idx = 0
    closest_pt = None
    
    for i in range(len(P) - 1):
        pt = get_closest_point_on_segment(P[i], P[i+1], X)
        dist = math.hypot(pt[0] - X[0], pt[1] - X[1])
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            closest_pt = pt
            
    if closest_pt is None:
        closest_pt = P[-1]
        
    rem_lookahead = lookahead
    p_des = None
    next_node = P[closest_idx + 1]
    d_next = math.hypot(next_node[0] - closest_pt[0], next_node[1] - closest_pt[1])
    
    if d_next >= rem_lookahead:
        r = rem_lookahead / d_next
        p_des = [closest_pt[0] + r * (next_node[0] - closest_pt[0]),
                 closest_pt[1] + r * (next_node[1] - closest_pt[1]), next_node[2]]
    else:
        rem_lookahead -= d_next
        curr_idx = closest_idx + 1
        while curr_idx < len(P) - 1:
            seg_len = math.hypot(P[curr_idx+1][0] - P[curr_idx][0], P[curr_idx+1][1] - P[curr_idx][1])
            if seg_len >= rem_lookahead:
                r = rem_lookahead / seg_len
                p_des = [P[curr_idx][0] + r * (P[curr_idx+1][0] - P[curr_idx][0]),
                         P[curr_idx][1] + r * (P[curr_idx+1][1] - P[curr_idx][1]), P[curr_idx+1][2]]
                break
            rem_lookahead -= seg_len
            curr_idx += 1
        if p_des is None:
            p_des = P[-1]
            
    return point_tracking_controller(p_des, X)


# =====================================================================
# PROVIDED - ROBOT SIMULATOR (same as Lab 3, do not change)
# =====================================================================
DT = 0.1                    # s, time step (same as the GUI control loop)
POS_TOL = 0.03              # m, goal reached if closer than this
ANG_TOL = 0.1               # rad, and heading error smaller than this


def wrap_angle(a):
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def get_robot_velocities(phi_dot_l, phi_dot_r):
    """Wheel speeds (rad/s) -> robot (v, w)."""
    L = WHEELBASE_M / 2.0
    v = WHEEL_RADIUS_M * (phi_dot_r + phi_dot_l) / 2.0
    w = WHEEL_RADIUS_M * (phi_dot_r - phi_dot_l) / (2.0 * L)
    return v, w


def robot_simulator(X_tm1, phi_dot_l, phi_dot_r, delta_T, n_substeps=10):
    """Moves the robot for delta_T seconds with constant wheel speeds.
    Returns X_t (new pose [x, y, theta]) and the list of poses visited."""
    x, y, th = X_tm1
    h = delta_T / n_substeps
    poses = []
    for _ in range(n_substeps):
        v, w = get_robot_velocities(phi_dot_l, phi_dot_r)
        x += v * h * math.cos(th + w * h / 2.0)
        y += v * h * math.sin(th + w * h / 2.0)
        th = wrap_angle(th + w * h)
        poses.append([x, y, th])
    return [x, y, th], poses


def is_at_goal(X_des, X):
    """True if X is within POS_TOL and ANG_TOL of X_des."""
    return (math.hypot(X_des[0] - X[0], X_des[1] - X[1]) < POS_TOL and
            abs(wrap_angle(X_des[2] - X[2])) < ANG_TOL)


# =====================================================================
# PROVIDED - THE GRID MAP (do not change)
# =====================================================================
GRID_N = map_4.GRID_N                   # 200 cells along x and along y
CELL_SIZE_M = map_4.CELL_SIZE_M         # 0.01 m
ROBOT_SPEED_M_S = 0.12                  # m/s, robot speed on normal floor (k = 1)

OCCUPANCY_GRID = None       # 200 x 200, 1 = robot centre cannot be in the cell
TRAVERSABILITY_GRID = None  # 200 x 200, k in (0, 1]: speed allowed = k * ROBOT_SPEED_M_S
COLLISION_GRID = None       # 200 x 200, 1 = the robot body touches an obstacle or a wall


def load_map(name):
    """Loads a map of map_4.MAPS: the obstacles grown by the robot radius
    and a safety margin (configuration space, Lecture 08A), and the
    traversability."""
    global COLLISION_GRID
    load_grid(map_4.build_c_space(name), map_4.build_traversability_grid(name))
    COLLISION_GRID = map_4.grow_obstacles(map_4.build_obstacle_grid(name), map_4.ROBOT_RADIUS_M)


def load_grid(occupancy, traversability=None):
    """Loads any 200 x 200 occupancy grid (1 = occupied) directly, e.g. for
    the unit tests. traversability = None means k = 1 everywhere."""
    global OCCUPANCY_GRID, TRAVERSABILITY_GRID, COLLISION_GRID
    OCCUPANCY_GRID = np.asarray(occupancy, dtype=np.uint8)
    COLLISION_GRID = OCCUPANCY_GRID
    if traversability is None:
        traversability = np.ones((GRID_N, GRID_N))
    TRAVERSABILITY_GRID = np.asarray(traversability, dtype=float)


def empty_grid():
    """200 x 200 grid of zeros, to build test maps: grid[i][j] = 1 ..."""
    return np.zeros((GRID_N, GRID_N), dtype=np.uint8)


def in_bounds(X):
    """True if cell X = [i, j] is on the grid."""
    return 0 <= X[0] < GRID_N and 0 <= X[1] < GRID_N


def is_occupied(X):
    """True if cell X = [i, j] is in collision."""
    return OCCUPANCY_GRID[X[0]][X[1]] == 1


def traversability(X):
    """Traversability k of cell X = [i, j]."""
    return float(TRAVERSABILITY_GRID[X[0]][X[1]])


def new_cost_array():
    """200 x 200 array full of infinity (one value per cell)."""
    return np.full((GRID_N, GRID_N), math.inf)


def meters_to_cell(x, y):
    """Position (m) -> cell [i, j] that contains it."""
    return [int(math.floor((x - map_4.GRID_ORIGIN_M[0]) / CELL_SIZE_M)),
            int(math.floor((y - map_4.GRID_ORIGIN_M[1]) / CELL_SIZE_M))]


def cell_to_meters(X):
    """Cell [i, j] -> position (m) [x, y] of its centre."""
    return [map_4.GRID_ORIGIN_M[0] + (X[0] + 0.5) * CELL_SIZE_M,
            map_4.GRID_ORIGIN_M[1] + (X[1] + 0.5) * CELL_SIZE_M]


def robot_in_collision(X):
    """True if the robot at pose X = [x, y, theta] (m) touches an obstacle
    or a wall of the map loaded with load_map (no safety margin)."""
    i, j = meters_to_cell(X[0], X[1])
    return not in_bounds([i, j]) or COLLISION_GRID[i][j] == 1


def check_path(P, X_0, X_d):
    """Checks a planned path of cells. Returns (True, "ok") or (False, reason)."""
    if not P:
        return False, "no path returned"
    if list(P[0]) != list(X_0) or list(P[-1]) != list(X_d):
        return False, "path must start at X_0 and end at X_d"
    for k, X in enumerate(P):
        if not in_bounds(X):
            return False, f"cell {k} {X} is off the grid"
        if is_occupied(X) and k > 0:
            return False, f"cell {k} {X} is in collision"
        if k > 0 and abs(X[0] - P[k - 1][0]) + abs(X[1] - P[k - 1][1]) != 1:
            return False, f"cells {k - 1} and {k} are not neighbors"
    return True, "ok"


def path_time_s(P):
    """Time (s) to drive along the path of cells P at the speed allowed in
    each cell entered (k * ROBOT_SPEED_M_S)."""
    return sum(CELL_SIZE_M / (ROBOT_SPEED_M_S * traversability(X)) for X in P[1:])


# =====================================================================
# PART 2 - THE A* PLANNER (Lecture 09A and 09B)
# =====================================================================
# DO NOT USE ANY GEN AI TOOLS OR EXAMPLE A* CODE FOR THIS PART (Step 6 only).
#
# Two cost functions are used in this lab (cost_mode):
#   "distance": shortest path. The cost of a step to a neighbor cell is the
#               distance travelled (m). Part 2.
#   "time":     minimum time. The cost of a step is the time (s) needed to
#               drive into the neighbor cell at the speed allowed there,
#               k * ROBOT_SPEED_M_S. Part 3.
# For now, write everything for "distance". You add "time" in Part 3.

# STUDENT TODO - Step 1: the Node class.
# Member variables:
#   X        - [i, j], the cell of the node
#   parent   - the parent Node (None for the root)
#   g_value  - g(n): path cost from the root to this node
#   h_value  - h(n): estimated path cost from this node to the goal
#   f_value  - f(n) = g(n) + h(n)
class Node:
    def __init__(self, X, parent, g_value, h_value):
        self.X = X
        self.parent = parent
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = g_value + h_value

    def __lt__(self, other):
        return self.f_value < other.f_value


# STUDENT TODO - Step 1: the cost of one step and the heuristic.
# get_step_cost(X_from, X_to, cost_mode) -> cost of moving from cell X_from
#   to the neighbor cell X_to (m for "distance", s for "time").
# get_heuristic(X, X_d, cost_mode) -> h(n) for a node at cell X and the goal
#   cell X_d. It must be ADMISSIBLE (never overestimates the cost to the goal,
#   Lecture 09B): use the straight line distance from X to X_d (m).
def get_step_cost(X_from, X_to, cost_mode="distance"):
    dist = math.hypot(X_to[0] - X_from[0], X_to[1] - X_from[1]) * CELL_SIZE_M
    if cost_mode == "distance":
        return dist
    elif cost_mode == "time":
        k = traversability(X_to)
        return dist / (k * ROBOT_SPEED_M_S)
    return dist


def get_heuristic(X, X_d, cost_mode="distance"):
    dist = math.hypot(X_d[0] - X[0], X_d[1] - X[1]) * CELL_SIZE_M
    if cost_mode == "distance":
        return dist
    elif cost_mode == "time":
        return dist / ROBOT_SPEED_M_S
    return dist


# STUDENT TODO - Step 2: get_neighbors(n, X_d, best_g, cost_mode)
# Returns a list of up to 4 NEW nodes, children of node n: the cells one
# step in +x, -x, +y and -y from n.X. Set the parent, g, h and f of each.
# Do not create a neighbor if it is:
#   - out of bounds:  off the grid (use in_bounds)
#   - in collision:   occupied (use is_occupied)
#   - already added:  best_g is a 200 x 200 array (new_cost_array()) with
#                     the lowest g found so far for each cell. Only create
#                     the neighbor if its g is LOWER than best_g at its
#                     cell, and then store its g in best_g. This stops the
#                     search from going around cycles, and keeps A* optimal
#                     when the cost of a step is not always the same (Part 3).
def get_neighbors(n, X_d, best_g, cost_mode="distance"):
    neighbors = []
    for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        next_x = n.X[0] + dx
        next_y = n.X[1] + dy
        next_pos = [next_x, next_y]
        
        if not in_bounds(next_pos) or is_occupied(next_pos):
            continue
            
        step_cost = get_step_cost(n.X, next_pos, cost_mode)
        next_g = n.g_value + step_cost
        
        if next_g < best_g[next_x][next_y]:
            best_g[next_x][next_y] = next_g
            h_val = get_heuristic(next_pos, X_d, cost_mode)
            neighbors.append(Node(next_pos, n, next_g, h_val))
            
    return neighbors


# STUDENT TODO - Step 3: is_goal(n, X_d) -> True if node n is at cell X_d.
def is_goal(n, X_d):
    return n.X == X_d


# STUDENT TODO - Step 4: A_star_planner(X_0, X_d, cost_mode)
#   X_0 = start cell [i0, j0], X_d = goal cell [id, jd].
#   Returns the path P = [X_0, X_1, ..., X_d], a list of cells [i, j], or
#   None if there is no path (the fringe becomes empty).
# Follow GENERAL-SEARCH (Lecture 09A) with the A* strategy (Lecture 09B):
# create the root node, keep a list of nodes on the fringe, pick the node
# with the lowest f(n) from the fringe, check if it is the goal, otherwise
# add its children to the fringe. Build P by following the parents from
# the goal node back to the root. Add any helper functions you want.
def A_star_planner(X_0, X_d, cost_mode="distance"):
    import heapq
    best_g = new_cost_array()
    start_h = get_heuristic(X_0, X_d, cost_mode)
    root = Node(X_0, None, 0.0, start_h)
    best_g[X_0[0]][X_0[1]] = 0.0
    
    fringe = []
    heapq.heappush(fringe, root)
    
    while fringe:
        current = heapq.heappop(fringe)
        
        if is_goal(current, X_d):
            path = []
            while current:
                path.append(current.X)
                current = current.parent
            return path[::-1]
            
        for neighbor in get_neighbors(current, X_d, best_g, cost_mode):
            heapq.heappush(fringe, neighbor)
            
    return None


# PROVIDED - example unit test (empty room, straight line).
def a_star_unit_test_1():
    load_grid(empty_grid())
    X_0, X_d = [25, 25], [125, 25]
    P = A_star_planner(X_0, X_d)
    ok, msg = check_path(P, X_0, X_d)
    optimal_cells = 101             # 100 steps of 1 cell + the start cell
    return ok and len(P) == optimal_cells, P


# STUDENT TODO - Step 5: 2 more unit tests, each returns (passed, P).
# Build a map with empty_grid(), set some cells to 1, call load_grid(...),
# then plan and compare with the optimal path that you computed by hand
# (use check_path and len(P)). Suggested cases:
#   2: a wall of obstacles between X_0 and X_d, with a gap the path must go through
#   3: X_d fully enclosed by obstacles -> A_star_planner must return None
def a_star_unit_test_2():
    grid = empty_grid()
    X_0, X_d = [25, 25], [125, 25]
    
    # Wall from y=0 to y=50 at x=75, leaving a gap from y=51+
    for y in range(0, 50):
        grid[75][y] = 1
        
    load_grid(grid)
    P = A_star_planner(X_0, X_d, cost_mode="distance")
    
    if P is None:
        return False, None
    ok, msg = check_path(P, X_0, X_d)
    # 50 cells +x (25 to 75), 25 cells +y (25 to 50), 50 cells +x (75 to 125), 25 cells -y (50 to 25)
    # Total optimal steps = 150. Total cells = 151.
    return ok and len(P) == 151, P


def a_star_unit_test_3():
    grid = empty_grid()
    X_0, X_d = [25, 25], [125, 25]
    
    # Fully enclose the goal
    for k in range(105, 146):
        grid[k][5] = 1
        grid[k][45] = 1
    for k in range(5, 46):
        grid[105][k] = 1
        grid[145][k] = 1
    
    load_grid(grid)
    P = A_star_planner(X_0, X_d, cost_mode="distance")
    
    # Should return None because it's completely enclosed
    return P is None, P


# STUDENT TODO: run the 3 tests, print how many pass out of 3, plot the
# map and the 3 paths (e.g. plt.imshow(grid.T, origin="lower")).
def run_a_star_unit_tests(plot=True):
    tests = [a_star_unit_test_1, a_star_unit_test_2, a_star_unit_test_3]
    passed_count = 0
    results = []
    
    for i, test in enumerate(tests, 1):
        passed, P = test()
        results.append(P)
        if passed:
            passed_count += 1
            print(f"A* Unit Test {i}: PASSED")
        else:
            print(f"A* Unit Test {i}: FAILED")
            
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (ax, P) in enumerate(zip(axes, results), 1):
            if i == 1:
                load_grid(empty_grid())
            elif i == 2:
                grid2 = empty_grid()
                for y in range(0, 50): grid2[75][y] = 1
                load_grid(grid2)
            elif i == 3:
                grid3 = empty_grid()
                for k in range(105, 146):
                    grid3[k][5] = 1
                    grid3[k][45] = 1
                for k in range(5, 46):
                    grid3[105][k] = 1
                    grid3[145][k] = 1
                load_grid(grid3)
                
            ax.imshow(OCCUPANCY_GRID.T, origin="lower", cmap="binary")
            if P:
                xs = [pt[0] for pt in P]
                ys = [pt[1] for pt in P]
                ax.plot(xs, ys, 'r-', linewidth=2)
                ax.plot(xs[0], ys[0], 'go', markersize=6)
                ax.plot(xs[-1], ys[-1], 'bx', markersize=6)
            if i == 3:
                ax.plot(25, 25, 'go', markersize=6)
                ax.plot(125, 25, 'bx', markersize=6)
                ax.plot([105, 145, 145, 105, 105], [5, 5, 45, 45, 5], 'r-', linewidth=2)
            ax.set_title(f"A* Unit Test {i}")
            ax.set_xlabel("i (cells)")
            ax.set_ylabel("j (cells)")
            
        plt.tight_layout()
        plt.show()
        
    print(f"A* Unit Tests: {passed_count}/3 passed")
    return passed_count


# =====================================================================
# PART 3 - MINIMUM TIME PLANNING WITH A TRAVERSABILITY MAP
# =====================================================================
# In map_4.py, "slow_zones" give each cell a traversability k in (0, 1]:
# the robot may drive at k * ROBOT_SPEED_M_S in that cell.
#
# STUDENT TODO - Step 1: add cost_mode "time" to get_step_cost and get_heuristic.
#   step cost = time (s) to travel one cell into X_to at the speed allowed in X_to
#   heuristic = time (s) to drive the straight line to the goal at the
#               highest possible speed. Explain in your report why it is admissible.
#
# STUDENT TODO - Step 2: 2 unit tests, each returns (passed, P_distance, P_time).
# Use path_time_s(P) to compare the time of two paths.
#   1: a slow zone (k = 0.2) across the straight line from X_0 to X_d, that the
#      robot can go around. "time" must go around it, "distance" goes through.
#   2: the same with k = 0.9: going around is now slower, "time" goes through.
def time_cost_unit_test_1():
    grid = empty_grid()
    load_grid(grid)
    X_0, X_d = [25, 25], [125, 25]
    
    for x in range(60, 91):
        for y in range(0, 41):
            TRAVERSABILITY_GRID[x][y] = 0.2

    P_distance = A_star_planner(X_0, X_d, cost_mode="distance")
    P_time = A_star_planner(X_0, X_d, cost_mode="time")
    
    passed = path_time_s(P_time) < path_time_s(P_distance) and len(P_distance) < len(P_time)
    return passed, P_distance, P_time


def time_cost_unit_test_2():
    grid = empty_grid()
    load_grid(grid)
    X_0, X_d = [25, 25], [125, 25]
    
    for x in range(60, 91):
        for y in range(0, 41):
            TRAVERSABILITY_GRID[x][y] = 0.9

    P_distance = A_star_planner(X_0, X_d, cost_mode="distance")
    P_time = A_star_planner(X_0, X_d, cost_mode="time")
    
    passed = len(P_time) == len(P_distance) and path_time_s(P_time) <= path_time_s(P_distance)
    return passed, P_distance, P_time


# STUDENT TODO: run the 2 tests, print how many pass out of 2, plot the paths.
def run_time_cost_unit_tests(plot=True):
    tests = [time_cost_unit_test_1, time_cost_unit_test_2]
    passed_count = 0
    results = []
    
    for i, test in enumerate(tests, 1):
        passed, P_dist, P_time = test()
        results.append((P_dist, P_time))
        if passed:
            passed_count += 1
            print(f"Time Cost Unit Test {i}: PASSED")
            print(f"  Distance mode time: {path_time_s(P_dist):.2f}s")
            print(f"  Time mode time:     {path_time_s(P_time):.2f}s")
        else:
            print(f"Time Cost Unit Test {i}: FAILED")
            
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, (ax, (P_dist, P_time)) in enumerate(zip(axes, results), 1):
            # recreate grids for visualization
            grid = empty_grid()
            load_grid(grid)
            k_val = 0.2 if i == 1 else 0.9
            for x in range(60, 91):
                for y in range(0, 41):
                    TRAVERSABILITY_GRID[x][y] = k_val
                    
            ax.imshow(TRAVERSABILITY_GRID.T, origin="lower", cmap="viridis_r")
            
            if P_dist:
                xd = [pt[0] for pt in P_dist]
                yd = [pt[1] for pt in P_dist]
                ax.plot(xd, yd, 'r-', linewidth=2, label="Distance Mode")
            if P_time:
                xt = [pt[0] for pt in P_time]
                yt = [pt[1] for pt in P_time]
                ax.plot(xt, yt, 'b--', linewidth=2, label="Time Mode")
                
            ax.set_title(f"Time Cost Unit Test {i} (k={k_val})")
            ax.set_xlabel("i (cells)")
            ax.set_ylabel("j (cells)")
            ax.legend()
            
        plt.tight_layout()
        plt.show()
        
    print(f"Time Cost Unit Tests: {passed_count}/2 passed")
    return passed_count


# =====================================================================
# PART 4 - FROM PATH TO TRAJECTORY, AND TRAJECTORY TRACKING (Lecture 06A)
# =====================================================================
# A path P only says WHERE the robot goes. A trajectory T also says WHEN:
#   T = [p_0, p_1, ..., p_M],   p_j = [t, x, y, theta]_j   (s, m, m, rad)
# With a trajectory, the robot drives slowly in slow zones, because the
# time stamps there are further apart.
TRAJ_LOOKAHEAD_S = 1       # s, the Delta t of Lecture 06A (you may tune it)
HEADING_LOOKAHEAD_CELLS = 15    # cells, see path_to_trajectory


# STUDENT TODO - Step 1: path_to_trajectory(P, cost_mode) -> T
#   P = path of cells from A_star_planner.
#   For each cell, p_j = [t_j, x_j, y_j, theta_j]:
#   - x_j, y_j:  centre of the cell in meters (cell_to_meters)
#   - t_j:       t_0 = 0, then t_j = t_(j-1) + (distance from p_(j-1) to p_j) / speed,
#                with speed = ROBOT_SPEED_M_S for "distance", and
#                k * ROBOT_SPEED_M_S of cell j for "time".
#   - theta_j:   direction of travel. On a grid path the direction of a
#                single step is always 0, 90, 180 or -90 deg, so use the
#                direction from cell j to cell j + HEADING_LOOKAHEAD_CELLS
#                (or to the last cell near the end). The last pose keeps the
#                heading of the pose before it.
def path_to_trajectory(P, cost_mode="distance"):
    if not P:
        return []
        
    T = []
    t = 0.0
    
    for j in range(len(P)):
        x, y = cell_to_meters(P[j])
        
        if j > 0:
            prev_x, prev_y = T[-1][1], T[-1][2]
            dist = math.hypot(x - prev_x, y - prev_y)
            if cost_mode == "distance":
                speed = ROBOT_SPEED_M_S
            else:
                k = traversability(P[j])
                speed = k * ROBOT_SPEED_M_S
            t += dist / speed
            
        if j == len(P) - 1:
            theta = T[-1][3] if len(T) > 0 else 0.0
        else:
            target_idx = min(j + HEADING_LOOKAHEAD_CELLS, len(P) - 1)
            dx = P[target_idx][0] - P[j][0]
            dy = P[target_idx][1] - P[j][1]
            theta = math.atan2(dy, dx)
            
        T.append([t, x, y, theta])
        
    return T


# STUDENT TODO - Step 2: trajectory_tracking_controller(T, X, t) -> (phi_dot_l, phi_dot_r)
#   T = trajectory, X = current pose [x, y, theta], t = time (s) since the
#   robot started the trajectory. Use "Trajectory Tracking Control via
#   Points" of Lecture 06A. At each time step t:
#     1. find the pose p_t on T that has the same time stamp as t
#     2. find the pose p_des,t on T that is a time TRAJ_LOOKAHEAD_S further
#        along T (the last pose of T if t + TRAJ_LOOKAHEAD_S is after the end)
#     3. return point_tracking_controller([x, y, theta] of p_des,t, X)
TURN_IN_PLACE_RAD = math.radians(100)   # if the aim point is more than this far off the
                                       # robot's heading, turn in place (never reverse)
MAX_WHEEL_SPEED_RAD_S = 7.0   # rad/s. From pwm_calibration_20261008_125131.csv: at PWM 255 the slowest
MAX_TURN_RATE_RAD_S = 1.2
MAX_AIM_AHEAD_CELLS = 20    # rad/s (69 deg/s), see the comment in the file              # wheel/direction reaches 7.76 rad/s (right wheel, backward); 90% of that
                              # leaves room for the wheel speed controller to correct
def scale_to_max_wheel_speed(phi_l, phi_r):
    factor = 1.0

    # 1. no wheel faster than the robot can drive it
    biggest = max(abs(phi_l), abs(phi_r))
    if biggest > MAX_WHEEL_SPEED_RAD_S:
        factor = MAX_WHEEL_SPEED_RAD_S / biggest

    # 2. no turn faster than MAX_TURN_RATE_RAD_S
    v, w = get_robot_velocities(phi_l, phi_r)
    if abs(w) * factor > MAX_TURN_RATE_RAD_S:
        factor = MAX_TURN_RATE_RAD_S / abs(w)

    return phi_l * factor, phi_r * factor


def trajectory_tracking_controller(T, X, t):
    if len(T) == 0:
        return 0.0, 0.0

    # 1. p_t: first pose whose time stamp is at or after t
    k = 0
    while k < len(T) - 1 and T[k][0] < t:
        k = k + 1

    # 2. p_des,t: first pose TRAJ_LOOKAHEAD_S further in time than p_t
    k_des = k
    while k_des < len(T) - 1 and T[k_des][0] < T[k][0] + TRAJ_LOOKAHEAD_S:
        k_des = k_des + 1

    closest = 0
    best_dist = math.inf
    for i in range(k_des + 1):
        d = math.hypot(T[i][1] - X[0], T[i][2] - X[1])
        if d < best_dist:
            best_dist = d
            closest = i
    if k_des > closest + MAX_AIM_AHEAD_CELLS:
        k_des = closest + MAX_AIM_AHEAD_CELLS
    p_des = T[k_des]

    # 3. never reverse: if the aim point is far off the heading, turn in place
    dx = p_des[1] - X[0]
    dy = p_des[2] - X[1]
    alpha = wrap_angle(math.atan2(dy, dx) - X[2])
    if math.hypot(dx, dy) > 0.02 and abs(alpha) > TURN_IN_PLACE_RAD:
        phi_l, phi_r = get_wheel_velocities(0.0, 3.0 * alpha)
        return scale_to_max_wheel_speed(phi_l, phi_r)

    phi_l, phi_r = point_tracking_controller(p_des[1:], X)
    return scale_to_max_wheel_speed(phi_l, phi_r)


# STUDENT TODO - Step 3: 2 unit tests, each returns (passed, T, visited_poses).
# Plan on a map of map_4.py with load_map(name) and A_star_planner, make the
# trajectory, then loop: trajectory_tracking_controller -> robot_simulator
# (t increases by DT each loop) until is_at_goal(T[-1][1:], X), with a time
# limit of T[-1][0] + 20 s. A test passes if the robot reaches the goal and
# robot_in_collision is False for every pose visited. Also record the largest
# distance between the robot and p_t (the tracking error) and print it.
#   1: map "box_sim", X = [0, 0, 0], goal (1.5, 0.0) m, cost_mode "distance"
#   2: map "terrain_sim", X = [0, 0, 0], goal (1.5, 0.0) m, cost_mode "time"
def traj_tracker_unit_test_1():
    load_map("box_sim")
    X = [0.0, 0.0, 0.0]
    goal_cell = meters_to_cell(1.5, 0.0)
    P = A_star_planner(meters_to_cell(X[0], X[1]), goal_cell, cost_mode="distance")
    T = path_to_trajectory(P, cost_mode="distance")
    
    t = 0.0
    visited_poses = [list(X)]
    max_error = 0.0
    collision = False
    
    while t <= T[-1][0] + 20.0 and not is_at_goal(T[-1][1:], X):
        k = 0
        while k < len(T) - 1 and T[k][0] < t:
            k += 1
            
        p_t = T[k]
        error = math.hypot(p_t[1] - X[0], p_t[2] - X[1])
        if error > max_error:
            max_error = error
            
        phi_l, phi_r = trajectory_tracking_controller(T, X, t)
        X, _ = robot_simulator(X, phi_l, phi_r, DT)
        visited_poses.append(list(X))
        
        if robot_in_collision(X):
            collision = True
            
        t += DT
        
    passed = is_at_goal(T[-1][1:], X) and not collision
    print(f"Test 1 Max tracking error: {max_error:.3f} m")
    return passed, T, visited_poses


def traj_tracker_unit_test_2():
    load_map("terrain_sim")
    X = [0.0, 0.0, 0.0]
    goal_cell = meters_to_cell(1.5, 0.0)
    P = A_star_planner(meters_to_cell(X[0], X[1]), goal_cell, cost_mode="time")
    T = path_to_trajectory(P, cost_mode="time")
    
    t = 0.0
    visited_poses = [list(X)]
    max_error = 0.0
    collision = False
    
    while t <= T[-1][0] + 20.0 and not is_at_goal(T[-1][1:], X):
        k = 0
        while k < len(T) - 1 and T[k][0] < t:
            k += 1
            
        p_t = T[k]
        error = math.hypot(p_t[1] - X[0], p_t[2] - X[1])
        if error > max_error:
            max_error = error
            
        phi_l, phi_r = trajectory_tracking_controller(T, X, t)
        X, _ = robot_simulator(X, phi_l, phi_r, DT)
        visited_poses.append(list(X))
        
        if robot_in_collision(X):
            collision = True
            
        t += DT
        
    passed = is_at_goal(T[-1][1:], X) and not collision
    print(f"Test 2 Max tracking error: {max_error:.3f} m")
    return passed, T, visited_poses


# STUDENT TODO: run the 2 tests, print how many pass out of 2, plot each
# planned path and the poses visited (as subfigures of one figure).
def run_traj_tracker_unit_tests(plot=True):
    tests = [traj_tracker_unit_test_1, traj_tracker_unit_test_2]
    passed_count = 0
    results = []
    
    for i, test in enumerate(tests, 1):
        passed, T, visited = test()
        results.append((T, visited))
        if passed:
            passed_count += 1
            print(f"Tracking Test {i}: PASSED")
        else:
            print(f"Tracking Test {i}: FAILED")
            
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (ax, (T, visited)) in enumerate(zip(axes, results), 1):
            if T:
                xs_t = [p[1] for p in T]
                ys_t = [p[2] for p in T]
                ax.plot(xs_t, ys_t, 'r--', linewidth=2, label="Planned Trajectory")
            if visited:
                xs_v = [p[0] for p in visited]
                ys_v = [p[1] for p in visited]
                ax.plot(xs_v, ys_v, 'b-', linewidth=1.5, label="Robot Output")
                
            ax.set_title(f"Trajectory Unit Test {i}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_aspect("equal")
            
        plt.tight_layout()
        plt.show()
        
    print(f"Tracking Tests: {passed_count}/2 passed")
    return passed_count


# =====================================================================
# PART 5 - A* ANALYSIS: PATH LENGTH AND PLANNING TIME VS MAP DENSITY
# =====================================================================
# PROVIDED - a random free cell, for random start and goal cells.
def random_free_cell(rng):
    while True:
        X = [int(rng.integers(0, GRID_N)), int(rng.integers(0, GRID_N))]
        if not is_occupied(X):
            return X


# STUDENT TODO: run_density_analysis(plot=True)
#   For 5 densities (e.g. 0.0, 0.05, 0.10, 0.15, 0.20):
#     load_grid(map_4.random_map(density, seed))
#     100 times: X_0 = random_free_cell(rng), X_d = random_free_cell(rng),
#       time the call A_star_planner(X_0, X_d) with time.perf_counter(),
#       record the path length (m) and the planning time (s).
#       Count the plans that return None (no path) separately.
#   Plot the mean path length vs density, and the mean planning time vs density.
# Tip: with a plain list as the fringe, searching it for the lowest f at every
# step makes this analysis slow (~10 min). Python's heapq module can give you
# the lowest f much faster. Mention your choice in the report.
def run_density_analysis(plot=True):
    densities = [0.0, 0.05, 0.10, 0.15, 0.20]
    rng = np.random.default_rng(0)

    mean_lengths = []
    mean_times = []

    for k in range(len(densities)):
        # --- build the map, then make the free space fully connected ---
        grid = map_4.random_map(densities[k], seed=k).copy()
        label = np.zeros((GRID_N, GRID_N), dtype=int)   # 0 = not visited yet
        sizes = [0]                                     # sizes[r] = cells of region r
        n_regions = 0
        for si in range(GRID_N):
            for sj in range(GRID_N):
                if grid[si][sj] == 1 or label[si][sj] != 0:
                    continue
                n_regions = n_regions + 1               # new free region
                label[si][sj] = n_regions
                stack = [[si, sj]]
                count = 0
                while len(stack) > 0:
                    i, j = stack.pop()
                    count = count + 1
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < GRID_N and 0 <= nj < GRID_N):
                            continue
                        if grid[ni][nj] == 1 or label[ni][nj] != 0:
                            continue
                        label[ni][nj] = n_regions
                        stack.append([ni, nj])
                sizes.append(count)
        largest = 1
        for r in range(1, n_regions + 1):
            if sizes[r] > sizes[largest]:
                largest = r
        for i in range(GRID_N):
            for j in range(GRID_N):
                if grid[i][j] == 0 and label[i][j] != largest:
                    grid[i][j] = 1                      # fill cut-off pockets
        load_grid(grid)

        # --- 100 plans on this map ---
        lengths = []
        times = []
        no_path = 0
        for _ in range(100):
            X_0 = random_free_cell(rng)
            X_d = random_free_cell(rng)
            t0 = time.perf_counter()
            P = A_star_planner(X_0, X_d, cost_mode="distance")
            seconds = time.perf_counter() - t0
            if P is None:
                no_path = no_path + 1
            else:
                lengths.append((len(P) - 1) * CELL_SIZE_M)
                times.append(seconds)

        mean_lengths.append(sum(lengths) / len(lengths))
        mean_times.append(sum(times) / len(times))
        print(f"Density {densities[k]:.2f}: mean length {mean_lengths[-1]:.3f} m, "
              f"mean time {mean_times[-1] * 1000:.1f} ms, no path: {no_path}/100")

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(densities, mean_lengths, marker="o", color="blue")
        ax1.set_title("Mean Path Length vs Density")
        ax1.set_xlabel("Map Density")
        ax1.set_ylabel("Mean Path Length (m)")
        ax2.plot(densities, mean_times, marker="s", color="red")
        ax2.set_title("Mean Planning Time vs Density")
        ax2.set_xlabel("Map Density")
        ax2.set_ylabel("Mean Planning Time (s)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== Part 2: A* (shortest path) ===")
    run_a_star_unit_tests()
    print("\n=== Part 3: A* (minimum time) ===")
    run_time_cost_unit_tests()
    print("\n=== Part 4: trajectory tracking ===")
    run_traj_tracker_unit_tests()
    print("\n=== Part 5: density analysis ===")   # slow: run it once for your report
    run_density_analysis()