import heapq
from environment import Environment, widget_get_occupied_cells
from state import State
from constants import BEE_ACTIONS


# StateWrapper class for UCS and A* priority queue management
class StateWrapper:
    def __init__(self, state, cost):
        self.state = state
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)


# Custom StateNode class for managing states and paths
class StateNode:
    """
    Adapted from the COMP3702 Tutorial 3 solution for the BeeBot environment.
    This class manages state, path, and cost information.
    """

    def __init__(self, env, state, parent=None, action_from_parent=None, path_steps=0, path_cost=0):
        self.env = env
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.path_steps = path_steps
        self.path_cost = path_cost

    def get_path(self):
        """
        Returns the list of actions from the initial state to this state.
        """
        path = []
        current = self
        while current.action_from_parent is not None:
            path.append(current.action_from_parent)
            current = current.parent
        path.reverse()
        return path

    def get_successors(self):
        """
        Generates successors from the current state.
        """
        successors = []
        for action in BEE_ACTIONS:
            success, action_cost, next_state = self.env.perform_action(self.state, action)
            if success:
                successors.append(
                    StateNode(self.env, next_state, self, action, self.path_steps + 1, self.path_cost + action_cost))
        return successors

    def __lt__(self, other):
        """
        Node comparison based on path cost.
        """
        return self.path_cost < other.path_cost


# Solver class implementing UCS and A* search methods
class Solver:
    def __init__(self, environment, lc=None):
        self.environment = environment
        self.lc = lc

    def solve_ucs(self):
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        :return: path (list of actions, where each action is an element of BEE_ACTIONS)

        The core logic of UCS was adapted from the COMP3702 Tutorial 3 solution and translated to the BeeBot environment.
        """
        frontier = [StateNode(self.environment, self.environment.get_init_state())]
        heapq.heapify(frontier)
        visited = {self.environment.get_init_state(): 0}
        n_expanded = 0

        while frontier:
            if self.lc:
                self.lc.inc()

            n_expanded += 1
            node = heapq.heappop(frontier)

            if self.environment.is_solved(node.state):
                return node.get_path()

            for s in node.get_successors():
                if s.state not in visited or s.path_cost < visited[s.state]:
                    visited[s.state] = s.path_cost
                    heapq.heappush(frontier, s)

        return None

    def solve_a_star(self):
        """
        Find a path which solves the environment using A* search.
        :return: path (list of actions, where each action is an element of BEE_ACTIONS)

        The core logic of A* search was adapted from the COMP3702 Tutorial 3 solution and translated to the BeeBot environment.
        """
        frontier = [(self.compute_heuristic(self.environment.get_init_state()),
                     StateNode(self.environment, self.environment.get_init_state()))]
        heapq.heapify(frontier)
        visited = {self.environment.get_init_state(): 0}
        n_expanded = 0

        while frontier:
            if self.lc:
                self.lc.inc()

            n_expanded += 1
            _, node = heapq.heappop(frontier)

            if self.environment.is_solved(node.state):
                return node.get_path()

            for s in node.get_successors():
                if s.state not in visited or s.path_cost < visited[s.state]:
                    visited[s.state] = s.path_cost
                    heapq.heappush(frontier, (s.path_cost + self.compute_heuristic(s.state), s))

        return None

    def compute_heuristic(self, state):
        """
        Compute a heuristic value h(n) for the given state.
        :param state: given state (GameState object)
        :return a real number h(n)
        """
        uncovered_targets = 0
        for target in self.environment.target_list:
            is_covered = False
            for i in range(self.environment.n_widgets):
                widget_cells = widget_get_occupied_cells(
                    self.environment.widget_types[i],
                    state.widget_centres[i],
                    state.widget_orients[i]
                )
                if target in widget_cells:
                    is_covered = True
                    break
            if not is_covered:
                uncovered_targets += 1
        return uncovered_targets

    def preprocess_heuristic(self):
        """
        Perform pre-processing (e.g. pre-computing repeatedly used values) necessary for your heuristic,
        """
        pass
