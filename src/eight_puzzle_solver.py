"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque

from utils import *


import numpy as np

import time



class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)




# ______________________________________________________________________________
# Uninformed Search algorithms


def breadth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = deque([Node(problem.initial)])  # FIFO queue

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None


def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    count = 0
    while frontier:
        node = frontier.pop()
        count += 1
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
                print("The total number of nodes that were removed from frontier == " + str(count))
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ______________________________________________________________________________
# A* heuristics

class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def max_h(self, node):
        a = self.h(node)
        b = self.manhattan_h(node)
        if(a > b):
            return a
        else:
            return b

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        # return sum(s != g for (s, g) in zip(node.state, self.goal))

        sum = 0
        for i in range(0, len(node.state)):
            if(node.state[i] != 0 and node.state[i] != i+1):
                sum += 1
        return sum

    # # manhattan distance function
    def manhattan_h(self,node):
        sum = 0
        for i in range (0,len(node.state)):
            if(node.state[i] != 0):
                sum += abs(int((node.state[i]-1)/3) - int(i/3)) + abs(((node.state[i]-1)%3) - i%3)
                # print("At: " + str(node.state[i]) + " " + str(abs(int((node.state[i]-1)/3) - int(i/3)) + abs(((node.state[i]-1)%3) - i%3)))
        return sum

class DuckPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP-3', 'UP-2', 'DOWN+3', 'DOWN+2', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        # if index_blank_square % 3 == 0:
        #     possible_actions.remove('LEFT')
        # if index_blank_square < 3:
        #     possible_actions.remove('UP')
        # if index_blank_square % 3 == 2:
        #     possible_actions.remove('RIGHT')
        # if index_blank_square > 5:
        #     possible_actions.remove('DOWN')

        up3ConditionList = [0, 1, 2, 3, 4, 5]
        down3ConditionList = [0, 1, 2, 6, 7, 8]
        up2Conditionist = [0, 1, 4, 5, 6, 7, 8]
        down2ConditionList = [2, 3, 4, 5, 6, 7, 8]
        leftConditionList = [0, 2, 6]
        rightConditionList = [1, 5, 8]

        if index_blank_square in up3ConditionList:
            possible_actions.remove('UP-3')
        if index_blank_square in up2Conditionist:
            possible_actions.remove('UP-2')
        if index_blank_square in down3ConditionList:
            possible_actions.remove('DOWN+3')
        if index_blank_square in down2ConditionList:
            possible_actions.remove('DOWN+2')
        if index_blank_square in rightConditionList:
            possible_actions.remove('RIGHT')
        if index_blank_square in leftConditionList:
            possible_actions.remove('LEFT')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP-3': -3, 'UP-2': -2, 'DOWN+3': 3, 'DOWN+2':2, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def max_h(self, node):
        a = self.h(node)
        b = self.manhattan_h(node)
        if(a > b):
            return a
        else:
            return b

    # def manhattan_h(self, node):
    def manhattan_h(self,node):
        sum = 0
        for i in range (0,len(node.state)):
            if(node.state[i] != 0):
                # sum += abs(int(node.state[i]/3) - int(i/3)) + abs((node.state[i]%3-1) - i%3)
                if node.state[i] < 4:
                    if (node.state[i] == 1 and i == 3):
                        sum += 2
                    elif((node.state[i] == 2 and i == 2)):
                        sum += 2
                    elif(node.state[i] == 3 and i == 1):
                        sum += 2
                    elif(node.state[i]-1 != i):
                        sum += 1
                else:
                    sum += abs(int((node.state[i]-4)/3) - int((i-3)/3)) + abs(((node.state[i]-4)%3) - (i-3)%3)
            # print(sum)
        return sum

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        # return sum(s != g for (s, g) in zip(node.state, self.goal))
        sum = 0
        for i in range(0, len(node.state)):
            if(node.state[i] != 0 and node.state[i] != i+1):
                sum += 1
        return sum


#  Helper functions

# making random 8puzzle
def make_rand_8puzzle():
    state = tuple(np.random.permutation(9))
    return state

# making random duck_puzzle
def make_rand_duckpuzzle():
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    create_duckpuzzle = DuckPuzzle(initial=goal_state)
    
    random_mix = random.randint(0, 100)
    # print("random mix times == " + str(random_mix))
    current_state = goal_state
    for i in range (random_mix):
        # print(str(i) + ":")
        actionlist = create_duckpuzzle.actions(state= current_state)
        random_action_index = random.randint(0, len(actionlist)-1)
        next_state = create_duckpuzzle.result(current_state, actionlist[random_action_index])
        current_state = next_state
        # display_duck(current_state)
    
    return current_state


# displaying 8-Puzzle in a 3*3 grid
def display(state):
    msg = ""
    for i in range(0, 9):
        if((i+1) % 3 == 0):
            if(state[i] != 0):
                msg += str(state[i])
            else:
                msg += "*"
            print(msg)
            msg = ""
        else:
            if(state[i] != 0):
                msg += str(state[i]) + " "
            else:
                msg += "* "

# displaying duck_puzzle
def display_duck(state):
    msg = ""
    i = 0
    for i in range(0, 2):
        if(state[i] == 0):
            msg += "* "
        else:
            msg += str(state[i]) + " "
    print(msg)
    msg = ""

    for i in range(2, 6):
        if(state[i] == 0):
            msg += "* "
        else:
            msg += str(state[i]) + " "
    print(msg)
    msg = "  "

    for i in range(6, 9):
        if(state[i] == 0):
            msg += "* "
        else:
            msg += str(state[i]) + " "
    print(msg)

# test function for 8-puzzle
def eight_puzzle_test():

    print("---------- Test 8-puzzle ----------\n")

    # create solvable 8-puzzle
    while(True):
        # making the random 8 puzzle state
        state = make_rand_8puzzle()

        dummy_puzzle = EightPuzzle(initial = state)
        if(dummy_puzzle.check_solvability(state)):
            break

    startNode = Node(state,None, None, 0)
    # display state
    display(state)

    # if the state is solvable proceeds the following
    if dummy_puzzle.check_solvability(state):

        # default heuristic function

        print("--------8-Puzzle: Default misplaced tiled heuristic function--------")
        
        start_time = time.time()
        endNode = astar_search(dummy_puzzle, display = True)
        elapsed_time = time.time() - start_time
        print(endNode)
        print(endNode.solution())

        length = len(endNode.solution())

        print("the length (i.e. number of tiles moved) of the solution == " + str(length))

        print(f'elapsed time (in seconds): {elapsed_time}s\n')
        

        # # Manhattan heuristic function
    
        print("--------8-Puzzle: Manhattan heuristic function--------")

        # print(dummy_puzzle.manhattan_h(startNode))

        start_time = time.time()
        endNode = astar_search(dummy_puzzle, h = dummy_puzzle.manhattan_h, display = True)
        elapsed_time = time.time() - start_time

        print(endNode)
        print(endNode.solution())
        length = len(endNode.solution())
        print("the length (i.e. number of tiles moved) of the solution == " + str(length))

        print(f'elapsed time (in seconds): {elapsed_time}s\n')

        # max

        print("--------8-Puzzle: Max heuristic function--------")

        start_time = time.time()
        endNode = astar_search(dummy_puzzle, h = dummy_puzzle.max_h, display = True)
        elapsed_time = time.time() - start_time
        print(endNode)
        print(endNode.solution())

        length = len(endNode.solution())

        print("the length (i.e. number of tiles moved) of the solution == " + str(length))
        
        print(f'elapsed time (in seconds): {elapsed_time}s\n\n')

    else:
        print("Not solvable!\n")

# test function for duck_puzzle
def duck_puzzle_test():

    print("---------- Test Duck_puzzle ----------\n")

    # making the random solvable duck_puzzle state
    state = make_rand_duckpuzzle()
    print("randomly created solvable duck_puzzle is:")
    print(state)

    # display state
    display_duck(state)

    duck_puzzle = DuckPuzzle(initial = state)
    
    startNode = Node(state,None, None, 0)

    # # default heuristic function
    print("--------default duck_puzzle--------")

    start_time = time.time()
    endNode = astar_search(duck_puzzle, display = True)
    elapsed_time = time.time() - start_time

    print(endNode)
    print(endNode.solution())
    length = len(endNode.solution())
    print("the length (i.e. number of tiles moved) of the solution == " + str(length))

    print(f'elapsed time (in seconds): {elapsed_time}s\n')
        

    # # Manhattan heuristic function
    
    print("--------Manhattan duck_puzzle-------")

    start_time = time.time()
    endNode = astar_search(duck_puzzle, h = duck_puzzle.manhattan_h, display = True)
    elapsed_time = time.time() - start_time

    print(endNode)
    print(endNode.solution())
    length = len(endNode.solution())
    print("the length (i.e. number of tiles moved) of the solution == " + str(length))
    
    print(f'elapsed time (in seconds): {elapsed_time}s\n')

    # # max
    print("--------Max duck_puzzle-------")
    start_time = time.time()
    endNode = astar_search(duck_puzzle, h = duck_puzzle.max_h, display = True)
    elapsed_time = time.time() - start_time

    print(endNode)
    print(endNode.solution())
    length = len(endNode.solution())
    print("the length (i.e. number of tiles moved) of the solution == " + str(length))
    
    print(f'elapsed time (in seconds): {elapsed_time}s\n')


if __name__ == '__main__': 
    eight_puzzle_test()
    duck_puzzle_test()
