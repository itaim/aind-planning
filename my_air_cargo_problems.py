from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache
from copy import deepcopy

class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list
            :return: list of Action objects
            """
            loads = []
            for p in self.planes:
                for c in self.cargos:
                    for a in self.airports:
                        precond_pos = [expr("At({},{})".format(c,a)),expr("At({},{})".format(p,a))]
                        precond_neg = []
                        effect_add = [expr("In({},{})".format(c,p))]
                        effect_rem = [expr("At({},{})".format(c,a))]
                        load = Action(expr("Load({},{},{})".format(c,p,a)),[precond_pos,precond_neg],[effect_add,effect_rem])
                        loads.append(load)

            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list
            :return: list of Action objects
            """
            unloads = []
            for p in self.planes:
                for c in self.cargos:
                    for a in self.airports:
                        precond_pos = [expr("In({},{})".format(c, p)), expr("At({},{})".format(p, a))]
                        precond_neg = []
                        effect_add = [expr("At({},{})".format(c, a))]
                        effect_rem = [expr("In({},{})".format(c, p))]
                        load = Action(expr("Unload({},{},{})".format(c, p, a)), [precond_pos, precond_neg],
                                      [effect_add, effect_rem])
                        unloads.append(load)
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr))]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        def is_cond_met(cond, required_state):
            idx = self.state_map.index(cond)
            current_state = state[idx]
            return current_state == required_state

        def is_possible_action(action):
            for precond in action.precond_pos:
                if not is_cond_met(precond,"T"):
                    return False
            for precond in action.precond_neg:
                if not is_cond_met(precond, "F"):
                    return False
            return True

        possible_actions = []
        for action in self.actions_list:
            if is_possible_action(action):
                possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """

        # pos_list = self.state_map[:state.index("F")]
        # neg_list = self.state_map[state.index("F"):]
        new_state_chars = [c for c in state]
        for exp in action.effect_add:
            idx = self.state_map.index(exp)
            new_state_chars[idx] = "T"
        for exp in action.effect_rem:
            idx = self.state_map.index(exp)
            new_state_chars[idx] = "F"
        pos_list = []
        neg_list = []
        for i,s in enumerate(new_state_chars):
            if s == "T":
                pos_list.append(self.state_map[i])
            else:
                neg_list.append(self.state_map[i])

        new_state = FluentState(pos_list, neg_list)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        # print('node action {}'.format(node.action))
        # print('node state {}'.format(node.state))
        # print('action list {}'.format([str(a) for a in self.actions_list]))
        # print('actions {}'.format([str(a) for a in self.actions(node.state)]))
        # print('state map {}'.format(self.state_map))
        # print('goal {}'.format(self.goal))
        # First relax the actions by removing all preconditions and all effects except for those that are literals in the goal.
        # for a in self.actions_list:
        #     action = deepcopy(a)
        #     action.precond_pos = []
        #     action.precond_neg = []
        #     action.effect_rem = []
        #     action.effect_add = [e for e in action.effect_add if e in self.goal]
        # Then we count the minimum number of actions required so that the union of those actions effects satisfies the goal.

        for s in self.goal:
            idx  = self.state_map.index(s)
            if node.state[idx] == 'F':
                count += 1
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    plane_airports = {'P1,SFO', 'P2,JFK', 'P3,ATL'}
    cargos_airports = {'C1,SFO', 'C2,JFK', 'C3,ATL'}
    positive_states = [expr('At({})'.format(pap)) for pap in plane_airports] + [expr('At({})'.format(cap)) for cap in cargos_airports]
    negative_cargos_in_planes = [expr('In({}, {})'.format(c, p)) for c in cargos for p in planes]
    negative_planes_at_airports = [expr('At({}, {})'.format(p, ap)) for p in planes for ap in airports if '{},{}'.format(p, ap) not in plane_airports]
    negative_cargos_at_airports = [expr('At({}, {})'.format(c, ap)) for c in cargos for ap in airports if '{},{}'.format(c, ap) not in cargos_airports]
    negative_states = negative_cargos_in_planes + negative_planes_at_airports + negative_cargos_at_airports
    # print("positive {}\nnegative {}".format(positive_states,negative_states))
    init = FluentState(positive_states, negative_states)

    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)

def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL','ORD']
    # planes_at_airports = {'P1,SFO', 'P2,JFK'}
    # cargos_at_airports = {'C1,SFO', 'C2,JFK', 'C3,ATL', 'C4,ORD'}
    # pos_planes_at_airports = [expr('At({})'.format(pap)) for pap in planes_at_airports]
    # pos_cargos_at_airports = [expr('At({})'.format(cap)) for cap in cargos_at_airports]
    # positive_states = pos_cargos_at_airports + pos_planes_at_airports

    # negative_cargos_in_planes = [expr('In({}, {})'.format(c, p)) for c in cargos for p in planes]
    # negative_planes_at_airports = [expr('At({}, {})'.format(p, ap)) for p in planes for ap in airports if '{},{}'.format(p, ap) not in planes_at_airports]
    # negative_cargos_at_airports = [expr('At({}, {})'.format(c, ap)) for c in cargos for ap in airports if '{},{}'.format(c, ap) not in cargos_at_airports]
    # negative_states = negative_cargos_in_planes + negative_planes_at_airports + negative_cargos_at_airports
    negative_states_strs = ["At(C4, JFK)", "At(C4, SFO)", "At(C4, ATL)", "In(C4, P1)", "In(C4, P2)",
                        "At(C3, JFK)", "At(C3, SFO)", "At(C3, ORD)", "In(C3, P1)", "In(C3, P2)",
                        "At(C2,SFO)", "At(C2, ATL)", "At(C2, ORD)", "In(C2, P1)", "In(C2, P2)",
                        "At(C1, JFK)", "At(C1, ATL)", "At(C1, ORD)", " In(C1, P1)", "In(C1, P2)",
                        "At(P1, JFK)", "At(P1, ATL)", "At(P1, ORD)",
                        "At(P2, SFO)", "At(P2, ATL)", "At(P2, ORD)"]
    negative_states = [expr(s) for s in negative_states_strs]
    positive_states = [expr("At(C1, SFO)"), expr("At(C2, JFK)"), expr("At(C3, ATL)"), expr("At(C4, ORD)"), expr("At(P1, SFO)"), expr("At(P2, JFK)")]
    init = FluentState(positive_states, negative_states)
    # print("positive {}\nnegative {}".format(positive_states, negative_states))
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)