#Copyright (c) 2022 Be Considerate: Avoiding Negative Side Effects
#in Reinforcement Learning Authors

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from random import Random
from typing import Dict, Hashable, List, Optional, Tuple

from environment import ActionId, State
from environment.kitchen import OBJECTS1, OBJECTS2
from .rl import Policy

class BaseEmpathic(Policy):
    alpha: float
    default_q: Tuple[float, bool]
    gamma: float
    num_actions: int
    Q: Dict[Tuple[Hashable, ActionId], Tuple[float, bool]]
    rng: Random

    def __init__(self, alpha: float, gamma: float, epsilon: float, default_q: float,
                 num_actions: int, rng: Random, others_policy, penalty: int, others_alpha, objects, problem_mode=1, others_dist = [1.0], others_init = [[1, 1]], our_alpha=1.0, do_locs = None, full_observability = False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_q = default_q, False
        self.num_actions = num_actions
        self.rng = rng
        self.Q = {}
        self.others_policy = others_policy
        self.penalty = penalty
        self.others_alpha = others_alpha
        self.our_alpha = our_alpha
        self.others_dist = others_dist
        self.objects = objects
        self.others_init = others_init
        self.problem_mode = problem_mode
        self.full_observability = full_observability
        self.do_locs = do_locs

    def clear(self):
        self.Q = {}

    def estimate(self, state: State) -> float:
        max_q = self.Q.get((state.uid, 0), self.default_q)
        for action in range(1, self.num_actions):
            q = self.Q.get((state.uid, action), self.default_q)
            if q > max_q:
                max_q = q
        return max_q[0]

    def get_best_action(self, state: State,
                        restrict: Optional[List[int]] = None) -> ActionId:
        if restrict is None:
            restrict = list(range(self.num_actions))
        max_q = self.Q.get((state.uid, restrict[0]), self.default_q)
        best_actions = [restrict[0]]
        for action in restrict[1:]:
            q = self.Q.get((state.uid, action), self.default_q)
            if q > max_q:  # or (self.evaluation and q[1] and not max_q[1]):
                max_q = q
                best_actions = [action]
            elif q == max_q:
                best_actions.append(action)

        return self.rng.choice(best_actions)

    def get_policy(self):
        ans = {}
        for state,_ in self.Q:
            max_q = self.Q.get((state, 0), self.default_q)
            max_action = 0
            for action in range(1, self.num_actions):
                q = self.Q.get((state, action), self.default_q)
                if q > max_q:
                    max_q = q
                    max_action = action
            ans[state] = max_action
        return ans

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None, restrict_locations = None) -> ActionId:
        restrict = list(range(self.num_actions))
        if restrict_locations != None:
            for loc_r in restrict_locations:
                if state.uid[0] == loc_r[0] and state.uid[1] == loc_r[1]:
                    restrict = list(range(self.num_actions - 1))

        if self.rng.random() < self.epsilon:
            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)

    def get_max_q(self, uid, qval):
        max_q = qval.get((uid, 0), (-1000, False))
        for action in range(1, self.num_actions):
            q = qval.get((uid, action), (-1000, False))
            if q > max_q:
                max_q = q
        if max_q[0] == 0:
            print("zero!", uid)
        return max_q[0]

    def estimate_other(self, state, prev_state):

        ans = 0
        tool_x = state.tool_x
        tool_y = state.tool_y

        plate_x = state.plate_x
        plate_y = state.plate_y
        for i,other in enumerate(self.others_policy):
            other_value = 0
            prev_loc = [state.default_x, state.default_y]
            if self.problem_mode == 1:

                cupboards = other["cupboards"]
                fridge = other["fridge"]
                door = other["door"]
                target = other["target"]
                target_x = plate_x
                target_y = plate_y

               # print(tool_x, tool_y, plate_x, plate_y, target, self.full_observability)
                if target == "tool":
                    target_x = tool_x
                    target_y = tool_y

                if self.full_observability:
                    other_value += self.gamma * (abs(target_x - prev_loc[0]) + abs(target_y - prev_loc[1]))
                    prev_loc = [target_x, target_y]
                    other_value += self.gamma * 1

                else:
                    for cup in cupboards:
                        other_value += self.gamma * (abs(cup[0] - prev_loc[0]) + abs(cup[1] - prev_loc[1]))
                        prev_loc = cup.copy()
                        other_value += self.gamma * 1
                        if cup[0] == target_x and cup[1] == target_y:
                            break


                other_value += self.gamma * (abs(fridge[0] - prev_loc[0]) + abs(fridge[1] - prev_loc[1]))
                prev_loc = fridge
                other_value += self.gamma * 1
                other_value += self.gamma * (abs(door[0] - prev_loc[0]) + abs(door[1] - prev_loc[1]))

            elif self.problem_mode == 2:
                fridge = other["fridge"]
                door = other["door"]
                other_value += self.gamma * (abs(fridge[0] - prev_loc[0]) + abs(fridge[1] - prev_loc[1]))
                prev_loc = fridge
                if state.facts[1] and not self.full_observability:
                    other_value += self.gamma * 10
                other_value += self.gamma * (abs(door[0] - prev_loc[0]) + abs(door[1] - prev_loc[1]))
            elif self.problem_mode == 3:
                fridge = other["fridge"]
                door = other["door"]
                cup = other["cupboards"][0]
                other_value += self.gamma * (abs(cup[0] - prev_loc[0]) + abs(cup[1] - prev_loc[1]))
                prev_loc = cup
                other_value += self.gamma * (abs(fridge[0] - prev_loc[0]) + abs(fridge[1] - prev_loc[1]))
                prev_loc = fridge
                if state.facts[2] and not self.full_observability:
                    other_value += self.gamma * 5
                other_value += self.gamma * (abs(door[0] - prev_loc[0]) + abs(door[1] - prev_loc[1]))
            other_value *= -1
            ans += self.others_dist[i] * self.others_alpha[i] * other_value
        return ans

    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool):
        q = (1.0 - self.alpha) * self.Q.get((s0.uid, a), self.default_q)[0]
        if end:
            others = self.estimate_other(s1, s0)
            q += self.alpha * (self.our_alpha * r + self.gamma * others)
        else:
            q += self.alpha * (self.our_alpha * r + self.gamma * (self.estimate(s1)))

        self.Q[(s0.uid, a)] = q, True

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def report(self) -> str:
        return "|Q| = {}".format(len(self.Q))

    def get_Q(self):
        return self.Q.copy()



