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

import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import ActionId, Environment, Observation, State

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # down
    (0, -1),  # up
    (-1, 0),  # left
    (1, 0),   # right
    (0, 0),   # put/take
]


OBJECTS1 = dict([(v, k) for k, v in enumerate(
    ["tool", "food", "plate", "tool_in_place", "plate_in_place"])])

OBJECTS2 = dict([(v, k) for k, v in enumerate(
    [ "food", "no_sign", "tool"])])

OBJECTS3 = dict([(v, k) for k, v in enumerate(
    ["tool", "food", "open"])])

def update_facts(problem_mode, facts: Sequence[bool], objects: Observation, grab=False):
    state = set([i for i, v in enumerate(facts) if v])
    put_down = False
    if problem_mode == 1:
        for o in objects:
            if o == "tool" and grab and OBJECTS1["food"] not in state:
                state.add(OBJECTS1[o])
            elif o == "cupboard" and grab and OBJECTS1["food"] not in state:
                state.add(OBJECTS1["plate"])
            elif o == "oven" and OBJECTS1["tool"] in state and OBJECTS1["plate"] in state and grab:
                state.add(OBJECTS1["food"])
            elif o == "tool" or o == "cupboard":
                if OBJECTS1["tool"] in state and OBJECTS1["food"] in state and grab:
                    state.remove(OBJECTS1["tool"])
                    put_down = True
                    if o == "tool":
                        state.add(OBJECTS1["tool_in_place"])

                elif OBJECTS1["plate"] in state and OBJECTS1["food"] in state and grab:
                    state.remove(OBJECTS1["plate"])
                    put_down = True
                    if o == "cupboard":
                        state.add(OBJECTS1["plate_in_place"])


    if problem_mode == 2:
        for o in objects:
            if o == "tool" and grab:
                state.add(OBJECTS2[o])
            if o == "oven" and OBJECTS2["tool"] in state and grab:
                state.add(OBJECTS2["food"])
            if o == "sign":
                state.add(OBJECTS2["no_sign"])
    if problem_mode == 3:
        for o in objects:
            if o == "tool" and grab and not OBJECTS3["open"] in state:
                state.add(OBJECTS3[o])
                state.add(OBJECTS3["open"])
            elif o == "tool" and grab and OBJECTS3["open"] in state:
                state.remove(OBJECTS3["open"])
            elif o == "oven" and OBJECTS3["tool"] in state and grab:
                state.add(OBJECTS3["food"])
    return state, put_down


class KitchenState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]

    def __init__(self, x: int, y: int, facts: Set[int], default_x, default_y, tool_x=-1, tool_y=-1, plate_x = -1, plate_y = -1):
        self.x = x
        self.y = y
        self.default_x = default_x
        self.default_y = default_y
        fact_list = [False] * len(OBJECTS1)

        for fact in facts:
            fact_list[fact] = True

        self.facts = tuple(fact_list)

        self.tool_x = tool_x
        self.tool_y = tool_y

        self.plate_x = plate_x
        self.plate_y = plate_y
        self.uid = (self.x, self.y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {:2d}, {:2d}, {})".format(self.x, self.y, self.facts)

    @staticmethod
    def random(problem_mood, rng: Random,
               map_data: Sequence[Sequence[Observation]], default_x, default_y, randomness=True) -> 'KitchenState':

        while True:
            x = default_x
            y = default_y
            if "wall" not in map_data[y][x]:
                facts, put_down = update_facts(problem_mood, (), map_data[y][x])
                return KitchenState(x, y, facts, default_x, default_y)


MAPPING: Mapping[str, FrozenSet[str]] = {
    't': frozenset(["tool"]),
    'p': frozenset(["counter"]),
    'c': frozenset(["cupboard"]),
    'o': frozenset(["oven"]),
    ' ': frozenset(),
    'X': frozenset(["wall"]),
    'w': frozenset(["sign"]),
    'f': frozenset(["fridge"]),
    }


def load_map(map_fn: str):
    with open(map_fn) as map_file:
        array = []
        for l in map_file:
            if len(l.rstrip()) == 0:
                continue

            row = []
            for cell in l.rstrip():
                row.append(MAPPING[cell])
            array.append(row)

    return array


class Kitchen(Environment[KitchenState]):
    map_data = [[]]
    num_actions = 5

    def __init__(self, map_fn: str, rng: Random, default_x, default_y, problem_mode, noise = 0.0):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.key_locations = self.get_all_item()
        self.default_x = default_x
        self.default_y = default_y
        self.noise = noise
        self.problem_mode = problem_mode

        super().__init__(KitchenState.random(problem_mode, self.rng, self.map_data, default_x, default_y))


    def get_all_item(self, item="key"):
        ans = []
        for y in range(self.height):
            for x in range(self.width):
                if item in self.map_data[y][x]:
                    ans.append([y, x])
        return ans

    def apply_action(self, a: ActionId):
        if self.rng.random() < self.noise:
            a = self.rng.randrange(self.num_actions)

        x, y = self.state.x + ACTIONS[a][0], self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x] :
            return
        objects = self.map_data[y][x]

        grab = False
        if a >= 4:
            grab = True
        new_facts, put_down = update_facts(self.problem_mode, self.state.facts, objects, grab = grab)
        tool_x = self.state.tool_x
        tool_y = self.state.tool_y
        plate_x = self.state.plate_x
        plate_y = self.state.plate_y

        if grab and put_down:
            if OBJECTS1["plate"] in new_facts:
                tool_x = x
                tool_y = y
            else:
                plate_x = x
                plate_y = y


        self.state = KitchenState(x, y, new_facts, self.default_x, self.default_y, tool_x, tool_y, plate_x, plate_y)
        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: KitchenState, a: ActionId, s1: KitchenState) -> float:
        return 1.0

    def observe(self, state: KitchenState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[KitchenState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = KitchenState.random(self.problem_mode, self.rng, self.map_data, self.default_x, self.default_y, randomness=True)
