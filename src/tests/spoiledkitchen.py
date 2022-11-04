import sys  # noqa
from os import path as p  # noqa


from rl.qvalue import EpsilonGreedy
from rl.rl import Agent
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa
import logging
from os import path
from random import Random
from time import time


from environment import ReachFacts
from environment.kitchen import Kitchen, KitchenState, OBJECTS3, update_facts, MAPPING
from utils.report import SequenceReport
from rl.baseline_empathic import BaseEmpathic



DEFAULT_Q = -1000.0

TOTAL_STEPS1 = 300000
TOTAL_STEPS2 = 300000

EPISODE_LENGTH = 1000
TEST_EPISODE_LENGTH = 50
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)

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

def print_state(state, action):
    if action == 0:
        print("Agent Location:", state.x, state.y, "Action: Down", state.facts)
    elif action == 1:
        print("Agent Location:", state.x, state.y, "Action: Up", state.facts)
    elif action == 2:
        print("Agent Location:", state.x, state.y,  "Action: Left", state.facts)
    elif action == 3:
        print("Agent Location:", state.x, state.y,  "Action: Right", state.facts)
    elif action == 4:
        print("Agent Location:", state.x, state.y,  "Action: Do", state.facts)
    else:
        print("Agent Location:", state.x, state.y)


def evaluate_agent(env, policy1, reward1, init):
    print("Evaluation:")
    state_rewards = []
    for initial_state1 in init:
        env.reset(initial_state1)
        reward1.reset()
        policy1.reset(evaluation=True)

        trial_reward: float = 0.0

        for step in range(TEST_EPISODE_LENGTH):
            s0 = env.state
            a = policy1.get_best_action(s0)
            env.apply_action(a)
            s1 = env.state
            print_state(s0, a)
            step_reward, finished = reward1(s0, a, s1)
            if not finished:
                trial_reward += step_reward
            logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)
            if finished:
                print_state(s1, -1)
                break

        state_rewards.append(trial_reward)
        return trial_reward, s1



def print_moves(prev, dest):
    rewards = 0
    lr = dest[0] - prev[0]
    ud = dest[1] - prev[1]
    lr_str = "Right"
    ud_str = "Down"
    dirs = { "Down":   (0, 1),   "Up": (0, -1),  "Left": (-1, 0),  "Right": (1, 0)}
    if lr < 0:
        lr_str = "Left"
        lr *= -1
    if ud < 0:
        ud_str = "Up"
        ud *= -1
    for i in range(int(lr)):
        print("Agent Location:", prev[0], prev[1], "Action:", lr_str)
        prev[0] += dirs[lr_str][0]
        prev[1] += dirs[lr_str][1]
        rewards += -1
    for i in range(int(ud)):
        print("Agent Location:", prev[0], prev[1], "Action:", ud_str)
        prev[0] += dirs[ud_str][0]
        prev[1] += dirs[ud_str][1]
        rewards += -1
    return rewards


def evaluate_human(food, init, policy):
    prev_loc = init
    rewards = 0
    door = policy["door"]
    fridge = policy["fridge"]

    cupboard = policy["cupboards"][0]
    rewards += print_moves(prev_loc, fridge)
    prev_loc = fridge
    print("Agent Location:", prev_loc[0], prev_loc[1], "Action:", "Do")
    rewards += -1
    if food:
        rewards += print_moves(prev_loc, cupboard)
        prev_loc = cupboard
        print("Agent Location:", prev_loc[0], prev_loc[1], "Action:", "Do")
        rewards += -1
        print("Agent got sick: -5")
        rewards += -5

    rewards += print_moves(prev_loc, door)

    print("Total Reward:", rewards)



def create_init(init_locations):
    ans = []
    for j in init_locations:
        ans.append(KitchenState(j[0], j[1], (), j[0], j[1]))

    return ans

def generate_policies(map_fn):
    map = load_map(map_fn)
    oven = None
    counter = None
    cupboard = None
    door = [1,3]
    tool = None
    fridge = None

    for x in range(len(map)):
        for y in range(len(map[0])):
            if "oven" in map[x][y]:
                oven = [y,x]
            if "counter" in map[x][y]:
                counter = [y,x]
            if "cupboard" in map[x][y]:
                cupboard = [y,x]
            if "tool" in map[x][y]:
                tool = [y,x]
            if "fridge" in map[x][y]:
                fridge = [y,x]
    policy1 = {"oven": oven, "counter": counter, "door": door, "cupboards": [tool, cupboard], "fridge": fridge}
    policy2 = {"oven": oven, "counter": counter, "door": door, "cupboards": [cupboard, tool], "fridge": fridge}
    return [policy1, policy2]

def train(filename, seed):
    logging.basicConfig(level=logging.DEBUG)
    problem_mode = 3
    here = path.dirname(__file__)
    map_fn = path.join(here, "maps/epkitchen.map")
    rng1 = Random(seed + 1)
    env1 = Kitchen(map_fn, rng1, 1, 3, problem_mode = problem_mode)
    init1 = create_init([[1,3]])


    rng2 = Random(seed + 2)
    env2 = Kitchen(map_fn, rng2, 1, 3, problem_mode = problem_mode)
    init2 = create_init([[1,3]])

    rng3 = Random(seed + 2)
    env3 = Kitchen(map_fn, rng3, 1, 3, problem_mode = problem_mode)
    init3 = create_init([[1,3]])

    tasks = [[OBJECTS3["food"]]]
    not_task = []
    tasks = tasks[START_TASK:END_TASK+1]
    with open(filename, "w") as csvfile:

        print("ql: begin experiment")

        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))

            reward1 = ReachFacts(env1, goal, not_task)
            policy1 = EpsilonGreedy(alpha=1.0, gamma=1, epsilon=0.4,
                                    default_q=DEFAULT_Q, num_actions=5, rng=rng1)
            agent1 = Agent(env1, policy1, reward1, rng1)
            report1 = SequenceReport(csvfile, LOG_STEP, init1, EPISODE_LENGTH, TRIALS)

            humans_policy = generate_policies(map_fn)

            reward2 = ReachFacts(env2, goal, not_task)
            policy2 = BaseEmpathic(alpha=1.0, gamma=1.0, epsilon=0.4,
                                       default_q=DEFAULT_Q, num_actions=5, rng=rng1, others_policy=humans_policy, others_init=[[1,3]], others_dist=[0.25, 0.75], penalty=-2*EPISODE_LENGTH, others_alpha=[4.01, 4.01], objects=OBJECTS3, problem_mode=problem_mode)
            agent2 = Agent(env2, policy2, reward2, rng2)
            report2 = SequenceReport(csvfile, LOG_STEP, init2, EPISODE_LENGTH, TRIALS)

            reward3 = ReachFacts(env3, goal, not_task)
            policy3 = BaseEmpathic(alpha=1.0, gamma=1.0, epsilon=0.4,
                                   default_q=DEFAULT_Q, num_actions=5, rng=rng3, others_policy=humans_policy,
                                   others_init=[[1, 3]], others_dist=[0.25, 0.75], penalty=-2 * EPISODE_LENGTH,
                                   others_alpha=[1.01, 1.01], objects=OBJECTS3, problem_mode=problem_mode,
                                   full_observability=True)
            agent3 = Agent(env3, policy3, reward3, rng3)
            report3 = SequenceReport(csvfile, LOG_STEP, init3, EPISODE_LENGTH, TRIALS)

            try:
                agent1.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report1)

                agent2.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report2)

                agent3.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report3)

                print("Default Human:")
                humans = generate_policies(map_fn)
                evaluate_human(False, [1, 3], humans[0])
                print("-----------------")
                print("Non-Augmented Robot:")
                base_step, last_state1 = evaluate_agent(env1, policy1, reward1, init1)
                print("Total Reward = ", base_step)
                print("-----------------")
                print("Human:")
                humans = generate_policies(map_fn)
                evaluate_human(last_state1.facts[2], [1, 3], humans[0])
                print("-----------------")
                print("Our Robot:")
                base_step2, last_state2 = evaluate_agent(env2, policy2, reward2, init2)
                print("Total Reward = ", base_step2)
                print("-----------------")
                print("Human:")
                humans = generate_policies(map_fn)
                evaluate_human(last_state2.facts[2], [1, 3], humans[0])
                print("-----------------")
                print("Robot with Full Observability Assumption:")
                base_step3, last_state3 = evaluate_agent(env3, policy3, reward3, init3)
                print("Total Reward = ", base_step3)
                print("-----------------")
                print("Human:")
                humans = generate_policies(map_fn)
                evaluate_human(last_state3.facts[2], [1, 3], humans[0])

            except KeyboardInterrupt:

                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)


start = time()
train("./test.csv", 2019)
end = time()
print("Total Time:", end - start)

