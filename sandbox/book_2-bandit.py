import math
import torch
import random
import progressbar
import matplotlib.pyplot as plt


'''
basic testbed k-armed bandit implementation
'''


class KBandit(object):
    def __init__(self, arm_count, nonstationary=False):
        self.arm_count = arm_count
        self.nonstationary = nonstationary

        self.non_stationary_dist = None
        self.arm_values = None

        self.reset()

    def reset(self):
        values_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.non_stationary_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.01]))

        values = []
        for i in range(self.arm_count):
            values.append(values_dist.sample().item())

        self.arm_values = torch.tensor(values)

    def play(self, arm):
        mean = torch.tensor([self.arm_values[arm]])
        deviation = torch.tensor([1.0])

        reward_dist = torch.distributions.normal.Normal(mean, deviation)

        if self.nonstationary:
            for i in range(self.arm_count):
                self.arm_values[i] += self.non_stationary_dist.sample().item()

        return reward_dist.sample().item()

    def optimal_action(self):
        return torch.max(torch.tensor(self.arm_values), 0)[1].item()


class Testbed(object):
    def __init__(self, num_tests=2000, num_steps=1000):
        self.num_tests = num_tests
        self.num_steps = num_steps

    def run(self, bandits, algorithm_class, params):
        total_average_rewards = []
        total_optimal_actions = []

        with progressbar.ProgressBar(max_value=self.num_tests) as bar:
            for test in range(0, self.num_tests):
                algorithm = algorithm_class(bandits[test], params)

                for step in range(0, self.num_steps):
                    algorithm.run_step()

                total_average_rewards.append(algorithm.average_rewards)
                total_optimal_actions.append(algorithm.optimal_actions)

                bar.update(test)

        total_average_rewards = torch.tensor(total_average_rewards)
        total_optimal_actions = torch.tensor(total_optimal_actions)

        total_average_rewards = torch.mean(total_average_rewards, 0)
        total_optimal_actions = torch.mean(total_optimal_actions, 0)

        return total_average_rewards, total_optimal_actions


class Algorithm(object):
    def __init__(self, k_bandit, params):
        self.k_bandit = k_bandit
        self.params = params
        self.optimal_action = 0

        self.ucb = self._get_param("ucb", False)
        self.init_value = self._get_param("init_value", 0.0)
        self.epsilon = self._get_param("epsilon", 0.0)
        self.alpha = self._get_param("alpha", 0.1)
        self.confidence = self._get_param("confidence", 0.0)

        self.Q = torch.zeros(self.k_bandit.arm_count) + self.init_value  # + torch.rand(k_bandit.arm_count) / 1000000

        self.current_step = 0

        self.total_rewards = 0
        self.average_rewards = []
        self.optimal_actions = []
        self.rewards_for_action = torch.zeros(self.k_bandit.arm_count)
        self.plays_for_action = torch.zeros(self.k_bandit.arm_count)

        self.rewards_for_action_history = [[] for i in range(self.k_bandit.arm_count)]

        #self.reset_test()

    def reset_test(self):
        self.Q = torch.zeros(self.k_bandit.arm_count) + self.init_value  # + torch.rand(k_bandit.arm_count) / 1000000

        self.current_step = 0

        self.total_rewards = 0
        self.average_rewards = []
        self.optimal_actions = []
        self.rewards_for_action = torch.zeros(self.k_bandit.arm_count)
        self.plays_for_action = torch.zeros(self.k_bandit.arm_count)

        self.rewards_for_action_history = [[] for i in range(self.k_bandit.arm_count)]

    def run_step(self):
        self.optimal_action = self.k_bandit.optimal_action()
        self.current_step += 1

        action = self.get_action()

        reward = self.k_bandit.play(action)

        self.rewards_for_action[action] += reward
        self.plays_for_action[action] += 1
        self.rewards_for_action_history[action].append(reward)

        q = self.compute(reward, action)
        if q is not None:
            self.Q[action] = q

        self.total_rewards += reward

        self.average_rewards.append(self.total_rewards / self.current_step)
        self.optimal_actions.append(self.plays_for_action[self.optimal_action] / self.current_step)

    def get_action(self):
        rand_for_egreedy = torch.rand(1).item()
        if rand_for_egreedy > self.epsilon:
            if self.ucb:
                temp_Q = []

                for i, q in enumerate(self.Q):
                    if self.plays_for_action[i] == 0:
                        c = 0
                    else:
                        c = self.confidence * math.sqrt(math.log(self.current_step) / self.plays_for_action[i])

                    temp_Q.append(q + c)

                action = torch.max(torch.tensor(temp_Q), 0)[1].item()
            else:
                action = torch.max(self.Q, 0)[1].item()
        else:
            action = int(torch.rand(1).item() * self.k_bandit.arm_count)

        return action

    def compute(self):
        return 0.0

    def _get_param(self, key, default=None):
        value = default
        if key in self.params:
            value = self.params[key]
        return value


class SimpleAlgorithm(Algorithm):
    def __init__(self, k_bandit, params):
        Algorithm.__init__(self, k_bandit, params)

    def compute(self, reward, action):
        q = self.Q[action] + (reward - self.Q[action]) * self.alpha

        return q


class WeightedAlgorithm(Algorithm):
    def __init__(self, k_bandit, params):
        Algorithm.__init__(self, k_bandit, params)

        self.reward_history = []

    def reset_test(self):
        Algorithm.reset_test(self)

        self.reward_history = []

    def compute(self, reward, action):
        #self.reward_history.append(reward)
        n = len(self.rewards_for_action_history[action])

        if n == 0:
            q = self.Q[action].item() + reward
            self.rewards_for_action_history[action].append(q)

        q = math.pow(1 - self.alpha, n) * self.rewards_for_action_history[action][0]

        for i in range(0, n):
            q += self.alpha * math.pow(1 - self.alpha, n - i - 1) * self.rewards_for_action_history[action][i]

        return q


class GradientAlgorithm(Algorithm):
    def __init__(self, k_bandit, params):
        Algorithm.__init__(self, k_bandit, params)

        self.r_ = 0

    def get_action(self):
        populication = [i for i in range(10)]
        weights = self._compute_softmax()

        action = random.choices(populication, weights)[0]

        return action

    def compute(self, reward, action):
        self.r_ = self.r_ + (reward - self.r_) * self.alpha

        p = self._compute_softmax()

        for i in range(self.k_bandit.arm_count):
            if i == action:
                h = self.Q[i] + self.alpha * (reward - self.r_) * (1 - p[i])
            else:
                h = self.Q[i] - self.alpha * (reward - self.r_) * p[i]

            self.Q[i] = h

        return None

    def _compute_softmax(self):
        q_exp = torch.exp(self.Q)
        sum_q_exp = torch.sum(q_exp)

        # p = [i / sum_q_exp for i in q_exp]
        return q_exp / sum_q_exp


def draw_graph(title, plots):
    plt.figure(figsize=(12, 5))

    plt.title(title)
    lines = []
    for i, plot in enumerate(plots):
        line, = plt.plot(plot["data"], alpha=0.6, color=plot["color"], label=plot["label"])
        lines.append(line)

    plt.legend(handles=lines, loc=2)

    plt.show()


def run_suite(suite):
    suite_rewards = []
    suite_actions = []

    bandits = []
    for i in range(suite["testbed"]["tests"]):
        bandits.append(KBandit(suite["bandit"]["arms"], suite["bandit"]["nonstationary"]))

    for task in suite["tasks"]:
        testbed = Testbed(num_tests=suite["testbed"]["tests"], num_steps=suite["testbed"]["steps"])
        algorithm = task["algorithm"]["class"]#(task["algorithm"]["params"])

        rewards, actions = testbed.run(bandits, algorithm, task["algorithm"]["params"])

        suite_rewards.append({
            "data": rewards.data.numpy(),
            "label": task["plot"]["label"],
            "color": task["plot"]["color"]
        })

        suite_actions.append({
            "data": actions.data.numpy(),
            "label": task["plot"]["label"],
            "color": task["plot"]["color"]
        })

    draw_graph("Average Reward", suite_rewards)
    draw_graph("Optimal Action", suite_actions)


single_suite = {
    "name": "Simple",
    "testbed": {
        "tests": 50,
        "steps": 1000
    },
    "bandit": {
        "arms": 10,
        "nonstationary": False
    },
    "tasks": [
        {
            "algorithm": {
                "class": GradientAlgorithm,
                "params": {
                    "ucb": False,
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1,
                    "confidence": 2.0
                }
            },
            "plot": {
                "label": "gradient",
                "color": "green"
            }
        }
    ]
}

simple_suite = {
    "name": "Simple",
    "testbed": {
        "tests": 50,
        "steps": 1000
    },
    "bandit": {
        "arms": 10,
        "nonstationary": False
    },
    "tasks": [
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.0,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "e=0 (greedy)",
                "color": "green"
            }
        },
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "e=0.1",
                "color": "blue"
            }
        },
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.01,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "e=0.01",
                "color": "red"
            }
        }
    ]
}

nonstationary_suite = {
    "name": "Simple",
    "testbed": {
        "tests": 20,
        "steps": 10000
    },
    "bandit": {
        "arms": 10,
        "nonstationary": True
    },
    "tasks": [
        {

            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "basic",
                "color": "red"
            }
        },
        {
            "algorithm": {
                "class": WeightedAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "weighted",
                "color": "green"
            }
        }
    ]
}

optimistic_suite = {
    "name": "Simple",
    "testbed": {
        "tests": 200,
        "steps": 1000
    },
    "bandit": {
        "arms": 10,
        "nonstationary": False
    },
    "tasks": [
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "realistic, Q1=0, e=0.1",
                "color": "grey"
            }
        },
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 5.0,
                    "epsilon": 0.0,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "optimistic, Q1=5, e=0.0",
                "color": "blue"
            }
        }
    ]
}

ucb_suite = {
    "name": "Simple",
    "testbed": {
        "tests": 2000,
        "steps": 1000
    },
    "bandit": {
        "arms": 10,
        "nonstationary": False
    },
    "tasks": [
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "ucb": True,
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1,
                    "confidence": 0.5
                }
            },
            "plot": {
                "label": "UCB, c=2",
                "color": "blue"
            }
        },
        {
            "algorithm": {
                "class": SimpleAlgorithm,
                "params": {
                    "init_value": 0.0,
                    "epsilon": 0.1,
                    "alpha": 0.1
                }
            },
            "plot": {
                "label": "e-greedy, e=0.1",
                "color": "grey"
            },
        }
    ]
}


run_suite(single_suite)

