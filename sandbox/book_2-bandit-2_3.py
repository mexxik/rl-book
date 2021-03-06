import torch
import progressbar
import matplotlib.pyplot as plt


'''
basic testbed k-armed bandit implementation
'''


class KBandit(object):
    def __init__(self, arm_count):
        #print("creating {}-armed bandit".format(arm_count))

        self.arm_count = arm_count

        values_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.arm_values = []
        values_string = ""
        for i in range(0, self.arm_count):
            value = values_dist.sample().item()

            self.arm_values.append(value)

            values_string += "[{0:.2f}]".format(value)

        #print(values_string)

    def play(self, arm):
        mean = torch.tensor([self.arm_values[arm]])
        deviation = torch.tensor([1.0])

        reward_dist = torch.distributions.normal.Normal(mean, deviation)

        return reward_dist.sample().item()

    def optimal_action(self):
        return torch.max(torch.tensor(self.arm_values), 0)[1].item()


gamma = 1

num_steps = 1000
num_tests = 2000


def run_testbed(epsilon):
    total_average_rewards = []
    total_optimal_actions = []
    with progressbar.ProgressBar(max_value=num_tests) as bar:
        for test in range(0, num_tests):
            k_bandit = KBandit(10)
            optimal_action = k_bandit.optimal_action()

            Q = torch.zeros(k_bandit.arm_count) + torch.rand(k_bandit.arm_count) / 1000000
            Q_total = torch.zeros(k_bandit.arm_count)
            Q_taken = torch.zeros(k_bandit.arm_count)

            total_rewards = 0
            average_rewards = []
            optimal_actions = []
            for step in range(0, num_steps):
                rand_for_egreedy = torch.rand(1).item()
                if rand_for_egreedy > epsilon:
                    action = torch.max(Q, 0)[1].item()
                else:
                    action = int(torch.rand(1).item() * 10)

                reward = k_bandit.play(action)

                Q_total[action] += reward
                Q_taken[action] += 1

                q = Q_total[action] / Q_taken[action]
                Q[action] = q

                total_rewards += reward

                average_rewards.append(total_rewards / (step + 1))

                optimal_actions.append(Q_taken[optimal_action] / (step + 1))

            total_average_rewards.append(average_rewards)
            total_optimal_actions.append(optimal_actions)


            #total_optimal_actions.append(optimal_actions)

            #print("average rewards for test {}: {}".format(1, total_rewards / num_steps))

            bar.update(test)

    total_average_rewards = torch.tensor(total_average_rewards)
    total_optimal_actions = torch.tensor(total_optimal_actions)

    total_average_rewards = torch.mean(total_average_rewards, 0)
    total_optimal_actions = torch.mean(total_optimal_actions, 0)

    return total_average_rewards, total_optimal_actions


rewards_0, action_0 = run_testbed(0.0)
rewards_0_1, action_0_1 = run_testbed(0.1)
rewards_0_0_1, action_0_0_1 = run_testbed(0.01)

#m = torch.mean(total_average_rewards, 0)

plt.figure(figsize=(12, 5))

plt.title("Average Reward")
plt.plot(rewards_0.data.numpy(), alpha=0.6, color='red')
plt.plot(rewards_0_1.data.numpy(), alpha=0.6, color='green')
plt.plot(rewards_0_0_1.data.numpy(), alpha=0.6, color='blue')
plt.show()

plt.title("Optimal Action")
plt.plot(action_0.data.numpy(), alpha=0.6, color='red')
plt.plot(action_0_1.data.numpy(), alpha=0.6, color='green')
plt.plot(action_0_0_1.data.numpy(), alpha=0.6, color='blue')
plt.show()

print("finished")
