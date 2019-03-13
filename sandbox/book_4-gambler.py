import torch
import math
import matplotlib.pyplot as plt


num_states = 100

state_policy = torch.zeros([num_states])

state_values = torch.zeros([num_states + 1])
state_values[num_states] = torch.tensor(1)

#rewards = torch.zeros([num_states + 1])
#rewards[0] = torch.tensor(-1)
#rewards[num_states] = torch.tensor(1)

gamma = 0.9
p_heads = 0.4
max_first = True


def transition(state, action):
    transition_p = torch.zeros([num_states + 1])
    rewards = torch.zeros([num_states + 1])

    win_state = state + action
    if win_state > num_states:
        win_state = torch.tensor(num_states)
        rewards[int(win_state.item())] = torch.tensor(1.0)

    lose_state = state - action
    if lose_state < 0:
        lose_state = torch.tensor(0)
        #rewards[int(win_state.item())] = torch.tensor(-1.0)

    transition_p[int(win_state.item())] = torch.tensor(p_heads)
    transition_p[int(lose_state.item())] = torch.tensor(1.0 - p_heads)

    return transition_p, rewards


def evaluate_policy():
    print("evaluating policy")

    for state, state_value in enumerate(state_values):
        if state == 0 or state > num_states-1:
            continue

        policy = state_policy[state]

        transition_p, rewards = transition(state, policy)
        new_value = torch.sum(transition_p * (rewards + gamma * state_values))

        state_values[state] = new_value

    #improve_policy()


def improve_policy():
    print("improving policy")

    policy_stable = True

    for state, state_value in enumerate(state_values):
        if state == 0 or state > num_states - 1:
            continue

        old_action = int(state_policy[state].item())

        max_action = num_states - state
        if max_action > state:
            max_action = state
        action_values = torch.zeros([max_action + 1])
        for a, action in enumerate(range(0, max_action + 1)):
            action = torch.tensor(action)
            transition_p, rewards = transition(state, action)
            #tmp = rewards + gamma * state_values
            #tmp = transition_p * tmp
            action_value = torch.sum(transition_p * (rewards + gamma * state_values))
            action_values[a] = action_value

        max_value = -1
        max_index = -1
        for i, action_value in enumerate(action_values):
            if action_value > max_value:
                if max_first and max_index < 0:
                    max_index = i
                else:
                    max_index = i

                max_value = action_value

        new_action = torch.tensor(max_index)
        #new_action = action_values.max(0)[1]

        state_policy[state] = new_action

        if old_action != new_action:
            policy_stable = False

    #if not policy_stable:
    #    print("policy not stable, reevaluating")

    return policy_stable


max_iterations = 20
iterations = 0
while True and iterations < max_iterations:
    evaluate_policy()
    stable = improve_policy()

    if stable:
        break

    iterations += 1


#plt.figure(figsize=(12, 5))

plt.title("")
lines = []

plt.plot(state_policy.data.numpy(), alpha=0.6, color="green", label="policy")
#lines.append(line)

#plt.legend(handles=lines, loc=2)

plt.show()

plt.plot(state_values.data.numpy(), alpha=0.6, color="red", label="values")
plt.show()

a = 1