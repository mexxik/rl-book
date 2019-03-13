import math
import torch
from torch.distributions.poisson import Poisson


max_num_cars_location_1 = 20
max_num_cars_location_2 = 20
num_states = (max_num_cars_location_1 + 1)*(max_num_cars_location_2 + 1)

num_actions = 11
actions = torch.tensor([5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5])

request_location_1_dist = Poisson(torch.tensor([3.0]))
request_location_2_dist = Poisson(torch.tensor([4.0]))
return_location_1_dist = Poisson(torch.tensor([3.0]))
return_location_2_dist = Poisson(torch.tensor([2.0]))

gamma = 0.8
theta = 60

value_table = torch.zeros([num_states])
#q_table = torch.zeros([num_states, num_actions])


def get_policy_for_state(state):
    policy = torch.zeros([num_actions])# + 1 / num_actions

    cars_at_location_1 = int(state % (max_num_cars_location_1 + 1))
    cars_at_location_2 = int(state / (max_num_cars_location_2 + 1))

    action_values = torch.zeros([num_actions])
    for a, action in enumerate(actions):
        cars_at_location_1_new = cars_at_location_1 - action
        cars_at_location_2_new = cars_at_location_2 + action

        if cars_at_location_1_new > 20:
            cars_at_location_1_new = torch.tensor(20)
        #if cars_at_location_1_new < 0:
        #    cars_at_location_1_new = torch.tensor(0)

        if cars_at_location_2_new > 20:
            cars_at_location_2_new = torch.tensor(20)
        #if cars_at_location_2_new < 0:
        #    cars_at_location_2_new = torch.tensor(0)

        if cars_at_location_1_new < 0 or cars_at_location_2_new < 0:
            action_values[a] = torch.tensor(0)
        else:
            new_state = cars_at_location_1_new + \
                        cars_at_location_2_new * (max_num_cars_location_2 + 1)

            action_values[a] = value_table[new_state]

    max_state_values = torch.max(action_values)
    #max_state_values_count = torch.sum(value_table[state] == 0)
    max_state_values_count = (action_values == max_state_values).sum()

    for a, action_value in enumerate(action_values):
        if action_value == max_state_values:
            policy[a] = 1 / max_state_values_count.item()

    return policy


def get_p_s_r(state):
    new_states = torch.zeros([num_actions])
    new_states_values = torch.zeros([num_actions])
    rewards = torch.zeros([num_actions])
    #alive = torch.zeros([num_actions])

    cars_at_location_1 = int(state % (max_num_cars_location_1 + 1))
    cars_at_location_2 = int(state / (max_num_cars_location_2 + 1))

    #request_location_1 = 4
    #request_location_2 = 4
    #return_location_1 = 3
    #return_location_2 = 3

    for a, action in enumerate(actions):
        #is_out = False
        actual_action = action

        request_location_1 = int(request_location_1_dist.sample().item())
        request_location_2 = int(request_location_2_dist.sample().item())
        return_location_1 = int(return_location_1_dist.sample().item())
        return_location_2 = int(return_location_2_dist.sample().item())

        #cars_at_location_1_left = cars_at_location_1 - request_location_1
        #cars_at_location_2_left = cars_at_location_2 - request_location_2

        cars_at_location_1_rented = request_location_1
        if cars_at_location_1 - request_location_1 < 0:
            cars_at_location_1_rented = cars_at_location_1

        cars_at_location_2_rented = request_location_2
        if cars_at_location_2 - request_location_2 < 0:
            cars_at_location_2_rented = cars_at_location_2

        #if cars_at_location_1_left < 0 or cars_at_location_2_left < 0:
        #    is_out = True
        #else:

        cars_after_rent_at_location_1 = cars_at_location_1 - cars_at_location_1_rented
        cars_after_rent_at_location_2 = cars_at_location_2 - cars_at_location_2_rented

        if action > 0:
            if cars_after_rent_at_location_1 < torch.abs(action):
                actual_action = torch.tensor(0)
        elif action < 0:
            if cars_after_rent_at_location_2 < torch.abs(action):
                actual_action = torch.tensor(0)

        rewards[a] = cars_at_location_1_rented * 10 + cars_at_location_2_rented * 10 - torch.abs(actual_action) * 2
            #rewards[a] = 10 - torch.abs(action) * 2

        cars_at_location_1_new = cars_after_rent_at_location_1 + return_location_1 - actual_action
        cars_at_location_2_new = cars_after_rent_at_location_2 + return_location_2 + actual_action

        if cars_at_location_1_new > 20:
            cars_at_location_1_new = torch.tensor(20)
        if cars_at_location_1_new < 0:
            cars_at_location_1_new = torch.tensor(0)

        if cars_at_location_2_new > 20:
            cars_at_location_2_new = torch.tensor(20)
        if cars_at_location_2_new < 0:
            cars_at_location_2_new = torch.tensor(0)

        new_state = torch.abs(cars_at_location_1_new) + \
                    torch.abs(cars_at_location_2_new) * (max_num_cars_location_2 + 1)

        #if is_out:
        #    rewards[a] = torch.tensor(0)

        #    new_state = torch.tensor(0)

        new_states[a] = new_state

    for s, new_state in enumerate(new_states):
        new_states_values[s] = value_table[int(new_state.item())]

    return new_states_values, rewards


def evaluate():

    evaluation_steps = 0

    while True:
        delta = 0

        for state in range(num_states):
            #state_policy = get_policy_for_state(state)

            new_states_values, rewards = get_p_s_r(state)

            #old_max_value = torch.max(q_table[state])
            old_value = value_table[state].clone()

            value = rewards + gamma * new_states_values
            value = torch.max(value)
            value_table[state] = value

            #max_value = torch.max(q_table[state])

            value_diff = torch.abs(old_value - value)
            if value_diff > delta:
                delta = value_diff

        evaluation_steps += 1

        print("evaluated step {}, delta: {}".format(evaluation_steps, delta))

        if delta < theta or evaluation_steps > 100:
        #if evaluation_steps > 100:
            print("threshold reached, improving")
            break

    improve()


def improve():
    print("improving policy")

    policy_stable = True
    for state in range(num_states):
        state_policy = get_policy_for_state(state)
        #old_action = state_policy.max(0)[1]
        old_actions = (state_policy == state_policy.max(0)[0]).nonzero()

        new_states_values, rewards = get_p_s_r(state)
        value = state_policy * (rewards + gamma * new_states_values)

        new_action = value.max(0)[1]

        #if old_action != new_action:
        if new_action not in old_actions:
            policy_stable = False
            break

    if not policy_stable:
        print("policy is not stable, reevauating")
        evaluate()
    else:
        print("policy is stable")


def print_values():
    #action_index = torch.max(q_table, 1)[1]

    strings = []

    string = ""
    for state, value in enumerate(value_table):
        if state % (max_num_cars_location_1 + 1) == 0:
            #print(string)
            strings.append(string)
            string = ""

        #action = actions[action_index[state]]



        #max_action_values = torch.max(q_table[state])
        #max_action_count = torch.sum(q_table[state] == max_action_values)
        #if max_action_count == num_actions:
        #    action = torch.tensor(0)

        s = "{:d}".format(int(value.item()))
        string += "|"
        if len(s) < 2:
            string += " "
        if len(s) < 3:
            string += " "
        string += s

    strings.append(string)
    #print(string)

    for string in reversed(strings):
        print(string)


def print_actions():
    strings = []

    string = ""
    for state, value in enumerate(value_table):
        if state % (max_num_cars_location_1 + 1) == 0:
            # print(string)
            strings.append(string)
            string = ""

        # action = actions[action_index[state]]

        # max_action_values = torch.max(q_table[state])
        # max_action_count = torch.sum(q_table[state] == max_action_values)
        # if max_action_count == num_actions:
        #    action = torch.tensor(0)

        state_policy = get_policy_for_state(state)
        action = state_policy.max(0)[1]

        s = "{:d}".format(actions[int(action.item())])
        string += "|"
        if len(s) < 2:
            string += " "
        #if len(s) < 3:
        #    string += " "
        string += s

    strings.append(string)
    # print(string)

    for string in reversed(strings):
        print(string)


evaluate()

print_values()
print_actions()
