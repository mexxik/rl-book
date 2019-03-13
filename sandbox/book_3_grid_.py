import torch


Tensor = torch.Tensor


actions_mapping = Tensor([[-1, 0], [1, 0], [0, 1], [0, -1]])

reward_grid = torch.zeros([4, 4], dtype=torch.int32) - 1
reward_grid[0, 0] = 0
reward_grid[3, 3] = 0
state_values = torch.zeros([4, 4], dtype=torch.float)


steps = 0

while True:
    delta = 0

    for i in range(1, 15):
        state_x = int(i % 4)
        state_y = int(i / 4)

        pi_s = torch.zeros([4])
        new_state_values = []
        rewards = []

        for action in actions_mapping:
            new_state_x = int(state_x + action[1].item())
            new_state_y = int(state_y + action[0].item())

            if new_state_x < 0:
                new_state_x = state_x

            if new_state_x > 3:
                new_state_x = state_x

            if new_state_y < 0:
                new_state_y = state_y

            if new_state_y > 3:
                new_state_y = state_y

            new_state_values.append(state_values[new_state_y, new_state_x])
            rewards.append(reward_grid[new_state_y, new_state_x])

        new_state_values = Tensor(new_state_values)
        rewards = Tensor(rewards)

        policy_for_a = torch.zeros([4]) + 0.25

        #max_state_values = torch.max(new_state_values)
        #max_state_values_count = torch.sum(new_state_values == max_state_values)

        #for p, policy in enumerate(policy_for_a):
        #    if new_state_values[p] == max_state_values:
        #        policy_for_a[p] = 1 / max_state_values_count.item()

        old_value = state_values[state_y, state_x].clone()

        state_value = torch.sum(policy_for_a * (rewards + new_state_values))

        state_values[state_y, state_x] = state_value

        value_diff = old_value - state_value

        if value_diff > delta:
            delta = value_diff

    print(state_values)
    print("delta: {}".format(delta))

    steps += 1
    if delta < 0.5:
        break


