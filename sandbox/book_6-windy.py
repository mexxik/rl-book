import torch
import progressbar
import time


Tensor = torch.Tensor

grid_width = 10
grid_height = 7

actions_count = 4
actions_mapping = Tensor([[-1, 0], [1, 0], [0, 1], [0, -1]])

Q = torch.zeros([grid_height, grid_width, actions_count], dtype=torch.float)


epsilon = 0.1
alpha = 0.5
gamma = 1.0

max_episodes = 1000
episodes_processed = 0


def reset():
    state = (3, 0)
    return state


def get_action(state):
    rand_for_egreedy = torch.rand(1).item()
    if rand_for_egreedy > epsilon:
        random_values = Q[state[0]][state[1]] + torch.rand(1, actions_count) / 1000

        action = torch.max(random_values, 1)[1][0].item()
    else:
        action = torch.LongTensor(1).random_(0, actions_count).item()



    #result = torch.argmax(Q[state[0]][state[1]])

    return action


def step(state, action):
    x_step = int(actions_mapping[action][1].item())
    y_step = int(actions_mapping[action][0].item())
    new_state = (state[0] + y_step, state[1] + x_step)

    if state[1] > 2 and state[1] < 9:
        new_state = (new_state[0] - 1, new_state[1])

        if state[1] > 5 and state[1] < 8:
            new_state = (new_state[0] - 1, new_state[1])

    if new_state[0] < 0:
        new_state = (0, new_state[1])
    if new_state[0] >= grid_height:
        new_state = (grid_height - 1, new_state[1])
    if new_state[1] < 0:
        new_state = (new_state[0], 0)
    if new_state[1] >= grid_width:
        new_state = (new_state[0], grid_width - 1)

    reward = 0
    done = False

    if new_state == (3, 7):
        reward = 1
        done = True

    return new_state, reward, done


reset()

with progressbar.ProgressBar(max_value=max_episodes) as bar:
    while episodes_processed < max_episodes:
        current_state = reset()
        current_step = 0

        while True:
            current_step += 1

            action = get_action(current_state)

            new_state, reward, done = step(current_state, action)

            new_action = get_action(new_state)

            q_s_a = Q[current_state[0], current_state[1], action]
            if done:
                q_s_a_new = 0
            else:
                q_s_a_new = Q[new_state[0], new_state[1], new_action]

            #q = q_s_a + alpha * (reward - current_step + gamma * q_s_a_new - q_s_a)
            q = q_s_a + alpha * (reward + gamma * q_s_a_new - q_s_a)

            Q[current_state[0], current_state[1], action] = q

            current_state = new_state

            if done:
                #Q[current_state[0], current_state[1], :] = 0

                break

        episodes_processed += 1

        bar.update(episodes_processed)


time.sleep(0.01)


route = []
route_step = (3, 0)
route.append(route_step)
steps = 0
while True:
    random_values = Q[route_step[0]][route_step[1]] + torch.rand(1, actions_count) / 1000

    action = torch.max(random_values, 1)[1][0].item()
    action = actions_mapping[action]

    route_step = (route_step[0] + int(action[0].item()), route_step[1] + int(action[1].item()))

    route.append(route_step)

    steps += 1

    if route_step == (3, 7) or steps > 20:
        break

#best_values = torch.max(Q, dim=2)[0]

strings = []
for y in range(grid_height):
    string = ""

    for x in range(grid_width):

        string += "| "

        if (y, x) in route:
            string += "x "
        else:
            string += "  "
        #string += " {:2.2f} ".format(float(value.item()))

    string += "|"

    strings.append(string)

for s in strings:
    print(s)
print("\n")

a = 1