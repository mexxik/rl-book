import torch
import progressbar
import time


Tensor = torch.Tensor

grid_width = 9
grid_height = 6

actions_count = 4
actions_mapping = Tensor([[-1, 0], [1, 0], [0, 1], [0, -1]])

Q = torch.zeros([grid_height, grid_width, actions_count], dtype=torch.float)
model = torch.zeros([grid_height, grid_width, actions_count], dtype=torch.float)

start = (2, 0)
goal = (0, 8)

walls = [
    (1, 2), (2, 2), (3, 2),
    (4, 5),
    (0, 7), (1, 7), (2, 7)
]

alpha = 0.1
gamma = 0.95
epsilon = 0.1


def draw(mode="empty"):
    for col in range(grid_height + 1):
        string_1 = ""
        string_2 = ""
        for row in range(grid_width):
            string_1 += "+-----"
            string_2 += "|"

            cell = (col, row)
            if cell == start:
                string_2 += "Start"
            elif cell == goal:
                string_2 += "Goal "
            elif cell in walls:
                string_2 += "XXXXX"
            else:
                if mode == "empty":
                    string_2 += "  "
                elif mode == "values":
                    if col < grid_height:
                        value = torch.max(Q[cell[0], cell[1]]).item()
                        value = "{:1.3f}".format(float(value))
                        string_2 += value
                elif mode == "actions":
                    if col < grid_height:
                        action = torch.max(Q[cell[0], cell[1]], 0)[1].item()
                        string_2 += "  "
                        if action == 0:
                            string_2 += "^"
                        elif action == 1:
                            string_2 += "v"
                        elif action == 2:
                            string_2 += ">"
                        elif action == 3:
                            string_2 += "<"
                        string_2 += "  "


            #string_2 += ""

        string_1 += "+"

        print(string_1)
        if col < grid_height:
            string_2 += "|"
            print(string_2)


max_episodes = 50
with progressbar.ProgressBar(max_value=max_episodes) as bar:
    for e in range(max_episodes):
        bar.update(e)

        state = start

        while True:
            rand_for_egreedy = torch.rand(1).item()
            if rand_for_egreedy > epsilon:
                random_values = Q[state[0]][state[1]] + torch.rand(1, actions_count) / 1000

                action = torch.max(random_values, 1)[1][0].item()
            else:
                action = torch.LongTensor(1).random_(0, actions_count).item()

            x_step = int(actions_mapping[action][1].item())
            y_step = int(actions_mapping[action][0].item())
            new_state = (state[0] + y_step, state[1] + x_step)

            if new_state[0] < 0:
                new_state = (0, new_state[1])
            if new_state[0] >= grid_height:
                new_state = (grid_height - 1, new_state[1])
            if new_state[1] < 0:
                new_state = (new_state[0], 0)
            if new_state[1] >= grid_width:
                new_state = (new_state[0], grid_width - 1)

            if new_state in walls:
                new_state = state

            if new_state == goal:
                reward = 1
            else:
                reward = 0

            q_s_a = Q[state[0], state[1], action]

            q_s_new = Q[new_state[0], new_state[1]] + torch.rand(1, actions_count) / 1000000
            q_s_a_new = torch.max(q_s_new, 1)[0][0].item()

            q = q_s_a + alpha * (reward + gamma * q_s_a_new - q_s_a)
            Q[state[0], state[1], action] = q

            if reward == 1:
                break

            state = new_state



time.sleep(0.01)
draw("values")
print("\n")
draw("actions")