import torch


Tensor = torch.Tensor


class GridWorld(object):
    def __init__(self, desc):
        self.desc = desc
        self.positions = []
        self.actions = Tensor([
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1]
        ])

        for item in self.desc["grid"]:
            self.positions.append(item["position"])

        self.positions = Tensor(self.positions)

        self.max_positions = torch.max(self.positions, 0)[0]
        self.max_width = int(self.max_positions[1]) + 1
        self.max_height = int(self.max_positions[0]) + 1

        self.reward_grid = torch.zeros([self.max_height, self.max_width], dtype=torch.int32)
        self.state_values = torch.zeros([self.max_height, self.max_width], dtype=torch.float)

        for item in desc["grid"]:
            reward = item["reward"]
            x_pos = item["position"][1]
            y_pos = item["position"][0]

            self.reward_grid[y_pos, x_pos] = reward

        a = 1

    def get_pi_s(self, state_index):
        state_x = int(state_index % self.max_width)
        state_y = int(state_index / self.max_height)
        state_value = self.state_values[state_y, state_x]

        pi_s = torch.zeros([4])
        new_state_values = []

        for action in self.actions:
            new_state_x = int(state_x + action[1].item())
            new_state_y = int(state_y + action[0].item())

            if new_state_x < 0:
                new_state_x = state_x

            if new_state_y < 0:
                new_state_y = state_y

            new_state_values.append(self.state_values[new_state_x, new_state_y])

        new_state_values = Tensor(new_state_values)

        a = 1



class Algorithm(object):
    def __init__(self, theta):
        self.theta = theta

    def run(self, grid_world):
        for i in range(1, 15):
            pi_s = grid_world.get_pi_s(i)


basic_desc = {
    "actions": [
        {
            "name": "up",
            "matrix": [-1, 0]
        },
        {
            "name": "down",
            "matrix": [1, 0]
        },
        {
            "name": "right",
            "matrix": [0, 1]
        },
        {
            "name": "left",
            "matrix": [0, -1]
        }
    ],
    "grid": [
        {"index": 0,  "terminal": True,  "reward": 0,  "position": [0, 0]},
        {"index": 1,  "terminal": False, "reward": -1, "position": [0, 1]},
        {"index": 2,  "terminal": False, "reward": -1, "position": [0, 2]},
        {"index": 3,  "terminal": False, "reward": -1, "position": [0, 3]},
        {"index": 4,  "terminal": False, "reward": -1, "position": [1, 0]},
        {"index": 5,  "terminal": False, "reward": -1, "position": [1, 1]},
        {"index": 6,  "terminal": False, "reward": -1, "position": [1, 2]},
        {"index": 7,  "terminal": False, "reward": -1, "position": [1, 3]},
        {"index": 8,  "terminal": False, "reward": -1, "position": [2, 0]},
        {"index": 9,  "terminal": False, "reward": -1, "position": [2, 1]},
        {"index": 10, "terminal": False, "reward": -1, "position": [2, 2]},
        {"index": 11, "terminal": False, "reward": -1, "position": [2, 3]},
        {"index": 12, "terminal": False, "reward": -1, "position": [3, 0]},
        {"index": 13, "terminal": False, "reward": -1, "position": [3, 1]},
        {"index": 14, "terminal": False, "reward": -1, "position": [3, 2]},
        {"index": 15, "terminal": True,  "reward": 0,  "position": [3, 3]}
    ]
}

grid_world = GridWorld(basic_desc)
algorithm = Algorithm(5)

algorithm.run(grid_world)