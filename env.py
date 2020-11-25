import numpy as np


class SartorettiEnv:
    def __init__(self, env_size=None, obstacle_density=None, num_agents=1):
        if env_size is None:
            self.env_size = np.random.choice([10, 40, 70], p=[0.5, 0.25, 0.25])
        else:
            self.env_size = env_size
        if obstacle_density is None:
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        else:
            self.obstacle_density = obstacle_density
        self.num_agents = num_agents
        self.generate_env()

    def get_obs(self):
        obs = []
        for i in range(self.num_agents):
            obs.append(dict())
            obs[i]['obstacles'] = np.ones((10, 10))
            obs[i]['agents'] = np.zeros((10, 10))
            obs[i]['targets'] = np.zeros((10, 10))
            obs[i]['goal'] = np.zeros((10, 10))
            agent_coordinates = self.coordinates_2d[self.agents == (i + 1)][0]
            target_coordinates = self.coordinates_2d[self.targets == (i + 1)][0]
            # print(agent_coordinates - 5)
            obs[i]['obstacles'] = np.copy(self.map[agent_coordinates[0] - 5:agent_coordinates[0] + 5,
                                          agent_coordinates[1] - 5:agent_coordinates[1] + 5])
            obs[i]['obstacles'][obs[i]['obstacles'] == -1] = 1
            obs[i]['agents'] = np.copy(self.agents[agent_coordinates[0] - 5:agent_coordinates[0] + 5,
                                       agent_coordinates[1] - 5:agent_coordinates[1] + 5])
            for agent in obs[i]['agents'][obs[i]['agents'] != 0]:
                neighbour_target_coordinate = self.coordinates_2d[self.targets == agent][0]
                coord = [0, 0]
                if neighbour_target_coordinate[0] < agent_coordinates[0] - 5:
                    coord[0] = 0
                elif neighbour_target_coordinate[0] > agent_coordinates[0] + 4:
                    coord[0] = -1
                else:
                    coord[0] = neighbour_target_coordinate[0] - (agent_coordinates[0] - 5)
                if neighbour_target_coordinate[1] < agent_coordinates[1] - 5:
                    coord[1] = 0
                elif neighbour_target_coordinate[1] > agent_coordinates[1] + 4:
                    coord[1] = -1
                else:
                    coord[1] = neighbour_target_coordinate[1] - (agent_coordinates[1] - 5)
                # print(coord, obs[i]['targets'])
                if i + 1 == agent:
                    obs[i]['goal'][coord[0], coord[1]] = agent
                else:
                    obs[i]['targets'][coord[0], coord[1]] = agent
            obs[i]['goal_vector'] = (target_coordinates - agent_coordinates) / np.linalg.norm((target_coordinates - agent_coordinates))
            obs[i]['goal_magnitude'] = np.linalg.norm((target_coordinates - agent_coordinates))
            obs[i]['agents'][obs[i]['agents'] != 0] = 1
            obs[i]['targets'][obs[i]['targets'] != 0] = 1
            obs[i]['goal'][obs[i]['goal'] != 0] = 1
            # obs[i]['agents'][obs[i]['agents'] != 0] = 1
            # print(obs[i]['agents'])
            # print(obs[i]['targets'])
            # print(self.agents[5:-4, 5:-4])
            # print(self.targets[5:-4, 5:-4])
            # print(obs[i]['obstacles'])
        return obs

    def step(self, actions):
        action_mapping = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
        rewards = np.zeros((self.num_agents,))

        for i in range(self.num_agents):
            agent_coordinates = self.coordinates_2d[self.agents == (i + 1)][0]
            target_coordinates = self.coordinates_2d[self.targets == (i + 1)][0]
            if actions[i] == 0:
                if np.all(agent_coordinates == target_coordinates):
                    rewards[i] = 0.0
                else:
                    rewards[i] = -0.3
            else:
                direction = action_mapping[actions[i]]
                if self.map[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] != 0 \
                        or self.agents[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] != 0:
                    rewards[i] = -2.0
                else:
                    self.agents[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] = i + 1
                    self.agents[agent_coordinates[0], agent_coordinates[1]] = 0
                    rewards[i] = -0.2
        if np.all(rewards == 0.0):
            rewards = 20.0 * np.ones((self.num_agents,))
            return self.get_obs(), rewards, True
        return self.get_obs(), rewards, False

    def reset(self):
        self.generate_env()
        return self.get_obs()

    def generate_env(self):
        self.steps = 0
        self.max_steps = 100
        self.padding_l = 5
        self.padding_r = 4
        self.map = np.zeros(
            (self.env_size + self.padding_l + self.padding_r, self.env_size + self.padding_l + self.padding_r))
        self.agents = np.zeros(
            (self.env_size + self.padding_l + self.padding_r, self.env_size + self.padding_l + self.padding_r))
        self.targets = np.zeros(
            (self.env_size + self.padding_l + self.padding_r, self.env_size + self.padding_l + self.padding_r))
        self.coordinates = np.array([[x, y] for y in range(self.env_size) \
                                     for x in range(self.env_size)]) + self.padding_l
        self.coordinates_2d = np.array([[[x, y] for y in range(self.env_size + self.padding_l + self.padding_r)] \
                                        for x in range(self.env_size + self.padding_l + self.padding_r)])
        obstacles = np.random.choice(range(len(self.coordinates)), size=int(self.env_size ** 2 * self.obstacle_density),
                                     replace=False)
        obstacles = self.coordinates[obstacles]
        self.map[obstacles[:, 0], obstacles[:, 1]] = -1
        self.map[:self.padding_l, :] = -1
        self.map[:, :self.padding_l] = -1
        self.map[-self.padding_r:, :] = -1
        self.map[:, -self.padding_r:] = -1

        self.visited = [[False for i in range(self.env_size + self.padding_l + self.padding_r)] for j in
                        range(self.env_size + self.padding_l + self.padding_r)]
        self.components = np.array([[-1 for i in range(self.env_size + self.padding_l + self.padding_r)] for j in
                                    range(self.env_size + self.padding_l + self.padding_r)])
        component_num = 0
        for i in range(self.env_size + self.padding_l + self.padding_r):
            for j in range(self.env_size + self.padding_l + self.padding_r):
                if not self.visited[i][j] and self.map[i][j] == 0:
                    self.dfs(i, j, component_num)
                    component_num += 1

        for agent in range(self.num_agents):
            agent_coords = [0, 0]
            target_coords = [0, 0]
            while True:
                agent_coords = self.coordinates_2d[self.map == 0][
                    np.random.choice(range(len(self.coordinates_2d[self.map == 0])))]
                target_coords = self.coordinates_2d[self.components == \
                                                    self.components[agent_coords[0], agent_coords[1]]][
                    np.random.choice(range(len(self.coordinates_2d[self.components == \
                                                                   self.components[
                                                                       agent_coords[0], agent_coords[1]]])))]
                if (agent_coords[0] != target_coords[0] or agent_coords[1] != target_coords[1]) and \
                        self.agents[agent_coords[0], agent_coords[1]] == 0 and \
                        self.targets[target_coords[0], target_coords[1]] == 0:
                    break
            self.agents[agent_coords[0], agent_coords[1]] = agent + 1
            self.targets[target_coords[0], target_coords[1]] = agent + 1
        # self.get_obs()

    def dfs(self, x, y, component):
        self.visited[x][y] = True
        self.components[x][y] = component
        if x > 0:
            if not self.visited[x - 1][y] and self.map[x - 1][y] == 0:
                self.dfs(x - 1, y, component)
        if x < self.env_size + self.padding_l + self.padding_r - 1:
            if not self.visited[x + 1][y] and self.map[x + 1][y] == 0:
                self.dfs(x + 1, y, component)
        if y > 0:
            if not self.visited[x][y - 1] and self.map[x][y - 1] == 0:
                self.dfs(x, y - 1, component)
        if y < self.env_size + self.padding_l + self.padding_r - 1:
            if not self.visited[x][y + 1] and self.map[x][y + 1] == 0:
                self.dfs(x, y + 1, component)
