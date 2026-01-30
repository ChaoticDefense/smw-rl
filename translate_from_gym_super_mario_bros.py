import gymnasium as gym

class JoypadSpaceSNES(gym.ActionWrapper):
    """
    Simplifies SNES controller to a list of allowed button combinations.
    Precomputes binary vectors for efficiency.
    """
    
    BUTTON_MAPPING = {
        'B': 0, 'Y': 1, 'SELECT': 2, 'START': 3,
        'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7,
        'A': 8, 'L': 9, 'R': 10
    }

    def __init__(self, env, combos):
        super().__init__(env)
        self.n_buttons = env.action_space.n  # MultiBinary(n)
        self.action_space = gym.spaces.Discrete(len(combos))
        
        # Precompute the full binary vectors
        self._actions = []
        for combo in combos:
            vector = np.zeros(self.n_buttons, dtype=np.int8)
            for button in combo:
                vector[self.BUTTON_MAPPING[button]] = 1
            self._actions.append(vector)

    def action(self, action):
        # Direct lookup, no loop at runtime
        return self._actions[action]