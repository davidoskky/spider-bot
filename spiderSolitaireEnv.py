import gymnasium as gym
from gymnasium import spaces
import numpy as np
from spiderSolitaire import SpiderSolitaire, SimpleSpiderSolitaire

NUM_POSSIBLE_ACTIONS = 92  # 90 pile to pile + 1 drawing from deck


class SpiderSolitaireEnv(gym.Env):
    def __init__(self):
        super(SpiderSolitaireEnv, self).__init__()
        self.max_steps = 1000  # Maximum number of steps before truncating the episode
        self.current_step = 0

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(NUM_POSSIBLE_ACTIONS)

        self.observation_space = spaces.Box(
            low=-1, high=52, shape=(251,), dtype=np.int32
        )

        # Initialize your game environment here
        self.game = SimpleSpiderSolitaire()

    def reset(self, seed=None):
        # Reset the game environment and return the initial state
        self.game = SimpleSpiderSolitaire(seed)
        self.current_step = 0
        return self.get_state(), {}

    def execute_action(self, action):
        """Executes the given action in the Spider Solitaire game."""
        if action >= 0 and action < 90:
            from_pile, to_pile = self._decode_action(action)
            valid_action = self.game.move_pile_to_pile(from_pile, to_pile)
        elif action == 90:
            # Execute the stock dealing action
            valid_action = self.game.draw_from_deck()
        else:
            valid_action = False

        return valid_action

    def _decode_action(self, action):
        """Decodes the given action number into a from_pile and to_pile tuple."""
        if action < 90:
            from_pile = action // 9
            to_pile = action % 9
            if to_pile >= from_pile:
                to_pile += 1
            return from_pile, to_pile
        else:
            raise ValueError("Invalid action: {}".format(action))

    def step(self, action):
        self.current_step += 1
        info = {}  # Initialize an empty info dictionary
        reward = -1
        truncated = False

        valid_action = self.execute_action(action)

        if action == 91:  # Reset game
            print("Game reset by the bot.")
            reward = -500
            info["game_interrupted"] = True
            done = True
            truncated = True
            return self.get_state(), reward, done, truncated, info

        next_state = self.get_state()
        if not valid_action:
            reward = -3
        elif self.game.just_completed_stack:
            reward = 500
        done = self.game.is_game_over()
        if done:
            if self.game.is_game_won():
                reward = 5000
                info["game_won"] = True
            else:
                truncated = True
                info["episode_failed"] = True
        if self.current_step >= self.max_steps:
            done = True  # End the episode due to too many steps
            truncated = True
            reward = -1200
        return next_state, reward, done, truncated, info

    def render(self, mode="human"):
        self.game.display_game_state()

    def close(self):
        # Implement any cleanup or resource release logic (optional)
        pass

    def get_state(self):
        """Return the current game state as a flattened array with a fixed size."""
        max_stack_size = 25  # Define the target stack size
        num_stacks = 10  # Number of stacks in Spider Solitaire
        total_size = max_stack_size * num_stacks + 1

        state = []
        state.append(len(self.game.deck.cards))

        # Flatten the stacks into a single list
        for stack in self.game.stacks:
            stack_representation = [card.encode() for card in stack.cards]
            padded_stack = stack_representation + [-1] * (
                max_stack_size - len(stack_representation)
            )
            state.extend(padded_stack)

        # Pad the entire state to make it of the total_size
        state += [-1] * (total_size - len(state))

        return np.array(state)
