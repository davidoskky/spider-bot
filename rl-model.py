import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from spiderSolitaireEnv import SpiderSolitaireEnv

# Create and wrap the environment
env = SpiderSolitaireEnv()

# Define the model
#model = PPO(
#  "MlpPolicy",
#  env,
#  verbose=1,
#  #ent_coef=0.001,
#  #learning_rate=0.0004,
#  policy_kwargs=dict(net_arch=[128]*25),
#  tensorboard_log="./tb_log/",
#)
model = PPO.load("ppo_spidersolitaire-2")
model.set_env(env)
# model.learning_rate = 0.002
# model.learning_rate = 0.05

# Train the model
model = model.learn(total_timesteps=50000, progress_bar=True)

# Save the model
model.save("ppo_spidersolitaire-2")


print("Initializing the test")

obs, _ = env.reset()
for i in range(2000):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render("human")

    if done:
        break

num_episodes = 100
num_wins = 0
num_interrupted = 0

for episode in range(num_episodes):
    obs, _ = env.reset(None)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print(f"One more {episode}")
            if "game_interrupted" in info and info["game_interrupted"]:
                num_interrupted += 1
        if done and "game_won" in info and info["game_won"]:
            num_wins += 1

print(
    f"Out of {num_episodes} episodes, the model won {num_wins} times and interrupted {num_interrupted}."
)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=False
)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
