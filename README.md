#1 Introduction# 

This project applies Deep Q-Learning (DQN) to an arcade-style Air Plane War game with both single-agent (Boss) and multi-agent (Boss + Enemies) training. Players face wave-based, AI-controlled enemies followed by a boss fight. Agents observe a compact, normalized game state—including positions, health values, nearby bullets, and timing—and select discrete actions (move and shoot) to maximize cumulative reward. In the multi-agent setup, the boss is controlled by its own DQN, while enemies share a policy that outputs actions per enemy instance.
The technology stack uses Python for core logic, PyTorch to build and train DQN (tensor ops, neural modules, optimizers), and Pygame for graphics, audio, input, and the game loop. The environment supports headless training for speed and safe asset handling without a display, while the play mode provides an interactive UI (Start, Pause, Restart). Training progress is printed per episode and logged to training_log.txt; completion metadata is written to training_done.txt.
Models are periodically saved during training and loaded in play to enable AI-driven behavior. If model files are missing, the game falls back to randomized policies so it remains playable. Through DQN integrated with the normalized state, agents learn policies that improve survivability, scoring, and phase progression within the Air Plane War environment.
 #2 Game Design#
 
 2.1 Rules of the Game
• Goal: Survive and score points; eliminate all the enemies and Bosses controlled by AI. 
• Controls: Arrow keys to move; Space to shoot.
• Player: Moves within screen bounds; shooting has cooldown; Game Over when health runs out. 
• Enemies: Spawn in waves, drift continuously, and fire bullets. 
• Boss: High health and diverse bullet patterns; moves horizontally and performs scatter attacks. 
• Bullets: Player bullets fly upward, enemy bullets fly downward; automatically disappear when out of bounds. 
• Scoring: Points awarded for destroying enemies and dealing damage to the Boss; high scores unlock stronger scatter attacks, When the score ≥0, it is a single shot ; when the score ≥20, it upgrades to double shot spread; when the score ≥50, it upgrades to triple shot spread; when the score is ≥80, it upgrades to five-shot spread .
2.2 Class Design of the Game
PlaneWar [plane_war.py]
  - Manages game loop, waves, boss phase, score, lives, sprites
  - APIs: reset, handle_events, render_frame, get_state/get_enemy_states, step
Player [entities.py]
  - rect, hp, speed, shoot_delay
  -update(keys), shoot(offsets), take_damage; bounded movement, cooldown shooting
Enemy [entities.py]
  - rect, speed, vx/vy drift, shoot_delay, max_y
  - update(), move_action(action), shoot(...); drift+bounce to avoid stalling
Boss [entities.py]
  - rect, hp, speed, last_shot, pattern, vx
  - move(action), shoot(), take_damage; horizontal motion and spread shots
Bullet [entities.py]
  - rect, speed_x, speed_y, source=player/boss/enemy
  - update(); off-screen cleanup and source-based attribution
Agent (DQN) [agent.py]
  - policy_net, target_net, optimizer, replay memory, epsilon
  - act, step, learn, save/load; epsilon-greedy with target sync
DQN Network / ReplayBuffer [dqn/network.py, dqn/replay_buffer.py]
  - MLP: state → Q-values; fixed-size FIFO with batch sampling
Config [game/config.py]
  - Screen sizes, speeds, sprite sizes, rewards, state/action dimensions
Entry Points [main.py]
  - play: UI states, model loading and fallbacks
  - train: single-agent (Boss) training
  - train_multi: Boss + Enemies training with per-episode logging
2.3 UI Design
UI design follows the principles of "clarity, lightness, and readability": The start interface centrally displays the title and concise operation prompts (arrow keys to move, Space to shoot), and provides a Start button to enter the game; in the running state, Pause and Restart are retained in the upper right corner, with buttons featuring strikingly contrasting strokes and hover feedback to ensure quick accessibility; the pause state uses a semi-transparent mask and the text "Paused" to enhance the sense of state, and it can still be resumed or restarted with one click; Game Over retains the result prompt and Restart to reduce the burden of secondary operations. Fonts prioritize matching system sans-serif Chinese/English fonts, with distinct font size hierarchies, and colors follow high contrast and consistency.
