
import sys
import pygame
import torch
import numpy as np
from game.plane_war import PlaneWar
from dqn.agent import Agent
from game.config import *
import os

def train():
    env = PlaneWar(render=False)
    agent = Agent(STATE_SIZE, NUM_ACTIONS)
    
    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Only handle events if rendering, or if we want to allow Ctrl+C to be caught by pygame?
            # If render=False, we don't have a window, so no events.
            if env.render:
                 if not env.handle_events():
                     return
            
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if env.render:
                env.render_frame()
        
        print(f"Episode {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        if (e+1) % 50 == 0:
            agent.save(f"dqn_model_{e+1}.pth")
            print(f"Model saved to dqn_model_{e+1}.pth")

    print("Training finished.")

def train_multi():
    env = PlaneWar(render=False, human_player=True)
    boss_agent = Agent(STATE_SIZE, NUM_ACTIONS)
    enemy_agent = Agent(ENEMY_STATE_SIZE, ENEMY_NUM_ACTIONS)
    episodes = 300
    print("Starting multi-agent training...", flush=True)
    log_path = os.path.join(os.getcwd(), "training_log.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[start] episodes={episodes}\n")
    except Exception:
        print("WARN: cannot open training_log.txt for writing", flush=True)
    for e in range(episodes):
        env.reset()
        env.boss_active = True
        if env.boss not in env.all_sprites:
            env.all_sprites.add(env.boss)
        done = False
        total_reward_boss = 0.0
        total_reward_enemy = 0.0
        step_count = 0
        while not done:
            if env.render:
                if not env.handle_events():
                    return
            boss_state = env.get_state()
            enemy_states = env.get_enemy_states()
            boss_action = boss_agent.act(boss_state)
            enemy_actions = [enemy_agent.act(s) for s in enemy_states]
            next_state, r_boss, done, info = env.step(None, {'boss': boss_action, 'enemies': enemy_actions})
            boss_agent.step(boss_state, boss_action, r_boss, next_state, done)
            next_enemy_states = info.get('enemy_states', enemy_states)
            r_enemy = info.get('enemy_reward', 0.0)
            for s, a, ns in zip(enemy_states, enemy_actions, next_enemy_states):
                enemy_agent.step(s, a, r_enemy, ns, done)
            total_reward_boss += r_boss
            total_reward_enemy += r_enemy
            if env.render:
                env.render_frame()
            step_count += 1
        line = f"Episode {e+1}/{episodes}, BossReward: {total_reward_boss:.2f}, EnemyReward: {total_reward_enemy:.2f}, eps(B):{boss_agent.epsilon:.2f}, eps(E):{enemy_agent.epsilon:.2f}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            print("WARN: cannot append training_log.txt", flush=True)
        if (e+1) % 50 == 0:
            boss_agent.save(f"boss_dqn_{e+1}.pth")
            enemy_agent.save(f"enemy_dqn_{e+1}.pth")
    boss_agent.save("boss_dqn_final.pth")
    enemy_agent.save("enemy_dqn_final.pth")
    print("Multi-agent training finished.")
def train_player():
    env = PlaneWar(render=False, human_player=True)
    player_agent = Agent(STATE_SIZE, NUM_ACTIONS)
    episodes = 100
    log_path = os.path.join(os.getcwd(), "training_player_log.txt")
    def safe_write(p, s):
        for _ in range(3):
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(s)
                    f.flush()
                return True
            except Exception:
                import time
                time.sleep(0.05)
        try:
            with open(os.path.join(os.getcwd(), "training_player_log_backup.txt"), "a", encoding="utf-8") as f:
                f.write(s)
        except Exception:
            pass
        return False
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[start] episodes={episodes}\n")
    except Exception:
        safe_write(log_path, f"[start] episodes={episodes}\n")
    for e in range(episodes):
        env.reset()
        done = False
        total_reward_player = 0.0
        safe_write(log_path, f"[episode] {e+1} start\n")
        steps = 0
        while not done:
            if env.render:
                if not env.handle_events():
                    return
            state = env.get_player_state()
            player_action = player_agent.act(state)
            next_state, r, done, info = env.step(None, {'player': player_action})
            pr = info.get('player_reward', r)
            player_agent.step(state, player_action, pr, next_state, done)
            total_reward_player += pr
            if env.render:
                env.render_frame()
            steps += 1
            if steps % 200 == 0:
                safe_write(log_path, f"[progress] episode={e+1} steps={steps} reward_acc={total_reward_player:.2f}\n")
            if steps >= 4000:
                break
        line = f"Episode {e+1}/{episodes}, PlayerReward: {total_reward_player:.2f}, eps(P):{player_agent.epsilon:.2f}"
        print(line, flush=True)
        if not safe_write(log_path, line + "\n"):
            print("WARN: failed to append training_player_log.txt", flush=True)
        if (e+1) % 50 == 0:
            player_agent.save(f"player_dqn_{e+1}.pth")
    player_agent.save("player_dqn_final.pth")
    print("Player training finished.")
def play():
    env = PlaneWar(render=True, human_player=True)
    boss_agent = Agent(STATE_SIZE, NUM_ACTIONS)
    enemy_agent = Agent(ENEMY_STATE_SIZE, ENEMY_NUM_ACTIONS)
    
    boss_model_path = "boss_dqn_final.pth"
    enemy_model_path = "enemy_dqn_final.pth"
    if os.path.exists(boss_model_path):
        boss_agent.load(boss_model_path)
        print(f"Loaded boss model from {boss_model_path}")
        boss_agent.epsilon = 0.0
    else:
        print("No boss model found, using random boss agent.")
    if os.path.exists(enemy_model_path):
        try:
            enemy_agent.load(enemy_model_path)
            print(f"Loaded enemy model from {enemy_model_path}")
            enemy_agent.epsilon = 0.0
        except Exception as e:
            print(f"Failed to load enemy model from {enemy_model_path}: {e}")
            print("Using random enemy agent due to state size mismatch.")
    else:
        print("No enemy model found, using random enemy agent.")
    
    start_button = pygame.Rect(SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 + 40, 160, 50)
    pause_button = pygame.Rect(SCREEN_WIDTH - 160, 10, 140, 40)
    restart_button = pygame.Rect(SCREEN_WIDTH - 160, 60, 140, 40)
    title_font = pygame.font.Font(None, 48)
    font_path = None
    for name in ["simhei", "msyh", "microsoft yahei", "microsoft yahei ui", "noto sans sc", "arial unicode ms"]:
        p = pygame.font.match_font(name)
        if p:
            font_path = p
            break
    ui_font = pygame.font.Font(font_path, 28) if font_path else pygame.font.Font(None, 28)
    state = None
    total_reward = 0
    mode = "start"

    def draw_button(screen, font, rect, text):
        pygame.draw.rect(screen, (40, 40, 40), rect, border_radius=6)
        pygame.draw.rect(screen, WHITE, rect, 2, border_radius=6)
        label = font.render(text, True, WHITE)
        screen.blit(label, label.get_rect(center=rect.center))

    def ui_draw(screen, font):
        if mode == "start":
            title = title_font.render("Air Plane War", True, WHITE)
            screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)))
            hint = ui_font.render("Press SPACE to shoot. Defeat the enemies.", True, WHITE)
            screen.blit(hint, hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60)))
            move = ui_font.render("Use Arrow Keys to move (↑ ↓ ← →)", True, WHITE)
            screen.blit(move, move.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)))
            draw_button(screen, font, start_button, "Start")
        elif mode == "running":
            draw_button(screen, font, pause_button, "Pause")
            draw_button(screen, font, restart_button, "Restart")
        elif mode == "paused":
            draw_button(screen, font, pause_button, "Resume")
            draw_button(screen, font, restart_button, "Restart")
            label = title_font.render("Paused", True, WHITE)
            screen.blit(label, label.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)))
        elif mode == "gameover":
            draw_button(screen, font, restart_button, "Restart")
            label = title_font.render("Game Over", True, WHITE)
            screen.blit(label, label.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mode == "start" and start_button.collidepoint(event.pos):
                    state = env.reset()
                    total_reward = 0
                    mode = "running"
                elif mode in ("running", "paused") and pause_button.collidepoint(event.pos):
                    mode = "paused" if mode == "running" else "running"
                if mode in ("running", "paused", "gameover") and restart_button.collidepoint(event.pos):
                    state = env.reset()
                    total_reward = 0
                    mode = "running"

        if mode == "running":
            boss_action = None
            if env.boss_active:
                boss_action = boss_agent.act(state)
                if boss_action in (ACTION_SHOOT, ACTION_SPECIAL):
                    boss_action = None
            enemy_actions = []
            enemy_list = list(env.enemies)
            if len(enemy_list) > 0:
                enemy_states = env.get_enemy_states()
                for enemy, s in zip(enemy_list, enemy_states):
                    a = enemy_agent.act(s)
                    dodge = env.get_enemy_dodge_action(enemy)
                    if dodge is not None:
                        a = dodge
                    enemy_actions.append(a)
            actions = {'boss': boss_action, 'enemies': enemy_actions}
            state, reward, done, _ = env.step(None, actions)
            total_reward += reward
            if done:
                mode = "gameover"

        env.render_frame(ui_draw=ui_draw)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "train_multi":
            train_multi()
        elif sys.argv[1] == "train_player":
            train_player()
    else:
        play()
