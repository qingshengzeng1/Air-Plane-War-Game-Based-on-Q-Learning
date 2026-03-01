
import os
import random
import pygame
import numpy as np
from game.config import *
from game.entities import Player, Boss, Bullet, Enemy

class PlaneWar:
    def __init__(self, render=True, human_player=False):
        pygame.init()
        self.render = render
        self.human_player = human_player
        self.training_player = (not render and human_player)
        self.max_steps_episode = 4000 if self.training_player else None
        self.background = None
        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption(CAPTION)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
            bg_path = os.path.join(asset_dir, "background.png")
            if os.path.exists(bg_path):
                self.background = pygame.image.load(bg_path).convert()
                self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.reset()

    def reset(self):
        self.player = Player(*PLAYER_START_POS)
        if self.training_player:
            self.player.shoot_delay = 150
        self.boss = Boss(*BOSS_START_POS)
        self.bullets = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)
        
        self.score = 0
        self.steps = 0
        self.done = False
        self.wave = 1
        self.round_level = 1
        self.phase = "boss" if not self.human_player else "enemies"
        self.waste_bullets_count = 0
        self.enemy_spawned = 0
        self.last_spawn = 0
        self.boss_active = self.phase == "boss"
        if self.boss_active:
            self.all_sprites.add(self.boss)
        return self.get_state()

    def get_state(self):
        # Normalize positions to 0-1
        px, py = self.player.rect.centerx / SCREEN_WIDTH, self.player.rect.centery / SCREEN_HEIGHT
        bx, by = self.boss.rect.centerx / SCREEN_WIDTH, self.boss.rect.centery / SCREEN_HEIGHT
        
        # Normalize HP
        php = self.player.hp / PLAYER_LIVES
        bhp = self.boss.hp / BOSS_MAX_HP
        
        # Bullets (4 nearest to Boss?)
        # For simplicity, let's take 4 nearest bullets to the Boss
        # Or maybe just 4 bullets in general?
        # The requirement says "bullets: [pos, type]"
        # I'll include 4 nearest bullets relative to boss
        
        bullet_list = []
        for b in self.bullets:
            dist = (b.rect.centerx - self.boss.rect.centerx)**2 + (b.rect.centery - self.boss.rect.centery)**2
            bullet_list.append((dist, b.rect.centerx / SCREEN_WIDTH, b.rect.centery / SCREEN_HEIGHT, 1.0 if b.is_player else -1.0))
        
        bullet_list.sort(key=lambda x: x[0])
        
        # Pad with 0 if fewer than 4 bullets
        # We need 8 values (x, y) * 4
        bullet_features = []
        for i in range(4):
            if i < len(bullet_list):
                bullet_features.extend([bullet_list[i][1], bullet_list[i][2]])
            else:
                bullet_features.extend([0.0, 0.0])
                
        # Time since last shot (normalized)
        # Boss shoot delay is 1000ms
        time_since = (pygame.time.get_ticks() - self.boss.last_shot) / 1000.0
        if time_since > 1.0: time_since = 1.0
        
        state = [bx, by, bhp, px, py, php] + bullet_features + [time_since, self.boss.pattern]
        return np.array(state, dtype=np.float32)

    def get_enemy_states(self):
        states = []
        px, py = self.player.rect.centerx / SCREEN_WIDTH, self.player.rect.centery / SCREEN_HEIGHT
        bx, by = self.boss.rect.centerx / SCREEN_WIDTH, self.boss.rect.centery / SCREEN_HEIGHT
        for enemy in self.enemies:
            ex, ey = enemy.rect.centerx / SCREEN_WIDTH, enemy.rect.centery / SCREEN_HEIGHT
            t = (pygame.time.get_ticks() - enemy.last_shot) / 1000.0
            if t > 1.0: t = 1.0
            bullet_list = []
            for b in self.bullets:
                if not b.is_player:
                    continue
                dist = (b.rect.centerx - enemy.rect.centerx)**2 + (b.rect.centery - enemy.rect.centery)**2
                bullet_list.append((dist, b.rect.centerx / SCREEN_WIDTH, b.rect.centery / SCREEN_HEIGHT))
            bullet_list.sort(key=lambda x: x[0])
            bullet_features = []
            for i in range(2):
                if i < len(bullet_list):
                    bullet_features.extend([bullet_list[i][1], bullet_list[i][2]])
                else:
                    bullet_features.extend([0.0, 0.0])
            states.append([ex, ey, px, py, bx, by, t, float(self.round_level)] + bullet_features)
        if len(states) == 0:
            states.append([0.0]*ENEMY_STATE_SIZE)
        return np.array(states, dtype=np.float32)
    def get_player_state(self):
        px, py = self.player.rect.centerx / SCREEN_WIDTH, self.player.rect.centery / SCREEN_HEIGHT
        bx, by = self.boss.rect.centerx / SCREEN_WIDTH, self.boss.rect.centery / SCREEN_HEIGHT
        php = self.player.hp / PLAYER_LIVES
        bhp = self.boss.hp / BOSS_MAX_HP
        bullet_list = []
        for b in self.bullets:
            dist = (b.rect.centerx - self.player.rect.centerx)**2 + (b.rect.centery - self.player.rect.centery)**2
            bullet_list.append((dist, b.rect.centerx / SCREEN_WIDTH, b.rect.centery / SCREEN_HEIGHT, 1.0 if b.is_player else -1.0))
        bullet_list.sort(key=lambda x: x[0])
        bullet_features = []
        for i in range(4):
            if i < len(bullet_list):
                bullet_features.extend([bullet_list[i][1], bullet_list[i][2]])
            else:
                bullet_features.extend([0.0, 0.0])
        t = (pygame.time.get_ticks() - self.player.last_shot) / 1000.0
        if t > 1.0: t = 1.0
        state = [px, py, php, bx, by, bhp] + bullet_features + [t, float(self.wave)]
        return np.array(state, dtype=np.float32)
    def get_enemy_dodge_action(self, enemy):
        nearest = None
        for b in self.bullets:
            if not b.is_player:
                continue
            dist = (b.rect.centerx - enemy.rect.centerx)**2 + (b.rect.centery - enemy.rect.centery)**2
            if nearest is None or dist < nearest[0]:
                nearest = (dist, b)
        if nearest is not None:
            b = nearest[1]
            dx = b.rect.centerx - enemy.rect.centerx
            dy = b.rect.centery - enemy.rect.centery
            if abs(dx) >= abs(dy):
                return ACTION_LEFT if dx > 0 else ACTION_RIGHT
            return ACTION_UP if dy > 0 else ACTION_DOWN
        return None
    def get_player_dodge_action(self):
        px = self.player.rect.centerx
        py = self.player.rect.centery
        nearest = None
        for b in self.bullets:
            if b.is_player:
                continue
            dist = (b.rect.centerx - px)**2 + (b.rect.centery - py)**2
            if nearest is None or dist < nearest[0]:
                nearest = (dist, b)
        if nearest is not None:
            b = nearest[1]
            if b.rect.centerx < px:
                return ACTION_RIGHT
            return ACTION_LEFT
        if self.boss_active:
            if self.boss.rect.centerx > px:
                return ACTION_RIGHT
            return ACTION_LEFT
        if len(self.enemies) > 0:
            ex = list(self.enemies)[0].rect.centerx
            if ex > px:
                return ACTION_RIGHT
            return ACTION_LEFT
        return ACTION_RIGHT if (pygame.time.get_ticks() // 400) % 2 == 0 else ACTION_LEFT

    def step(self, action=None, multi_actions=None):
        self.steps += 1
        reward = REWARD_SURVIVAL
        enemy_reward = 0.0
        player_reward = 0.0
        
        if self.boss_active:
            if multi_actions is not None:
                boss_action = multi_actions.get('boss')
                self.boss.move(boss_action)
                should_fire = boss_action in (ACTION_SHOOT, ACTION_SPECIAL)
                if self.human_player and not should_fire:
                    if pygame.time.get_ticks() - self.boss.last_shot >= 1000:
                        should_fire = True
                if should_fire:
                    bullets = self.boss.shoot()
                    for b in bullets:
                        self.bullets.add(b)
                        self.all_sprites.add(b)
            else:
                if self.human_player:
                    self.boss.move(None)
                else:
                    self.boss.move(action)
                if action in (ACTION_SHOOT, ACTION_SPECIAL):
                    bullets = self.boss.shoot()
                    for b in bullets:
                        self.bullets.add(b)
                        self.all_sprites.add(b)
        
        if self.human_player:
            player_action = None
            if multi_actions is not None:
                player_action = multi_actions.get('player')
            if player_action is None:
                keys = pygame.key.get_pressed()
                self.player.update(keys)
                if keys[pygame.K_SPACE]:
                    offsets = self.get_player_shot_offsets()
                    p_bullets = self.player.shoot(offsets)
                    for b in p_bullets:
                        self.bullets.add(b)
                        self.all_sprites.add(b)
            else:
                fired = (player_action in (ACTION_SHOOT, ACTION_SPECIAL))
                px0 = self.player.rect.x
                py0 = self.player.rect.y
                self.apply_player_action(player_action)
                if not self.training_player and self.player.rect.x == px0 and self.player.rect.y == py0:
                    fallback = self.get_player_dodge_action()
                    if fallback == ACTION_LEFT:
                        self.player.rect.x -= PLAYER_SPEED
                    elif fallback == ACTION_RIGHT:
                        self.player.rect.x += PLAYER_SPEED
                    elif fallback == ACTION_UP:
                        self.player.rect.y -= PLAYER_SPEED
                    elif fallback == ACTION_DOWN:
                        self.player.rect.y += PLAYER_SPEED
                if not self.training_player and not fired:
                    offsets = self.get_player_shot_offsets()
                    p_bullets = self.player.shoot(offsets)
                    for b in p_bullets:
                        self.bullets.add(b)
                        self.all_sprites.add(b)
        else:
            if self.player.rect.centerx < self.boss.rect.centerx:
                self.player.rect.x += PLAYER_SPEED
            elif self.player.rect.centerx > self.boss.rect.centerx:
                self.player.rect.x -= PLAYER_SPEED
            
            if np.random.rand() < 0.1:
                self.player.rect.x += np.random.choice([-PLAYER_SPEED*5, PLAYER_SPEED*5])
                
            p_bullets = self.player.shoot([0])
            for b in p_bullets:
                self.bullets.add(b)
                self.all_sprites.add(b)
        
        # Keep player in bounds
        if self.player.rect.left < 0: self.player.rect.left = 0
        if self.player.rect.right > SCREEN_WIDTH: self.player.rect.right = SCREEN_WIDTH
        if self.player.rect.top < 0: self.player.rect.top = 0
        if self.player.rect.bottom > SCREEN_HEIGHT: self.player.rect.bottom = SCREEN_HEIGHT
            
        self.bullets.update()
        if multi_actions is not None:
            enemy_actions = multi_actions.get('enemies')
            if enemy_actions is None or len(enemy_actions) == 0:
                self.enemies.update()
            else:
                for enemy, ea in zip(list(self.enemies), enemy_actions):
                    enemy.move_action(ea)
        else:
            self.enemies.update()

        if self.human_player:
            self.update_enemies()
            if len(self.enemies) > 0:
                enemy_reward += REWARD_ENEMY_SURVIVAL * len(self.enemies)
        
        if self.boss_active:
            hits = pygame.sprite.spritecollide(self.boss, self.bullets, False)
            for bullet in hits:
                if bullet.is_player:
                    bullet.kill()
                    reward += REWARD_BOSS_HIT
                    player_reward += -REWARD_BOSS_HIT
                    self.score += SCORE_BOSS_HIT
                    if self.boss.take_damage():
                        reward += REWARD_BOSS_DEATH
                        player_reward += -REWARD_BOSS_DEATH
                        self.score += SCORE_BOSS_KILL
                        if self.human_player:
                            self.start_enemy_phase()
                        else:
                            self.done = True
        
        hits = pygame.sprite.spritecollide(self.player, self.bullets, False)
        for bullet in hits:
            if not bullet.is_player:
                bullet.kill()
                reward += REWARD_HIT_PLAYER
                player_reward -= REWARD_HIT_PLAYER
                if getattr(bullet, 'source', 'enemy') == 'enemy':
                    enemy_reward += REWARD_HIT_PLAYER
                if self.player.take_damage():
                    reward += REWARD_PLAYER_LOSE_LIFE
                    player_reward -= REWARD_PLAYER_LOSE_LIFE
                    if getattr(bullet, 'source', 'enemy') == 'enemy':
                        enemy_reward += REWARD_PLAYER_LOSE_LIFE
                    if self.player.hp <= 0:
                        reward += REWARD_GAME_OVER
                        player_reward -= REWARD_GAME_OVER
                        self.done = True

        if self.human_player:
            enemy_hits = pygame.sprite.groupcollide(self.enemies, self.bullets, False, False)
            for enemy, bullets in enemy_hits.items():
                for bullet in bullets:
                    if bullet.is_player:
                        bullet.kill()
                        enemy.kill()
                        self.score += SCORE_ENEMY
                        player_reward += REWARD_SHOOT_HIT
                        enemy_reward += REWARD_ENEMY_HIT
                        break
        info = {}
        if self.training_player and self.waste_bullets_count > 0:
            player_reward += REWARD_WASTE_BULLET * 0.1 * self.waste_bullets_count
            self.waste_bullets_count = 0
        if self.training_player and self.max_steps_episode is not None and self.steps >= self.max_steps_episode:
            self.done = True
        if multi_actions is not None:
            info['enemy_states'] = self.get_enemy_states()
            info['enemy_reward'] = enemy_reward
            info['player_reward'] = player_reward
        return self.get_state(), reward, self.done, info

    def apply_player_action(self, action):
        if action == ACTION_LEFT:
            self.player.rect.x -= PLAYER_SPEED
        elif action == ACTION_RIGHT:
            self.player.rect.x += PLAYER_SPEED
        elif action == ACTION_UP:
            self.player.rect.y -= PLAYER_SPEED
        elif action == ACTION_DOWN:
            self.player.rect.y += PLAYER_SPEED
        if action == ACTION_SHOOT or action == ACTION_SPECIAL:
            offsets = self.get_player_shot_offsets()
            p_bullets = self.player.shoot(offsets)
            for b in p_bullets:
                self.bullets.add(b)
                self.all_sprites.add(b)
            hx = 1 if self.player.rect.centerx < self.boss.rect.centerx else -1
            self.player.rect.x += hx * PLAYER_SPEED
        if self.training_player:
            self.waste_bullets_count += len(p_bullets)

    def render_frame(self, ui_draw=None):
        if not self.render: return
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        
        text = self.font.render(f"Boss HP: {self.boss.hp}", True, WHITE)
        self.screen.blit(text, (10, 10))
        text = self.font.render(f"Player HP: {self.player.hp}", True, WHITE)
        self.screen.blit(text, (10, 50))
        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(text, (10, 90))
        text = self.font.render(f"Wave: {self.wave}/{ENEMY_WAVES}", True, WHITE)
        self.screen.blit(text, (10, 130))
        text = self.font.render(f"Phase: {self.phase}", True, WHITE)
        self.screen.blit(text, (10, 170))
        if ui_draw:
            ui_draw(self.screen, self.font)
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def get_player_shot_offsets(self):
        offsets = SHOT_LEVELS[0][1]
        for threshold, level_offsets in SHOT_LEVELS:
            if self.score >= threshold:
                offsets = level_offsets
        return offsets

    def get_enemy_shot_offsets(self):
        level = min(max(ENEMY_SHOT_LEVELS.keys()), self.round_level)
        return ENEMY_SHOT_LEVELS[level]

    def get_enemy_stats(self):
        scale = 1 + (self.round_level - 1) * ENEMY_DIFFICULTY_GROWTH
        speed = ENEMY_SPEED * scale
        shoot_delay = int(ENEMY_SHOOT_DELAY / scale)
        bullet_speed = ENEMY_BULLET_SPEED * scale
        return speed, shoot_delay, bullet_speed

    def update_enemies(self):
        now = pygame.time.get_ticks()
        if self.phase == "enemies":
            if self.enemy_spawned < ENEMY_PER_WAVE and len(self.enemies) < ENEMY_MAX_ON_SCREEN:
                if now - self.last_spawn > ENEMY_SPAWN_INTERVAL:
                    self.last_spawn = now
                    speed, shoot_delay, _ = self.get_enemy_stats()
                    x = random.randint(ENEMY_SIZE, SCREEN_WIDTH - ENEMY_SIZE)
                    max_y = max(ENEMY_SIZE * 2, self.player.rect.top)
                    y = 0 - ENEMY_SIZE
                    enemy = Enemy(x, y, speed, shoot_delay, max_y)
                    self.enemies.add(enemy)
                    self.all_sprites.add(enemy)
                    self.enemy_spawned += 1

            if self.enemy_spawned >= ENEMY_PER_WAVE and len(self.enemies) == 0:
                if self.wave < ENEMY_WAVES:
                    self.wave += 1
                    self.enemy_spawned = 0
                else:
                    self.start_boss_phase()

        _, _, bullet_speed = self.get_enemy_stats()
        offsets = self.get_enemy_shot_offsets()
        max_y = max(ENEMY_SIZE * 2, self.player.rect.top)
        enemies_list = list(self.enemies)
        sep_r = ENEMY_SIZE * 1.5
        sep_r2 = sep_r * sep_r
        for i, e in enumerate(enemies_list):
            ax = 0.0
            ay = 0.0
            for j, o in enumerate(enemies_list):
                if i == j:
                    continue
                dx = e.rect.centerx - o.rect.centerx
                dy = e.rect.centery - o.rect.centery
                d2 = dx*dx + dy*dy
                if d2 > 0 and d2 < sep_r2:
                    inv = 1.0 / max(1.0, d2)
                    ax += dx * inv
                    ay += dy * inv
            if ax != 0.0 or ay != 0.0:
                e.vx += ax * 0.8
                e.vy += ay * 0.8
                lim = e.speed * 1.8
                if e.vx > lim: e.vx = lim
                if e.vx < -lim: e.vx = -lim
                if e.vy > lim: e.vy = lim
                if e.vy < -lim: e.vy = -lim
        for enemy in enemies_list:
            enemy.max_y = max_y
            bullets = enemy.shoot(bullet_speed, offsets)
            for b in bullets:
                self.bullets.add(b)
                self.all_sprites.add(b)

    def start_boss_phase(self):
        self.phase = "boss"
        self.boss_active = True
        self.boss.hp = BOSS_MAX_HP
        self.boss.rect.center = BOSS_START_POS
        if self.boss not in self.all_sprites:
            self.all_sprites.add(self.boss)
        for sprite in list(self.bullets):
            sprite.kill()
        for sprite in list(self.enemies):
            sprite.kill()

    def start_enemy_phase(self):
        self.phase = "enemies"
        self.boss_active = False
        if self.boss in self.all_sprites:
            self.all_sprites.remove(self.boss)
        for sprite in list(self.bullets):
            sprite.kill()
        self.enemies.empty()
        self.wave = 1
        self.enemy_spawned = 0
        self.round_level += 1
