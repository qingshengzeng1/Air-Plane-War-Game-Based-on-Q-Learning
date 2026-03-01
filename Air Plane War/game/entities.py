
import os
import pygame
from game.config import *
import random

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
        player_path = os.path.join(asset_dir, "player.png")
        if os.path.exists(player_path):
            self.image = pygame.image.load(player_path)
            if pygame.display.get_surface():
                self.image = self.image.convert_alpha()
            self.image = pygame.transform.scale(self.image, (PLAYER_SIZE, PLAYER_SIZE))
        else:
            self.image = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE))
            self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect(center=(x, y))
        self.hp = PLAYER_LIVES
        self.speed = PLAYER_SPEED
        self.last_shot = 0
        self.shoot_delay = 500  # ms

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_UP]:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN]:
            self.rect.y += self.speed

        # Keep in bounds
        if self.rect.left < 0: self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH: self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0: self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT: self.rect.bottom = SCREEN_HEIGHT

    def shoot(self, offsets=None):
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shoot_delay:
            self.last_shot = now
            if offsets is None:
                offsets = [0]
            bullets = []
            for speed_x in offsets:
                bullets.append(Bullet(self.rect.centerx, self.rect.top, -BULLET_SPEED, True, speed_x, source='player'))
            return bullets
        return []

    def take_damage(self):
        self.hp -= 1
        return self.hp <= 0

class Boss(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
        boss_path = os.path.join(asset_dir, "boss.png")
        if os.path.exists(boss_path):
            self.image = pygame.image.load(boss_path)
            if pygame.display.get_surface():
                self.image = self.image.convert_alpha()
            self.image = pygame.transform.scale(self.image, (BOSS_SIZE, BOSS_SIZE))
        else:
            self.image = pygame.Surface((BOSS_SIZE, BOSS_SIZE))
            self.image.fill(BOSS_COLOR)
        self.rect = self.image.get_rect(center=(x, y))
        self.hp = BOSS_MAX_HP
        self.speed = BOSS_SPEED
        self.last_shot = 0
        self.shoot_delay = 600
        self.pattern = 0
        self.action = 0
        self.vx = random.choice([-1, 1]) * self.speed
        self.change_dir_chance = 0.03
    
    def move(self, action):
        if action is None:
            if random.random() < self.change_dir_chance:
                self.vx = random.choice([-1, 1]) * self.speed
            self.rect.x += self.vx
        else:
            if action == ACTION_LEFT:
                self.rect.x -= self.speed
            elif action == ACTION_RIGHT:
                self.rect.x += self.speed
            elif action == ACTION_UP:
                self.rect.y -= self.speed
            elif action == ACTION_DOWN:
                self.rect.y += self.speed
        
        # Keep in bounds
        if self.rect.left < 0: self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH: self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0: self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT: self.rect.bottom = SCREEN_HEIGHT
        if action is None:
            if self.rect.left == 0 or self.rect.right == SCREEN_WIDTH:
                self.vx = -self.vx

    def shoot(self):
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shoot_delay:
            self.last_shot = now
            patterns = [
                [0],
                [-2, 2],
                [-3, 0, 3],
                [-4, -2, 0, 2, 4],
                [-5, -3, -1, 1, 3, 5],
            ]
            offsets = random.choice(patterns)
            self.pattern = len(offsets)
            bullets = []
            for speed_x in offsets:
                bullets.append(Bullet(self.rect.centerx, self.rect.bottom, BULLET_SPEED, False, speed_x, source='boss'))
            return bullets
        return []

    def take_damage(self):
        self.hp -= 10 # Boss takes 10 damage per hit
        if self.hp < 0: self.hp = 0
        return self.hp <= 0

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, speed, shoot_delay, max_y):
        super().__init__()
        asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
        enemy_path = os.path.join(asset_dir, "enemy.png")
        if os.path.exists(enemy_path):
            self.image = pygame.image.load(enemy_path)
            if pygame.display.get_surface():
                self.image = self.image.convert_alpha()
            self.image = pygame.transform.scale(self.image, (ENEMY_SIZE, ENEMY_SIZE))
        else:
            self.image = pygame.Surface((ENEMY_SIZE, ENEMY_SIZE))
            self.image.fill(ENEMY_COLOR)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = speed
        self.last_shot = 0
        self.shoot_delay = shoot_delay
        self.max_y = max_y
        self.vx = random.uniform(-self.speed, self.speed)
        self.vy = random.uniform(-self.speed, self.speed)
        if self.vx == 0 and self.vy == 0:
            self.vx = self.speed

    def update(self):
        if random.random() < 0.05:
            self.vx = random.uniform(-self.speed * 1.5, self.speed * 1.5)
            self.vy = random.uniform(-self.speed * 1.5, self.speed * 1.5)
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)
        if self.rect.left < 0: self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH: self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
            if self.vy < 0:
                self.vy = -self.vy
        if self.rect.bottom > self.max_y:
            self.rect.bottom = self.max_y
            if self.vy > 0:
                self.vy = -self.vy
        if self.rect.left == 0 or self.rect.right == SCREEN_WIDTH:
            self.vx = -self.vx

    def move_action(self, action):
        if random.random() < 0.05:
            self.vx = random.uniform(-self.speed * 1.5, self.speed * 1.5)
            self.vy = random.uniform(-self.speed * 1.5, self.speed * 1.5)
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)
        if action == ACTION_LEFT:
            self.rect.x -= self.speed
        elif action == ACTION_RIGHT:
            self.rect.x += self.speed
        elif action == ACTION_UP:
            self.rect.y -= self.speed
        elif action == ACTION_DOWN:
            self.rect.y += self.speed
        if self.rect.left < 0: self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH: self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
            if self.vy < 0:
                self.vy = -self.vy
        if self.rect.bottom > self.max_y:
            self.rect.bottom = self.max_y
            if self.vy > 0:
                self.vy = -self.vy
        if self.rect.left == 0 or self.rect.right == SCREEN_WIDTH:
            self.vx = -self.vx

    def shoot(self, bullet_speed, offsets):
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shoot_delay:
            self.last_shot = now
            bullets = []
            for speed_x in offsets:
                bullets.append(Bullet(self.rect.centerx, self.rect.bottom, bullet_speed, False, speed_x, source='enemy'))
            return bullets
        return []

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, speed_y, is_player, speed_x=0, source=None):
        super().__init__()
        self.image = pygame.Surface((BULLET_SIZE, BULLET_SIZE))
        self.image.fill(BULLET_COLOR_PLAYER if is_player else BULLET_COLOR_ENEMY)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed_y = speed_y
        self.speed_x = speed_x
        self.is_player = is_player
        self.source = source if source else ('player' if is_player else 'enemy')

    def update(self):
        self.rect.y += self.speed_y
        self.rect.x += self.speed_x
        if self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT or self.rect.right < 0 or self.rect.left > SCREEN_WIDTH:
            self.kill()
