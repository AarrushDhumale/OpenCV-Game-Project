import cv2
import mediapipe as mp
import numpy as np
import pygame
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


WIDTH, HEIGHT = 1280, 640
player_size = 50
player_pos = [WIDTH // 2, HEIGHT - 100]


enemy_size = 50
enemy_speed = 8
enemy_list = []
points = 0  # Score counter


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Object Dodging Game")
font = pygame.font.Font(pygame.font.match_font('consolas', 'courier', 'monospace'), 36)

def draw_text(text, x, y, color=(0, 0, 0)):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)

def create_enemy():
    x_pos = random.randint(0, WIDTH - enemy_size)
    enemy_list.append([x_pos, 0])

def move_enemies():
    global points
    for enemy in enemy_list[:]:
        enemy[1] += enemy_speed
        if enemy[1] > HEIGHT:
            enemy_list.remove(enemy)
            points += 1  

def check_collision():
    for enemy in enemy_list:
        if (player_pos[0] < enemy[0] + enemy_size // 2 < player_pos[0] + player_size and
            player_pos[1] < enemy[1] + enemy_size // 2 < player_pos[1] + player_size):
            return True
    return False


cap = cv2.VideoCapture(0)
running = True
game_over = False

while running:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_pos = (1 - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x) * WIDTH
            player_pos[0] = int(x_pos - player_size // 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(176, 273, 230), thickness=2))
    
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    frame = pygame.transform.scale(frame, (WIDTH, HEIGHT))
    screen.blit(frame, (0, 0))
    
    if not game_over:
        if random.randint(1, 35) == 1:
            create_enemy()
        move_enemies()
        
        for enemy in enemy_list:
            pygame.draw.circle(screen, (255, 0, 0), (enemy[0] + enemy_size // 2, enemy[1] + enemy_size // 2), enemy_size // 2)
        
        pygame.draw.circle(screen, (0, 0, 255), (player_pos[0] + player_size // 2, player_pos[1] + player_size // 2), player_size // 2)
        pygame.draw.rect(screen, (0, 0, 0), (25, 10, 200, 50))
        draw_text(f"Score: {points}", 125, 35, (255, 255, 255))
        
        if check_collision():
            game_over = True
    else:
        screen.fill((0, 0, 0))
        draw_text("GAME OVER!", WIDTH // 2, HEIGHT // 2 - 80, (255, 0, 0))
        draw_text(f"Final Score: {points}", WIDTH // 2, HEIGHT // 2 - 30, (255, 255, 255))
        draw_text("Press 'R' to Restart", WIDTH // 2, HEIGHT // 2 + 20, (255, 255, 255))
        draw_text("Press 'Q' to Quit", WIDTH // 2, HEIGHT // 2 + 60, (255, 255, 255))
        pygame.display.flip()
        
        while game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_over = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        enemy_list.clear()
                        points = 0
                        game_over = False
                    elif event.key == pygame.K_q:
                        running = False
                        game_over = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    pygame.display.flip()
    pygame.time.delay(30)

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
