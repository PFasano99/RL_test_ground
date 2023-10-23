import pygame
import sys
from maze_generator import *

# Define colors
WHITE = (255, 255, 255)
BROWN = (139, 69, 19)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255,255,0)
ORANGE = (255, 165, 0)
BLACK = (0,0,0)

def draw_matrix_grid(matrix, size_x, size_y):
    GRID_HEIGHT = size_x
    GRID_WIDTH = size_y
    CELL_SIZE = 10  # Size of each cell in pixels
    WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + 200
    WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Matrix Grid")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    window.fill((0, 0, 0))

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            cell_value = matrix[row,col]
            if cell_value == 0:
                color = WHITE
            elif cell_value == 1:
                color = BROWN
            elif cell_value == 2:
                color = BLUE
            elif cell_value == 3:
                color = GREEN
            elif cell_value == 4:
                color = RED
            elif cell_value == 5:
                color = ORANGE
            elif cell_value == 6:
                color = YELLOW
            elif cell_value == 9:
                color = BLACK
                
            pygame.draw.rect(window, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()

    return window

def update_ui(window, matrix, size_x, size_y, score):
    GRID_HEIGHT = size_x
    GRID_WIDTH = size_y
    CELL_SIZE = 10  # Size of each cell in pixels
    WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + 200
    WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

    # Clear the screen
    window.fill(BLACK)

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            cell_value = matrix[row,col]
            if cell_value == 0:
                color = WHITE
            elif cell_value == 1:
                color = BROWN
            elif cell_value == 2:
                color = BLUE
            elif cell_value == 3:
                color = GREEN
            elif cell_value == 4:
                color = RED
            elif cell_value == 5:
                color = ORANGE
            elif cell_value == 6:
                color = YELLOW
            elif cell_value == 9:
                color = BLACK
                
            pygame.draw.rect(window, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    score_txt = "Score: "+ str(round(score, 2))
    display_text(score_txt, WINDOW_WIDTH, window)

    pygame.display.flip()

    return window

# Function to render and display text in the top right corner
def display_text(text, width, screen, font_size = 30):

    font = pygame.font.Font(None, font_size)
    
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.topright = (width - 10, 10)
    screen.blit(text_surface, text_rect)

#generator = maze_generator()

#maze, path = generator.generate_maze()

#draw_matrix_grid(maze, size_x=70, size_y=100)

#input()
# Quit pygame
#pygame.quit()
#sys.exit()