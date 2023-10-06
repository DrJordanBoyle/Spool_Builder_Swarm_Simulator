''''
This sofware was used to perform the experiments described in 
"An exploration of control and coordination strategies for swarms 
of autonomous construction robots.", submitted to Spool. Namely:
    Leveling of an initially random, bumpy terrain field
    Building "all or nothing" patterns based on a global template    
    Building a fuzzy verions of a repeated local template
    Doing 'repair work' on a smooth shape
'''


''' Import Necessary Libraries '''
import sys
import pygame
import random
import math
import numpy as np
import datetime
import os
import time
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

max_val = 0
min_val = 0

''' Define constants '''
# Simulation
FPS = 30
EXPORT_DT = 30
DEBUG = False
SHOW_VIEWS = True
AUTO_EXPORT = True
PLOT_AND_IMAGE = True

# Terrain 
WIDTH = 300
START_TERRAIN = 'empty'  # Options: 'empty', 'random', 'template'
TEMPLATE_SCALEUP = 10
TERRAIN_AMOUNT = 5 #25 
BOUNDARY = 'wrapping' # Options: 'wrapping' or 'wall'

# Agent
N_AGENTS = 80   # 80 to 100 (depending on computer "mood") seems to be the max that will run at 30FPS after optimising the match functions
SPEED = 100
RADIUS = 5      #Must not be more than SUB_WIDTH / 2!
P_TURN = 0.02
P_PD_MULT = 100
P_PD_POW = 1
SUB_WIDTH = 30
SPREAD_OUT = True

# Pheremone 
BUILD_PH = 0    # Pheremones can be 0 - 2
TERRAIN_PH = 1
BLUEPRINT_PH = 2

BUILD_PH_TURN_MOD = 10  
BUILD_PH_THRESHOLD = 5

BLUEPRINT_PH_BLUR_SIGMA = 10
BLUEPRINT_PH_STEEPNESS = 50
BLUEPRINT_PH_MIDPOINT = 0.4

TERRAIN_PH_BLUR_SIGMA = 50
TERRAIN_PH_STEEPNESS = 10
TERRAIN_PH_MIDPOINT = 0.5

PH_DEPOSIT = [(1000*TERRAIN_AMOUNT)//255, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max]
PH_DECAY_DENOMINATOR = [400, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max]
PH_DECAY_DENOMINATOR = np.reshape(PH_DECAY_DENOMINATOR, (1, 1, 3))
PH_RADIUS = [50, 75, 15]
RADIUS_MULTIPLIER = 2
USER_PH_MAX = 0.25

# General UI
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Template Builder UI
MAGNIFY = 10                # Magnify factor for template drawing
POSITION_OFFSET = 10        # Offset from top left of screen for draw window
LINE_WIDTH = 2              # Width of line around draw window
DIRECTORY = 'templates/'    # Sub-directory where template files are stored
BORDER = 10                 # Size of border around grid of template images
SPACING = SUB_WIDTH + 10    # Spacing between template images

# Terrain damage
N_DAMAGE = 10
R_DAMAGE = 2*RADIUS
M_DAMAGE = 10*TERRAIN_AMOUNT
T_DAMAGE = 5

# For gradient computation
LAPLACIAN_KERNEL = np.array([
    [0, 0, 0, -1, 0, 0, 0],
    [0, 0, -1, -2, -1, 0, 0],
    [0, -1, -2, -3, -2, -1, 0],
    [-1, -2, -3, 40, -3, -2, -1],
    [0, -1, -2, -3, -2, -1, 0],
    [0, 0, -1, -2, -1, 0, 0],
    [0, 0, 0, -1, 0, 0, 0]
])

''' Class Definitions '''
class Agent:
    """Define class for agent."""
    def __init__(self, arena_width, sub_width, spread_out=False):
        if spread_out:
            self.x = random.uniform(5, arena_width-5)
            self.y = random.uniform(5, arena_width-5)
        else:
            self.x = arena_width/2
            self.y = arena_width/2
            
        self.theta = random.uniform(0.0, 2 * math.pi)        
        self.last_pheremone = [0,0,0]
        self.view = np.zeros((sub_width, sub_width,3))
        self.build_state = 'idle'
        self.min_max = [0, 0]

''' Initializer Functions '''
def initialize_agents():
    # Initialize and return list of agents.
    return [Agent(WIDTH, SUB_WIDTH, SPREAD_OUT) for _ in range(N_AGENTS)]

def initialize_terrain(pheremone_map):    
    # Build terrain from template
    if START_TERRAIN == 'template':
        terrain_map_2d = np.round(pheremone_map[:, :, TERRAIN_PH]*(200/np.max(pheremone_map[:, :, TERRAIN_PH])))
        return np.stack([terrain_map_2d]*3, axis=-1).astype('uint8')
    # Build terrain from random noise
    elif START_TERRAIN == 'random':
        SCALEUP = 2                     
        p_terrain = 0.5  # Chance of an element being 0
        p_empty = 1 - p_terrain  # Chance of an element being 255
        terrain_map_course = np.random.choice([0, 255], size=(round(WIDTH/SCALEUP), round(WIDTH/SCALEUP)), p=[p_empty, p_terrain])        
        terrain_map_2d = np.repeat(terrain_map_course, SCALEUP, axis=0)
        terrain_map_2d = np.repeat(terrain_map_2d, SCALEUP, axis=1)  
        terrain_map_2d = gaussian_filter(terrain_map_2d, sigma=2*SCALEUP)
        return np.stack([terrain_map_2d]*3, axis=-1).astype('uint8')    
    # Blank terrain
    else:
        return np.zeros((WIDTH, WIDTH, 3), dtype='uint8')
    
def initialize_pheremone(global_template=None, gradient_template=None):       
    # Each color channel can be a different pheremone  
    # Needs to be int32 to prevent value wraparound before clipping in the deposit function
    pheremone_array = np.zeros((WIDTH, WIDTH, 3), dtype='int32')
    if global_template is not None:
        N_repeat = WIDTH//SUB_WIDTH
        global_template_scaleup = np.repeat((np.repeat(global_template, N_repeat, axis=0)), N_repeat, axis=1)
        pheremone_array[:,:,BLUEPRINT_PH] = gaussian_filter((global_template_scaleup*np.iinfo(np.uint16).max), sigma=BLUEPRINT_PH_BLUR_SIGMA)
    
    if gradient_template is not None:
        N_repeat = WIDTH//SUB_WIDTH
        gradient_template_scaleup = np.repeat((np.repeat(gradient_template, N_repeat, axis=0)), N_repeat, axis=1)
        pheremone_array[:,:,TERRAIN_PH] = gaussian_filter((gradient_template_scaleup*np.iinfo(np.uint16).max), sigma=TERRAIN_PH_BLUR_SIGMA)
       
    return pheremone_array

''' Pre-Simulation User Interface Functions '''
def make_template(screen, clock, black_square_surface, font):
    # Initialize template
    template_user = np.zeros((SUB_WIDTH, SUB_WIDTH, 3), dtype='uint8')
    cursor_index = 5    
    cursor_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10, 7: 15}
    cursor_size = cursor_dict[cursor_index]
    done = False  

    # Main loop
    while True:         
        template_display = np.repeat(np.repeat(template_user, MAGNIFY, axis=0), MAGNIFY, axis=1)
        template_surface = pygame.surfarray.make_surface(template_display)
        screen.blit(black_square_surface, (0, 0)) 
        screen.blit(template_surface, (POSITION_OFFSET, POSITION_OFFSET))
        pygame.draw.rect(screen, GREEN, (POSITION_OFFSET - LINE_WIDTH, POSITION_OFFSET - LINE_WIDTH, SUB_WIDTH * MAGNIFY + 2*LINE_WIDTH, SUB_WIDTH * MAGNIFY + 2*LINE_WIDTH), LINE_WIDTH)
        mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
        N = MAGNIFY * cursor_size
        square_pos_x = ((mouse_pos_x - POSITION_OFFSET) // MAGNIFY) * MAGNIFY + POSITION_OFFSET
        square_pos_y = ((mouse_pos_y - POSITION_OFFSET) // MAGNIFY) * MAGNIFY + POSITION_OFFSET
        pygame.draw.rect(screen, BLUE, (square_pos_x, square_pos_y, N, N), 1)
        text = font.render("Left / Right click to Add / Remove", True, (0, 0, 255))
        screen.blit(text, (20, WIDTH*2 - 150))
        text = font.render("Up / Down to Increase / Decrease cursor size", True, (0, 0, 255))
        screen.blit(text, (20, WIDTH*2 - 120))
        text = font.render("Space to exit", True, (0, 0, 255))
        screen.blit(text, (20, WIDTH*2 - 90))
        text = font.render(f"Cursor size: {int(cursor_size*(WIDTH/SUB_WIDTH))}", True, (0, 0, 255))
        screen.blit(text, (WIDTH*1.2, 50))
        
        pygame.display.flip()           

        # Handle events
        for event in pygame.event.get():                   
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    done = True
                elif event.key == pygame.K_UP:
                    cursor_index = min(cursor_index + 1, 7)
                elif event.key == pygame.K_DOWN:
                    cursor_index = max(cursor_index - 1, 1)
                cursor_size = cursor_dict[cursor_index]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_button = event.button
                
                if mouse_pos_x < SUB_WIDTH*MAGNIFY + POSITION_OFFSET and mouse_pos_x > POSITION_OFFSET and mouse_pos_y < SUB_WIDTH*MAGNIFY + POSITION_OFFSET and mouse_pos_y > POSITION_OFFSET:            
                    click_x = round((square_pos_x - POSITION_OFFSET) // MAGNIFY)
                    click_y = round((square_pos_y - POSITION_OFFSET) // MAGNIFY)
                    
                    if mouse_button == 1:                
                        template_user[click_x:click_x + cursor_size, click_y:click_y + cursor_size, :] = WHITE
                    
                    elif mouse_button == 3:
                        template_user[click_x:click_x + cursor_size, click_y:click_y + cursor_size, :] = BLACK    
        # If done, save template and exit
        if done:                       
            now = datetime.datetime.now()                
            filename = now.strftime('%Y-%m-%d_%H-%M-%S')  
            template_surface = pygame.surfarray.make_surface(template_user)
            pygame.image.save(template_surface, f'templates/{filename}.png')
            return template_surface, template_user[:,:,1]             
    clock.tick(FPS)


def load_template(screen, clock, black_square_surface, font):
    # Get a list of all image file names in that directory
    template_files = os.listdir(DIRECTORY)
    
    # Load all the images into a list of Surfaces
    templates = [pygame.image.load(os.path.join(DIRECTORY, tmp)) for tmp in template_files]
    
    N_per_row = (2*WIDTH - 2*BORDER)//SPACING
    
    choice_made = False
    while not choice_made:
        screen.blit(black_square_surface, (0, 0))
        text = font.render("Select template with mouse", True, (0, 0, 255))
        screen.blit(text, (20, WIDTH*2 - 40))
        
        for index, tmp in enumerate(templates):                                    
            draw_subscreen(screen, tmp, BORDER + (index*SPACING)%(2*WIDTH - (SPACING)), BORDER + SPACING*int((index*SPACING)//(2*WIDTH - (SPACING))), True)
               
        mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()        
        square_pos_x = max(((mouse_pos_x - BORDER) // SPACING) * SPACING + BORDER//2, 0)
        square_pos_y = max(((mouse_pos_y - BORDER) // SPACING) * SPACING + BORDER//2, 0)
        pygame.draw.rect(screen, (0, 0, 255), (square_pos_x, square_pos_y, SPACING, SPACING), 1)        
        pygame.display.flip()
        
        for event in pygame.event.get():                                 
            if event.type == pygame.MOUSEBUTTONDOWN:                
                mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()                    
                mouse_button = event.button                               
                                           
                if mouse_button == 1:
                    row = round(square_pos_y/SPACING)
                    col = round(square_pos_x/SPACING)
                    choice = row * N_per_row + col
                    if choice < len(templates):
                        pygame.draw.rect(screen, GREEN, (square_pos_x, square_pos_y, SPACING, SPACING), 3)                            
                        pygame.display.flip()
                        time.sleep(0.5)
                        choice_made = True               
        
        clock.tick(FPS)
                
    template_surface = templates[choice]            
    chosen_template = pygame.surfarray.array3d(template_surface)
    return template_surface, chosen_template[:,:,1]

def display_instructions(screen, font, template_surface, global_template_surface, gradient_template_surface, black_square_surface):
    """Display instructions on the screen."""   
    screen.blit(black_square_surface, (0, 0))
    screen.blit(black_square_surface, (0, WIDTH))
    screen.blit(black_square_surface, (WIDTH, 0))
    screen.blit(black_square_surface, (WIDTH, WIDTH))
    
    texts = ["1: Load view template", 
             "2: Make view template", 
             "3: Load blueprint template", 
             "4: Make blueprint template",
             "5: Load terrain template",
             "6: Make terrain template",
             f"S: Toggle agents start (S)pread out: {SPREAD_OUT}",
             f"T: Toggle starting (T)errain: {START_TERRAIN}",
             f"B: Toggle (B)oundary: {BOUNDARY}",
             "D: Done"]
    
        
    for i, text in enumerate(texts):
        screen.blit(font.render(text, True, BLUE), (20, 20 + i*30))
        
    if template_surface is not None:
        draw_subscreen(screen, template_surface, WIDTH*1.5, 20, True)  

    if global_template_surface is not None:
        draw_subscreen(screen, global_template_surface, WIDTH*1.5, 80, True)
        
    if gradient_template_surface is not None:
        draw_subscreen(screen, gradient_template_surface, WIDTH*1.5, 140, True)

    pygame.display.flip()

def template_configurator_UI(screen, font, clock):
    finished = False
    template_surface = None
    global_template_surface = None
    gradient_template_surface = None
    local_template = None
    global_template = None
    gradient_template = None
    
    while not finished:        
        black_square = np.zeros((WIDTH*2, WIDTH*2, 3), dtype='uint8')
        black_square_surface = pygame.surfarray.make_surface(black_square)        
        display_instructions(screen, font, template_surface, global_template_surface, gradient_template_surface, black_square_surface)        
        no_choice = True        
        
        while no_choice:             
            for event in pygame.event.get():                   
                                  
                if event.type == pygame.KEYDOWN and event.key == pygame.K_1:                
                    template_surface, local_template = load_template(screen, clock, black_square_surface, font)
                    no_choice = False
    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_2:                
                    template_surface, local_template = make_template(screen, clock, black_square_surface, font)
                    no_choice = False
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_3:                
                    global_template_surface, global_template = load_template(screen, clock, black_square_surface, font)
                    no_choice = False
    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_4:                    
                    global_template_surface, global_template = make_template(screen, clock, black_square_surface, font)
                    no_choice = False
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_5:                
                    gradient_template_surface, gradient_template = load_template(screen, clock, black_square_surface, font)
                    no_choice = False
    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_6:                    
                    gradient_template_surface, gradient_template = make_template(screen, clock, black_square_surface, font)
                    no_choice = False
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    global SPREAD_OUT                    
                    SPREAD_OUT = not SPREAD_OUT
                    no_choice = False
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    global BOUNDARY                    
                    if BOUNDARY == 'wrapping':
                        BOUNDARY = 'wall'
                    else:
                        BOUNDARY = 'wrapping'
                    no_choice = False
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:                                    
                    global START_TERRAIN
                    if START_TERRAIN == 'empty':
                        START_TERRAIN = 'random'
                    elif START_TERRAIN == 'random':
                        START_TERRAIN = 'template'
                    else:
                        START_TERRAIN = 'empty'                    
                    no_choice = False
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_d:                    
                    no_choice = False
                    finished = True
                      
    if local_template is None:
        local_template = np.zeros([SUB_WIDTH, SUB_WIDTH]).astype('uint8')
        
    if global_template is None:
        global_template = np.zeros([SUB_WIDTH, SUB_WIDTH]).astype('uint8')
        
    if gradient_template is None:
        gradient_template = np.zeros([SUB_WIDTH, SUB_WIDTH]).astype('uint8')
    
    return local_template, global_template, gradient_template


''' Update Functions for Material and Pheremone'''
def modify_terrain(terrain_map, agent, amount, deposit, sub_section=False):

    if sub_section:
        start_x, end_x = max(0, SUB_WIDTH//2-RADIUS), min(SUB_WIDTH, SUB_WIDTH//2+RADIUS)
        start_y, end_y = max(0, SUB_WIDTH//2-RADIUS), min(SUB_WIDTH, SUB_WIDTH//2+RADIUS)
        
        # Get the range of x and y values
        x_values = np.arange(start_x, end_x)[:, None]
        y_values = np.arange(start_y, end_y)
        
        # Create a matrix of distances from the center
        distance = np.sqrt((x_values - SUB_WIDTH//2)**2 + (y_values - SUB_WIDTH//2)**2)
        
    else:
        
        start_x, end_x = max(0, round(agent.x)-RADIUS), min(WIDTH, round(agent.x)+RADIUS)
        start_y, end_y = max(0, round(agent.y)-RADIUS), min(WIDTH, round(agent.y)+RADIUS)
        
        # Get the range of x and y values
        x_values = np.arange(start_x, end_x)[:, None]
        y_values = np.arange(start_y, end_y)
        
        # Create a matrix of distances from the center
        distance = np.sqrt((x_values - round(agent.x))**2 + (y_values - round(agent.y))**2)
    
    # Create a circular mask where the distance is less than or equal to PH_RADIUS
    mask = distance <= RADIUS
    
    # Create a linear dropoff from 1 at the center to 0 at the edge of the mask    
    dropoff = mask * (np.cos(np.pi * distance / RADIUS) + 1) / 2
    dropoff = dropoff[:, :, np.newaxis]
    dropoff = np.repeat(dropoff, 3, axis=2)    
    
    # Cast the pheremone map to int32 to prevent overflow, then add the pheremones
    terrain_chunk = (terrain_map[start_x:end_x, start_y:end_y, :].astype(np.int16) + (1 if deposit else -1)*(amount * dropoff).astype(np.int16))
    terrain_chunk = np.clip(terrain_chunk, 0, np.iinfo(np.uint8).max).astype(np.uint8)
    
    terrain_change = np.mean(terrain_map[start_x:end_x, start_y:end_y, :] - terrain_chunk)
    
    # Clip to prevent overflow when casting back to uint16
    terrain_map[start_x:end_x, start_y:end_y, :] = terrain_chunk
    
        
    return terrain_change, terrain_map    

def damage_terrain(terrain_map):
    for i in range(N_DAMAGE):
        x = round(random.uniform(R_DAMAGE, WIDTH-R_DAMAGE-1))
        y = round(random.uniform(R_DAMAGE, WIDTH-R_DAMAGE-1))
        
        start_x, end_x = x - R_DAMAGE, x + R_DAMAGE
        start_y, end_y = y - R_DAMAGE, y + R_DAMAGE
        
        # Get the range of x and y values
        x_values = np.arange(start_x, end_x)[:, None]
        y_values = np.arange(start_y, end_y)
        
        # Create a matrix of distances from the center
        distance = np.sqrt((x_values - x)**2 + (y_values - y)**2)
    
        # Create a circular mask where the distance is less than or equal to PH_RADIUS
        mask = distance <= R_DAMAGE
        
        # Create a linear dropoff from 1 at the center to 0 at the edge of the mask    
        dropoff = mask * (np.cos(np.pi * distance / R_DAMAGE) + 1) / 2
        dropoff = dropoff[:, :, np.newaxis]
        dropoff = np.repeat(dropoff, 3, axis=2)    
        
        # Cast the pheremone map to int32 to prevent overflow, then add the pheremones
        terrain_chunk = (terrain_map[start_x:end_x, start_y:end_y, :].astype(np.int16) + (1 if random.random() < 0.5 else -1)*(M_DAMAGE * dropoff).astype(np.int16))
        terrain_chunk = np.clip(terrain_chunk, 0, np.iinfo(np.uint8).max).astype(np.uint8)       
        terrain_map[start_x:end_x, start_y:end_y, :] = terrain_chunk
    
    return terrain_map

def decay_pheremone(pheremone_map):
    # Pheremone value at each location is reduced proportional to its current value, scaled by the PH_DECAY_DENOMINATOR parameter 
    pheremone_map -= np.minimum(pheremone_map, pheremone_map//PH_DECAY_DENOMINATOR)
    return pheremone_map

def deposit_pheremone(x, y, pheremone_map, which_pheremone, agent_ph=True, by_human=False, deposit=False):
    # Separate cases depending on whether pheremone is being deposited by robot or user (properties are different)
    if by_human:
        
        # Get the bounds of the area to modify    
        start_x, end_x = max(0, round(x)-PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER), min(WIDTH, round(x)+PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER)
        start_y, end_y = max(0, round(y)-PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER), min(WIDTH, round(y)+PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER)
        
        # Get the range of x and y values
        x_values = np.arange(start_x, end_x)[:, None]
        y_values = np.arange(start_y, end_y)
    
        # Create a matrix of distances from the center
        distance = np.sqrt((x_values - round(x))**2 + (y_values - round(y))**2)
    
        # Create a circular mask where the distance is less than or equal to PH_RADIUS
        mask = distance <= PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER
        not_mask = distance > PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER
    
        # Create a linear dropoff from 1 at the center to 0 at the edge of the mask
        dropoff = np.where(mask, (PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER - distance) / PH_RADIUS[which_pheremone]*RADIUS_MULTIPLIER, 0)
        
        if deposit:
            # Cast the pheremone map to int32 to prevent overflow, then add the pheremones
            pheremone_map[start_x:end_x, start_y:end_y, which_pheremone] = (pheremone_map[start_x:end_x, start_y:end_y, which_pheremone].astype(np.int32) + (np.iinfo(np.uint16).max * dropoff * USER_PH_MAX).astype(np.int32))
        else:
            pheremone_map[start_x:end_x, start_y:end_y, which_pheremone] = (pheremone_map[start_x:end_x, start_y:end_y, which_pheremone].astype(np.int32) * not_mask)
       
        # Clip to prevent overflow when casting back to uint16
        pheremone_map[start_x:end_x, start_y:end_y, which_pheremone] = np.clip(pheremone_map[start_x:end_x, start_y:end_y, which_pheremone], 0, np.iinfo(np.uint16).max).astype(np.uint16)        
        
    else:    
        if agent_ph:
            # Get the bounds of the area to modify    
            start_x, end_x = max(0, round(x)-PH_RADIUS[which_pheremone]), min(WIDTH, round(x)+PH_RADIUS[which_pheremone])
            start_y, end_y = max(0, round(y)-PH_RADIUS[which_pheremone]), min(WIDTH, round(y)+PH_RADIUS[which_pheremone])
            
            # Get the range of x and y values
            x_values = np.arange(start_x, end_x)[:, None]
            y_values = np.arange(start_y, end_y)
        
            # Create a matrix of distances from the center
            distance = np.sqrt((x_values - round(x))**2 + (y_values - round(y))**2)
        
            # Create a circular mask where the distance is less than or equal to PH_RADIUS
            mask = distance <= PH_RADIUS[which_pheremone]
        
            # Create a linear dropoff from 1 at the center to 0 at the edge of the mask
            dropoff = np.where(mask, (PH_RADIUS[which_pheremone] - distance) / PH_RADIUS[which_pheremone], 0)
            
            # Cast the pheremone map to int32 to prevent overflow, then add the pheremones
            pheremone_map[start_x:end_x, start_y:end_y, which_pheremone] = (pheremone_map[start_x:end_x, start_y:end_y, which_pheremone].astype(np.int32) + (PH_DEPOSIT[which_pheremone] * dropoff).astype(np.int32))
            
            # Clip to prevent overflow when casting back to uint16
            pheremone_map[start_x:end_x, start_y:end_y, which_pheremone] = np.clip(pheremone_map[start_x:end_x, start_y:end_y, which_pheremone], 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return pheremone_map

def get_agent_view(agent, terrain_map):
    # It's worth noting that the agent view wraps around the boundaries, even if
    # the agent movement does not! This might be somethign to change...
    start_x = round(agent.x)-round(SUB_WIDTH/2)
    end_x = round(agent.x)+round(SUB_WIDTH/2)
    x_range = np.mod(np.arange(start_x, end_x), WIDTH).reshape(-1,1)
    start_y = round(agent.y)-round(SUB_WIDTH/2)
    end_y = round(agent.y)+round(SUB_WIDTH/2)
    y_range = np.mod(np.arange(start_y, end_y), WIDTH).reshape(1,-1)        
    return terrain_map[x_range, y_range, :]


def update_agent_and_terrain(agent, dt, terrain_map, template_fft, pheremone_map, agent_ph=True, first_iteration=False, debug=False):      
    # Random walk biased by pheremone concentrations
    sensed_pheremone = pheremone_map[math.floor(agent.x), math.floor(agent.y),:].astype('int32')
    delta_pheremone = sensed_pheremone - agent.last_pheremone    
    agent.last_pheremone = sensed_pheremone
          
    # Decrease, increase or leave turn probability depending change in build pheremone concentration
    if delta_pheremone[BUILD_PH] > BUILD_PH_THRESHOLD:
        modified_probability = P_TURN / BUILD_PH_TURN_MOD
    elif delta_pheremone[BUILD_PH] < -1*BUILD_PH_THRESHOLD:
        modified_probability = P_TURN * BUILD_PH_TURN_MOD
    else:
        modified_probability = P_TURN
    
    # Then roll the dice to see if you change direction    
    if random.random() < (modified_probability):
        agent.theta += random.gauss(math.pi, math.pi/4)
       
    # Then move
    dx = SPEED*dt*math.cos(agent.theta)
    dy = SPEED*dt*math.sin(agent.theta)
    agent.x += dx
    agent.y += dy    
    
    if BOUNDARY == 'wrapping':
        agent.x = agent.x % WIDTH
        agent.y = agent.y % WIDTH
        
    elif BOUNDARY == 'wall':
    # If agent hits the right or left boundary, reverse x direction
        if agent.x < 0 or agent.x > WIDTH:
            dx = -dx  # Reflect the x-component of the velocity
            agent.theta = math.atan2(dy, dx)  # Recalculate the angle based on the new velocity components
            agent.x = max(0.1, min(agent.x, WIDTH-0.1))  # Keep agent.x within [0, WIDTH]
        
        # If agent hits the top or bottom boundary, reverse y direction
        if agent.y < 0 or agent.y > WIDTH:
            dy = -dy  # Reflect the y-component of the velocity
            agent.theta = math.atan2(dy, dx)  # Recalculate the angle based on the new velocity components
            agent.y = max(0.1, min(agent.y, WIDTH-0.1))  # Keep agent.y within [0, WIDTH]
        
    
    # Implement state-based pickup / deposit behaviour
    # If in 'global' build state, pickup / deposit based on pheremone template
    if agent.build_state == 'global':     
        blueprint_response = (sensed_pheremone[BLUEPRINT_PH])/(np.iinfo(np.uint16).max)        
        deposit = sigmoid(blueprint_response, BLUEPRINT_PH_STEEPNESS, BLUEPRINT_PH_MIDPOINT)
        pickup = (1-deposit)         
            
    elif agent.build_state == 'local':
        # Impliment pickup / deposit behaviour based on template match                
        agent.view = get_agent_view(agent, terrain_map)        
        deposit, pickup = perform_template_match(template_fft, agent, TERRAIN_AMOUNT, False)
        
    elif agent.build_state == 'level':       
        # Impliment pickup / deposit behaviour based on template match             
        agent.view = get_agent_view(agent, terrain_map)       
        deposit, pickup = perform_trough_peak_detection(agent, True) 
            
    elif agent.build_state == 'idle':
        # Just hang out and move around              
        agent.view = get_agent_view(agent, terrain_map) 
        agent = learn_terrain(agent)            
        deposit, pickup = 0, 0
        
    elif agent.build_state == 'repair':
        # Impliment repair  of existing terrain               
        agent.view = get_agent_view(agent, terrain_map)               
        deposit, pickup = perform_repairs(agent)
    
    # Do deposit / pickup based on probabilities (note that at the moment these are often 0 / 1)           
    if random.random() < deposit and not first_iteration:
        #terrain_map = deposit_material(terrain_map, agent)
        terrain_change, terrain_map = modify_terrain(terrain_map, agent, TERRAIN_AMOUNT, True)        
        if terrain_change != 0:
            pheremone_map = deposit_pheremone(agent.x, agent.y, pheremone_map, BUILD_PH, agent_ph)                         
    
    if random.random() < pickup and not first_iteration:
        #terrain_map = pickup_material(terrain_map, agent)
        terrain_change, terrain_map = modify_terrain(terrain_map, agent, TERRAIN_AMOUNT, False)
        if terrain_change != 0:
            pheremone_map = deposit_pheremone(agent.x, agent.y, pheremone_map, BUILD_PH, agent_ph)       
    
    return agent, terrain_map, pheremone_map


''' Utility functions for implementing robot senses '''
def sigmoid(x, steepness=1, midpoint=0):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))


def prepare_template(template):
    # No longer calculating norm because not used
    # Compute the two-dimensional FFT of the template array
    template_fft = np.fft.fft2(template)
    # Compute the normalization constant for the template
       
    return template_fft.conjugate()


def match_template(template_fft, sample):
    # Note: this function has changed, and is no longer normalising the correlation
    # by the norm of the signal and template! This was necessary to solve a problem
    # where deposition in a logical place was not improving the correlation value.
    # It does mean that the absolute value is less meaningful, but we are always
    # comparing different terrain states, so this doesn't matter.
    
    # Compute the two-dimensional FFT of the sample array
    sample_fft = np.fft.fft2(sample)
    # Perform element-wise multiplication of the complex conjugates
    correlation = np.fft.ifft2(template_fft * sample_fft)  
    
    # Find the maximum correlation value
    max_correlation = np.abs(correlation).max()
    return max_correlation


def perform_template_match(template_fft, agent, amount, debug=False):
    deposit_array = np.copy(agent.view)    
    deposit_changed, deposit_array = modify_terrain(deposit_array, agent, amount, True, True)
    
    pickup_array = np.copy(agent.view)
    pickup_changed, pickup_array = modify_terrain(pickup_array, agent, amount, False, True)
    
    match = match_template(template_fft, agent.view[:, :, 1])
    
    if deposit_changed != 0:
        deposit_match = match_template(template_fft, deposit_array[:, :, 1])
    else:
        deposit_match = match
    
    if pickup_changed != 0:
        pickup_match = match_template(template_fft, pickup_array[:, :, 1])
    else:
        pickup_match = match
        
    # Determine the change in match for both possible terrain modifications        
    delta_deposit = deposit_match - match
    delta_pickup = pickup_match - match
    
    if debug:
        #print(f"match = {match}")
        #print(f"pickup match = {pickup_match}")
        print(f"delta pickup = {delta_pickup}")
        #print(f"deposit match = {deposit_match}")
        print(f"delta deposit = {delta_deposit}")
        
    # Calulate pickup and deposit probabilities (only one of which can be nonzero)
    if delta_deposit > 0 and delta_deposit > delta_pickup:
        deposit = 1 
        pickup = 0
    elif delta_pickup > 0 and delta_pickup > delta_deposit:
        pickup = 1 
        deposit = 0
    else:
        pickup = 0
        deposit = 0   
   
    return deposit, pickup
 

def get_laplacian(sample, normalise=True):    
    # Convolve the data with the Laplacian kernel
    laplacian_result = convolve(sample.astype('float64'), LAPLACIAN_KERNEL) 
    laplacian_smooth = gaussian_filter(laplacian_result, 2)    
    if normalise and np.max(np.abs(laplacian_smooth)) != 0:
        return laplacian_smooth/np.max(np.abs(laplacian_smooth))
    else:
        return laplacian_smooth  

def get_local_curvature(sample):
    sample_laplace = get_laplacian(sample, True)
    
    return np.mean(sample_laplace[SUB_WIDTH//2 - 1: SUB_WIDTH//2, SUB_WIDTH//2 - 1: SUB_WIDTH//2])

def learn_terrain(agent):
    local_curvature = get_local_curvature(agent.view[:, :, 1])
    if local_curvature > agent.min_max[1]:
        agent.min_max[1] = local_curvature
        
    elif local_curvature < agent.min_max[0]:
        agent.min_max[0] = local_curvature
        
    return agent

def perform_repairs(agent):    
    local_curvature = get_local_curvature(agent.view[:, :, 1])    
    deposit = 0
    pickup = 0    
    
    if local_curvature > 1.1*agent.min_max[1]:
        pickup = 2.0*(local_curvature - 1.1*agent.min_max[1])
        deposit = 0
    elif local_curvature < 1.1*agent.min_max[0]:
        pickup = 0
        deposit = -2.0*(local_curvature - 1.1*agent.min_max[0])
             
    return deposit, pickup


def perform_trough_peak_detection(agent, build_in_troughs=True):
    
    view_laplace = get_laplacian(agent.view[:, :, 1], True)
    local_curvature = np.mean(view_laplace[SUB_WIDTH//2 - 1: SUB_WIDTH//2, SUB_WIDTH//2 - 1: SUB_WIDTH//2])
    
    deposit = 0
    pickup = 0
    
    if build_in_troughs:
        if local_curvature > 0:
            pickup = local_curvature
            deposit = 0
        else:
            pickup = 0
            deposit = -1*local_curvature
    else:
        if local_curvature < 0:
            pickup = -1*local_curvature
            deposit = 0
        else:
            pickup = 0
            deposit = local_curvature           
    #print([local_curvature, pickup, deposit])        
    return deposit, pickup
    
    

''' Utility Functions for populating simulation screen '''
def draw_agent(screen, agent, agent_ph=True):
    """Draw agent on the screen."""
    if agent.build_state == 'global':     
        agent_color = "red"
    elif agent.build_state == 'local':
        agent_color = "green"
    elif agent.build_state == 'level':
        agent_color = "orange"    
    elif agent.build_state == 'repair':
        agent_color = "cyan"
    else:
        agent_color = "gray"
        
    pygame.draw.circle(screen, agent_color, (agent.x, agent.y), 4)

def draw_subscreen(screen, subscreen_array, x_location, y_location, already_surface=False):
    """Draw subscreen on the screen."""
    if not already_surface:
        subscreen_surface = pygame.surfarray.make_surface(subscreen_array)   
    else:
        subscreen_surface = subscreen_array
    screen.blit(subscreen_surface, (x_location, y_location))
    pygame.draw.rect(screen, GREEN, (x_location, y_location, SUB_WIDTH, SUB_WIDTH), 1)
    
def make_black_square_surface():
    black_square = np.zeros((WIDTH, WIDTH, 3), dtype='uint8')
    return pygame.surfarray.make_surface(black_square)

def make_template_image(template):
    return np.stack([template]*3, axis=-1).astype('uint8')

def draw_screen_background(screen, terrain_map, pheremone_map):
    black_square_surface = make_black_square_surface()
    terrain_surface = pygame.surfarray.make_surface(terrain_map)    
    screen.blit(terrain_surface, (0, 0))
    screen.blit(black_square_surface, (WIDTH, 0))
    screen.blit(black_square_surface, (WIDTH, WIDTH))    
    pheremone_surface = pygame.surfarray.make_surface(pheremone_map//257) 
    screen.blit(pheremone_surface, (0, WIDTH)) 
   
    
def draw_screen_overlay(screen, template_image, font, small_font, dt, agent_ph): #!!
    if template_image is not None:
        draw_subscreen(screen, template_image, 2*WIDTH - SUB_WIDTH - 10, WIDTH - SUB_WIDTH - 10)        
    
    text = font.render("Pheremone Map", True, BLUE)
    screen.blit(text, (WIDTH//3, WIDTH+10))
    text = font.render("FPS = " + str(round(1/dt)), True, RED)
    screen.blit(text, (WIDTH+10, 10))        
    text = font.render("Run Time (s): " + str(round(pygame.time.get_ticks()/1000)), True, RED)
    screen.blit(text, (WIDTH+10, 30))
    text = font.render("Interaction Instructions: ", True, RED)
    screen.blit(text, (WIDTH+10, 60))
    text = small_font.render("Click on terrain or pheremone map ", True, RED)
    screen.blit(text, (WIDTH+10, 75))
    text = small_font.render("Left = deposit; Right = remove ", True, RED)
    screen.blit(text, (WIDTH+10, 90))
    text = small_font.render(f"Toggle robot (P)heremone deposit [{agent_ph}]", True, RED)
    screen.blit(text, (WIDTH+10, 115))
    text = small_font.render("Toggle (H)ide / unhide agents", True, RED)
    screen.blit(text, (WIDTH+10, 130))
    text = small_font.render("Select behaviour:", True, RED)
    screen.blit(text, (WIDTH+10, 145)) 
    text = small_font.render("(1)Idle, (2)Level, (3)Global, (4)Local, (5)Repair", True, RED)
    screen.blit(text, (WIDTH+10, 160))
    text = small_font.render("(E)xport terrain image", True, RED)
    screen.blit(text, (WIDTH+10, 175)) 
    text = small_font.render("Plot terrain (G)raph", True, RED)
    screen.blit(text, (WIDTH+10, 190))
    text = small_font.render("(D)amage terrain", True, RED)
    screen.blit(text, (WIDTH+10, 205))
    text = small_font.render("(Q)uit simulation", True, RED)
    screen.blit(text, (WIDTH+10, 220))
    text = small_font.render("(Space): Toggle Pause", True, RED)
    screen.blit(text, (WIDTH+10, 235))     
    text = font.render("Agent Vision Template: ", True, RED)
    screen.blit(text, (WIDTH+10, WIDTH-35)) 
    text = font.render("Individual Agent Views: ", True, BLUE)
    screen.blit(text, (WIDTH+50, WIDTH + 5))        
    pygame.draw.line(screen, (0, 255, 0), (WIDTH, 0), (WIDTH, 2*WIDTH), 2)
    pygame.draw.line(screen, (0, 255, 0), (0, WIDTH), (WIDTH*2, WIDTH), 2)

''' Utility Functions for data logging '''
def plot_graph(two_d_array, z_lim=None):
    # Note: this requires use of an IDE that can plot matplotlib graphs
    # Create a 2D array for axes
    dim1 = two_d_array.shape[0]
    dim2 = two_d_array.shape[1]
    x = np.linspace(1, dim1, dim1)
    y = np.linspace(1, dim2, dim2)
    X, Y = np.meshgrid(x, y)                    
    # Create a figure
    fig = plt.figure(dpi=400)                   
    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    if z_lim is not None:
        ax.set_zlim(z_lim)
    # Plot a surface
    ax.plot_surface(X, Y, two_d_array, cmap='coolwarm') 
    # Remove numbering from axes
    ax.set_xticks([])
    ax.set_yticks([])    
    
    ax.view_init(elev=40, azim=120)    
                    
    # Show the plot
    plt.show()    


def export_terrain(terrain_map):
    N = 2     
    scale_up = np.repeat(np.repeat(terrain_map, N, axis=0), N, axis=1)
    image = pygame.surfarray.make_surface(scale_up)
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string
    # This will create a string like '2023-05-31_16-30-00'
    filename = now.strftime('%Y-%m-%d_%H-%M-%S')
    # Use the string as part of the filename
    pygame.image.save(image, f'paper_results/{filename}.png') 
    

''' Main function including simulation loop '''
def main():
    # For debugging: set random seeds
    if DEBUG:
        random.seed(1)
        np.random.seed(1)   
        
    # Initialise simulation environment
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 50)
    pygame.init()
    clock = pygame.time.Clock()
    running = True
    paused = True
    dt = 0
    last_click_time = 0    
    dt = 1/FPS
    first_iteration = True
    last_image_export = -100
    
    # Initialise things for display
    screen = pygame.display.set_mode((WIDTH*2, WIDTH*2))
    font = pygame.font.Font(None, 24)   
    small_font = pygame.font.Font(None, 20)      
    
    # Initialise terrain, agents, pheremone and template
    template, global_template, gradient_template = template_configurator_UI(screen, font, clock)
    
    pheremone_map = initialize_pheremone(global_template, gradient_template)
    
    terrain_map = initialize_terrain(pheremone_map)
    agents = initialize_agents()    
    template_image = make_template_image(template)
    template_fft = prepare_template(template)
    
    agent_ph = False     
    show_agents = True           
    
    while running:
        # Limit framerate to target FPS
        actual_dt = clock.tick(FPS)/1000
        time_now = pygame.time.get_ticks()          
                    
        # Draw the bottom layer of the screen (terrain map, pheremone map and blanking other squares)        
        draw_screen_background(screen, terrain_map, pheremone_map)
                
        # Apply pheremone decay to map
        pheremone_map = decay_pheremone(pheremone_map)
               
        # Loop through agents, applying their behaviour
        for index, agent in enumerate(agents):            
                      
            agent, terrain_map, pheremone_map = update_agent_and_terrain(agent, dt, terrain_map, template_fft, pheremone_map, agent_ph, first_iteration)
            
            if show_agents:
                draw_agent(screen, agent, agent_ph)
            # Optionally, draw the individual agent views
            if SHOW_VIEWS:
                x_index = index % 10 
                y_index = index // 10
                draw_subscreen(screen, agent.view, WIDTH + x_index*SUB_WIDTH, 30 + WIDTH + y_index*SUB_WIDTH)    
        
        # This is a kludge, I don't love it. But it's a solution to allowing you to turn off the agent pheremone deposition from the start
        # This goes with the boolean operation in the function call to update_agent_and_terrain above
        if first_iteration:
            first_iteration = False
        # Draw the top layer of the screen, then flip it to show
        draw_screen_overlay(screen, template_image, font, small_font, actual_dt, agent_ph)
        pygame.display.flip() 
        
        if AUTO_EXPORT:
            if (time_now // 1000) - last_image_export > EXPORT_DT:
                last_image_export = time_now // 1000
                export_terrain(terrain_map)
                if PLOT_AND_IMAGE:
                    plot_graph(terrain_map[:, :, 1], [0, 255])
                    
            
        
        # Check for user input and act accordingly
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
                    paused = False
                
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:                
                    agent_ph = not agent_ph
                 
                if event.type == pygame.KEYDOWN and event.key == pygame.K_1:                
                     for agent in agents:                         
                         agent.build_state = 'idle'                         
                                         
                if event.type == pygame.KEYDOWN and event.key == pygame.K_2:                
                     for agent in agents:                         
                         agent.build_state = 'level'
                         
                if event.type == pygame.KEYDOWN and event.key == pygame.K_3:                
                    agent_ph = False 
                    for agent in agents:                         
                         agent.build_state = 'global'
                         
                if event.type == pygame.KEYDOWN and event.key == pygame.K_4: 
                    agent_ph = False
                    for agent in agents:                         
                          agent.build_state = 'local'
                          
                if event.type == pygame.KEYDOWN and event.key == pygame.K_5: 
                    agent_ph = False
                    for agent in agents:                         
                          agent.build_state = 'repair'                          
                          
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:                
                    paused = not paused
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:                
                    show_agents = not show_agents
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    export_terrain(terrain_map) 
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    plot_graph(terrain_map[:, :, 1], [0, 255])
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    terrain_map = damage_terrain(terrain_map)
                    
                                   
                if event.type == pygame.MOUSEBUTTONDOWN and (time_now - last_click_time) > 250:
                    last_click_time = time_now
                    mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
                    mouse_button = event.button
                    if mouse_pos_x < WIDTH:
                        mouse_pos_y = mouse_pos_y % WIDTH
                        # Manually add build pheremone
                        if mouse_button == 1:                            
                            pheremone_map = deposit_pheremone(mouse_pos_x, mouse_pos_y, pheremone_map, BUILD_PH, True, True, True)
                        # Manually remove build pheremone
                        elif mouse_button == 3:                            
                            pheremone_map = deposit_pheremone(mouse_pos_x, mouse_pos_y, pheremone_map, BUILD_PH, True, True, False)
                        
            if not paused:
                break        
                        
    
    # Shutdown
    pygame.quit()
    sys.exit()

# Call main function
if __name__ == "__main__":
    main()
    
