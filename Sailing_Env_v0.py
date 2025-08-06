import random
import pandas as pd
import math
import pygame as pg
import numpy as np

from Polar_parser import parse

class SailingWorld:
    def __init__(self, delta_t=2, fps=5, width=20, length=80, cell_size=None, max_steps = None, polar_source="A35.csv", polar_seperation=";"):
        """
        Purpose:
            Initializes the sailing simulation environment, including grid size, wind model, polar data, simulation speed, and resolution.

        Parameters:
            •	delta_t (int): simulation step length (seconds)
            •	fps (int): frames per second for rendering
            •	width, length (int): grid dimensions
            •	cell_size (float or None): rendering resolution; auto-calculated if None
            •	max_steps (int or None): max steps per episode; auto-calculated if None
            •	polar_source (str): file name of the polar diagram (CSV)
            •	polar_seperation (str): separator used in the polar CSV (e.g., ";", ",")

        Behavior:
            1.	Loads polar diagram (speed depending on wind angle + strength)
            2.	Initializes wind, boat state, and environment parameters
            3.	Calculates a default max_steps based on estimated boat speed and course distance if none given
            4.	Sets cell_size if not given (so render fits ~600x600px)
            5.	Raises error if grid is too small to be playable

        Side Effects:
            •	Creates internal environment state like visited_points, action_space, etc.
            •	Sets up constants for rendering and simulation"""
        self.polar = parse(polar_source,polar_seperation)

        self.running = False
        self.truncated = True

        self.number_of_steps_per_ep = 0
        self.window = None
        self.action_space = [0,45,90,135,180,-135,-90,-45] # In compass notation not mathematic notation!
        self.action_space_size = len(self.action_space)
        self.wind_angle, self.wind_speed = 0,10 # speed in m/s

        self.length = length
        self.width = width
        self.cell_size = cell_size

        self.delta_t = delta_t
        self.fps = fps
        self.visited_points = []
        self.action = None

        #rough estimation of max_steps
        v_45 = self.get_speed(45)
        v_m45 = self.get_speed(-45)
        v = 0.5 * (v_45 + v_m45)    #Vector in convex hull if going "straigt" to the finish line
        d = 0.97 * abs(self.length) # distance to the goal line
        buffer = 5
        self.max_steps = max_steps if max_steps is not None else buffer*int(d/(v*self.delta_t)) # distance / speed = time, time / delta_t = steps

        # Setting cell_size to be roughly 600x600 pixels
        self.cell_size = 0.5*(600/self.width+ 600/self.length) if cell_size is None else cell_size

        # Nutzerparameter
        if length < 10:
            raise ValueError("length to short, can't sail here")
        if width < 5:
            raise ValueError("Width to short, can't sail here")
    
    def reset(self):
        """
        Purpose:
            Resets the environment for a new episode by reinitializing the boat’s position and internal state.

        Behavior:
            1.	Places the boat at a random position:
                •	Horizontally between 30%–70% of the width
                •	Vertically in the bottom 10% of the map
            2.	Resets:
                •	Wind and speed variables
                •	Step counter
                •	Flags running (to True) and truncated (to False)
                •   Some stuff for rendering

        Returns:
            •	tuple: (rounded x-position, wind angle) — the initial state for the agent"""
        
        self.boat_position = [random.uniform(0.3*self.width, 0.7*self.width), random.uniform(0, 0.1*self.length)]
        self.prev_boat_pos = self.boat_position
        self.prev_boat_position = self.boat_position.copy()
        self.running = True
        self.truncated = False
        self.number_of_steps_per_ep = 0
        self.speed = 0
        self.visited_points = []
        return (round(self.boat_position[0],0), self.wind_angle)
    
    def get_wind_data(self):
        self.wind_angle, self.wind_speed = 0, 10 # Placeholder for simulated wind change/ wind data retrieval
  
    def get_speed(self, boat_heading_angle):
        """
        Purpose:
            Looks up the boat’s speed based on the current wind angle and strength using the polar diagram.

        Parameters:
            •	boat_heading_angle (int): direction the boat is trying to sail (in degrees, compass-style)

        Behavior:
            1.	Computes True Wind Angle (TWA) = difference between wind and heading (modulo 360)
            2.	Converts it to the symmetric angle between 0° and 180°
            3.	Finds the row in the polar table based on TWA in 45° steps (e.g. 0–44 → row 0)
            4.	Finds the column in the polar table based on wind speed in 2-knot steps
            5.	Returns the value from the polar table

        Returns:
            •	float: Boat speed in m/s for the given heading under current wind conditions"""
        # Calcluate Angle between Wind an Boat
        theta = (boat_heading_angle - self.wind_angle) % 360
        if theta > 180:
            theta = 360 - theta

        # Get row index (TWA in 45°-Steps)
        row = theta // 45

        # Get column index (TWS in 2-knots-steps)
        column = self.wind_speed // 2

        return self.polar[row][column]
    
    def step(self, action):
        """
        Purpose:
            Executes one simulation step: moves the boat based on the selected action, updates state, checks for terminal conditions, and calculates reward.

        Parameters:
            •	action (int): sailing direction in degrees (must be in action_space)

        Behavior:
            1.	Saves previous position
            2.	Calculates new boat position using polar speed and action angle
            3.	Checks for:
                •	Out-of-bounds → ends episode
                •	Goal line reached → ends episode
                •	Max steps exceeded → ends episode
            4.	Updates wind conditions (placeholder)
            5.	Increments step counter

        Returns:
            •	tuple (rounded x-position, wind angle)
            •	terminated (bool): if episode ended
            •	reward (float)
            •	truncated (bool): if max steps reached"""
        
        self.action = action
        self.prev_boat_pos = self.boat_position
        # Calculating where the boat goes next
        boat_heading_angle = action
        self.speed = self.get_speed(boat_heading_angle)
        self.boat_position[0] += self.delta_t * self.speed * math.cos(math.radians(90-action))
        self.boat_position[1] += self.delta_t * self.speed * math.sin(math.radians(90-action))

        # Check if the boat is out of bounds
        if self.boat_position[0] < 0 or self.boat_position[0] > self.width or \
           self.boat_position[1] < 0:
            reward = -500 # Extreme Punishment for going out of bounds 
            self.running = False
            return (round(self.boat_position[0],0), self.wind_angle), not self.running, reward, self.truncated
        
        #Chekk if the boat reached the goal line
        if self.boat_position[1] >= 0.97*self.length:
            reward = 100
            self.running = False
        else:
            reward = -1  # Small penalty for each step to encourage faster sailing

        if self.number_of_steps_per_ep >= self.max_steps:
            self.truncated = True
            reward = -100 # Penalty for exceeding max steps

        self.number_of_steps_per_ep += 1

        self.get_wind_data()

        return (round(self.boat_position[0],0), self.wind_angle), not self.running, reward, self.truncated
    
    def render_human(self, reward=None):
        """
        Purpose:
            Renders the current simulation step in a Pygame window — showing boat, wind, reward, path, and goal line.

        Parameters:
            •	reward (float, optional): current reward to display in the UI

        Behavior:
            1.	Initializes Pygame window and font if not already done
            2.	Computes num_frames for smooth interpolation between last and current position
            3.	Draws:
                •	Water background
                •	Boat (as red circle)
                •	Wind info (angle + speed)
                •	Current action in degrees
                •	Reward (if given)
                •	Path of the boat
                •	Goal line (top of screen)
        """
        if not hasattr(self, "prev_boat_position"):
            self.prev_boat_position = self.boat_position.copy()
        if not hasattr(self, 'window') or self.window is None:
            pg.init()
            self.window = pg.display.set_mode((self.width * self.cell_size, self.length * self.cell_size))
            pg.display.set_caption("SailingWorld")
        if not hasattr(self, 'visited_points'):
            self.visited_points = [prev_boat_pos.copy()]

        BLUE = (70, 130, 180) # Water color
        BOAT_COLOR = (175, 0, 0)
        GOAL_COLOR = (129,94,94) # Gold color for the goal line

        self.window.fill(BLUE)

        num_frames = int(self.delta_t * self.fps)
        prev_boat_pos = np.array(self.prev_boat_position)
        boat_pos = np.array(self.boat_position)

        for frame_idx in range(num_frames):
            interp_pos = prev_boat_pos + (boat_pos - prev_boat_pos) * (frame_idx / num_frames)
            self.visited_points.append(interp_pos.copy())
            x = int(interp_pos[0] * self.cell_size)
            y = self.length*self.cell_size - int(interp_pos[1] * self.cell_size)

            self.window.fill(BLUE)

            pg.draw.circle(self.window, BOAT_COLOR, (x, y), 10)

            font = pg.font.SysFont(None, 24)
            action_text = font.render(f"Action: {self.action}°", True, (255, 255, 255))
            self.window.blit(action_text, (10, 0.1*self.length*self.cell_size))

            if reward is not None:
                reward_text = font.render(f"Reward: {reward:.2f}", True, (255, 255, 255))
                self.window.blit(reward_text, (self.width*self.cell_size-150, 0.05*self.length*self.cell_size))

            wind_text = font.render(f"Wind: {self.wind_angle}° at {self.wind_speed} m/s", True, (255, 255, 255))
            self.window.blit(wind_text, (10, 0.05*self.length*self.cell_size))

            if len(self.visited_points) > 2:
                cell_points = [(x * self.cell_size, self.length*self.cell_size - y * self.cell_size) for x, y in self.visited_points]
                pg.draw.lines(self.window, BOAT_COLOR, False, cell_points, 2)

            pg.draw.line(self.window, GOAL_COLOR, (0, 0.03*self.length*self.cell_size), (self.width * self.cell_size,0.03*self.length*self.cell_size), 2)
            goal_text = font.render("Goal Line", True, GOAL_COLOR)
            self.window.blit(goal_text, (0.5*self.width * self.cell_size-50, 0.03*self.length*self.cell_size - 15))

            pg.display.flip()
            pg.time.delay(int(1000/self.fps))

        self.prev_boat_position = self.boat_position.copy()

    def did_window_got_closed(self):
        if self.window is not None:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    self.window = None
                    print("Simulation over")

    def close(self):
        print("Done!")
        self.visited_points = []
        self.window = None
        self.running = False

    def test_env(self):
        self.reset()
        while self.running:
            action = random.choice(self.action_space)
            print(action)
            self.step(action)
            self.render_human()
        self.did_window_got_closed()
        self.close()

if __name__ == "__main__":
    print("Welcome to my Sailing Environment Version 0!")
    Env = SailingWorld(delta_t=2, width=30, length=60, cell_size=15,fps =60)

    #while input("Do you wnat to test the Env? (y/n)").lower() == "y":
    Env.test_env()