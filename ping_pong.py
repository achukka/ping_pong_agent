
# coding: utf-8

# In[17]:

# Library for writing games in python
import pygame # Library for making UI for games in python
import random # Used for defining directions UI for the ball


# # Using DQN ( Deep Q Learning) 
# 1. A Convolutional Neural Network reads the pixel data.
# 2. Q-Learning : Network learns by trial and error by maximizing actions based on agent's reward
# 3. Policy - Maps State to Action
# 4. Network learns by experience using past policies)

# In[18]:

# Variables for the game
frames_per_second = 60


# In[19]:

# Window size (The window is a rectangle)
window_width = 400
window_height = 400


# In[20]:

# Paddle size (The paddle is a rectangle)
paddle_width = 10
paddle_height = 60
# Offset from the window
paddle_buffer = 10


# In[21]:

# Ball size (The ball is a rectangle)
ball_width = 10
ball_height = 10


# In[22]:

# Paddle speed and ball speeds
paddle_velocity = 2
ball_x_velocity = 3
ball_y_velocity = 2


# In[29]:

# Colors for the paddle and ball
red = (255, 0, 0) # Evil Agent
white = (255, 255, 255) # Ball
green = (0, 255, 0) # RL Agent
black = (0, 0, 0) # Screen


# In[24]:

# initialize the screen
screen = pygame.display.set_mode((window_width, window_height))


# In[25]:

# Draw the ball
def draw_ball(ball_x_position, ball_y_position):
    ball = pygame.Rect(ball_x_position, ball_y_position, ball_width, ball_height)
    pygame.draw.rect(screen , white, ball)


# In[27]:

# Draw the first paddle - RL Agent
def draw_first_paddle(first_paddle_y_position):
    # Create a rectangle for the first agent
    first_paddle = pygame.Rect(paddle_buffer, first_paddle_y_position, paddle_width, paddle_height)
    pygame.draw.rect(screen, green, first_paddle)


# In[26]:

# Draw the second paddle - Evil Agent
def draw_second_paddle(second_paddle_y_position):
    # Create a rectangle for the second agent
    second_paddle = pygame.Rect(window_width - paddle_buffer - paddle_width , 
                          second_paddle_y_position, paddle_width, paddle_height)
    pygame.draw.rect(screen, red, second_paddle)


# In[11]:

# Update the ball and the paddles
def update_ball(first_paddle_y_position, second_paddle_y_position, ball_x_position, 
                ball_y_position, ball_x_direction, ball_y_direction):
    # First update the ball co-ordinates
    ball_x_position = ball_x_position + ball_x_direction * ball_x_velocity
    ball_y_position = ball_y_position + ball_y_direction * ball_y_velocity
    
    # Start with zero score
    score = 0
    
    # Check for collision, if the ball hits the left side then switch direction
    if(ball_x_position <= paddle_buffer+ paddle_width 
       and ball_y_position + ball_height >= first_paddle_y_position
       and ball_y_position - ball_height <= first_paddle_y_position + paddle_height):
        ball_x_direction = 1
    elif(ball_x_position <=0):
        # This should not happen, so return the negative the score
        ball_x_direction = 1
        score = -1
        return [score, first_paddle_y_position, second_paddle_y_position, 
                ball_x_position, ball_y_position, ball_x_direction, ball_y_direction]
    
    # If the ball hits the other (right) side of the window
    if (ball_x_position >= window_width - paddle_buffer - paddle_width 
        and ball_y_position + ball_height >= second_paddle_y_position
        and ball_y_position - ball_height <= second_paddle_y_position + paddle_height):
        ball_x_direction = -1 #switch directions
    elif (ball_x_position >= window_width - ball_width):
        ball_x_direction = -1
        score = 1 # Positive score for us
        return [score, first_paddle_y_position, second_paddle_y_position, 
                ball_x_position, ball_y_position, ball_x_direction, ball_y_direction]
    
    # Check for the top side and the bottom side
    if ball_y_position <=0: # top side
        ball_y_position = 0
        ball_y_direction = 1 # Switch Direction
    elif(ball_y_position >= window_height - ball_height): # bottom side
        ball_y_position = window_height - ball_height
        ball_y_direction = -1 # Switch Direction
    return [score, first_paddle_y_position, second_paddle_y_position, 
                ball_x_position, ball_y_position, ball_x_direction, ball_y_direction]    


# In[12]:

# Update the first paddle (Our agent)
def update_first_paddle(action, first_paddle_y_position):
    # The actions are up and down, up - [_, 1,0], down - [_, 0,1]
    #Move UP
    if(action[1] == 1):
        first_paddle_y_position = first_paddle_y_position - paddle_velocity
    # Move Down
    if(action[2]==1):
        first_paddle_y_position = first_paddle_y_position + paddle_velocity
    
    # Check the window limits
    if(first_paddle_y_position<0): # top
        first_paddle_y_position = 0
    if(first_paddle_y_position > window_height - paddle_height):
        first_paddle_y_position = window_height - paddle_height
    return first_paddle_y_position


# In[13]:

# Update the second paddle (Evil agent)
def update_second_paddle(second_paddle_y_position, ball_y_position):
    # The actions are up and down, up - [_, 1,0], down - [_, 0,1]
    #Move down if the ball is in upper half
    if(second_paddle_y_position + paddle_height/2 < ball_y_position + ball_height/2 ):
        second_paddle_y_position = second_paddle_y_position + paddle_velocity
    # Move up if the ball is in the lower half
    if(second_paddle_y_position + paddle_height/2 > ball_y_position + ball_height/2 ):
        second_paddle_y_position = second_paddle_y_position - paddle_velocity
    
    # Check the window limits
    if(second_paddle_y_position<0):
        second_paddle_y_position = 0
    if(second_paddle_y_position > window_height - paddle_height):
        second_paddle_y_position = window_height - paddle_height
    return second_paddle_y_position


# In[28]:

# Make a class for the pong game
class PongGame:
    # Initialize the variables
    def __init__(self):
        # Random Number for the initial direciton of the ball
        num = random.randint(0, 20)
        
        # keep track of the score
        self.score = 0
        
        # Initialize the postions of the two paddles
        self.first_paddle_y_position = window_height/2 - paddle_height/2
        self.second_paddle_y_position = window_height/2 - paddle_height/2
        
        # Ball Directions (x,y)
        self.ball_x_direction = 1
        self.ball_y_direction = 1
        
        # Starting point
        self.ball_x_position = window_width/2 - ball_width/2
        
        # Randomize the ball direction
        if 0 <= num < 5:
            self.ball_x_direction = 1
            self.ball_y_direction = 1
        elif 5 <= num < 10:
            self.ball_x_direction = -1
            self.ball_y_direction = 1
        elif 10 <= num < 15:
            self.ball_x_direction = -1
            self.ball_y_direction = -1
        else:
            self.ball_x_direction = 1
            self.ball_y_direction = -1
        
        # Again generate a random number for the ball initial y position
        num = random.randint(0, 20)
        
        # Y postion of the ball
        self.ball_y_position = num*(window_height - ball_height)/19                        
    
    # Get the current frame
    def get_current_frame(self):
        # Call the event queue for each frame
        pygame.event.pump()
        
        # Make background black
        screen.fill(black)
        
        # Draw the paddles
        draw_first_paddle(self.first_paddle_y_position)
        draw_second_paddle(self.second_paddle_y_position)
        
        # Draw the ball
        draw_ball(self.ball_x_position, self.ball_y_position)
        
        # Get pixels (copies the pixels from the surface to a 3d array)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        
        # Update the window
        pygame.display.flip()
        
        # Return the screen data
        return image_data
    
    # Get the next frame, based on the current frame and the action
    def get_next_frame(self, action):
        # Call the event queue for each frame
        pygame.event.pump()
        
        score = 0
        # Make background black
        screen.fill(black)
        
        # Update the paddle positions
        self.first_paddle_y_position = update_first_paddle(action, 
                                                           self.first_paddle_y_position)
        self.second_paddle_y_position = update_second_paddle(self.second_paddle_y_position,
                                                             self.ball_y_position)
        
        # Draw the paddles
        draw_first_paddle(self.first_paddle_y_position)
        draw_second_paddle(self.second_paddle_y_position)
        
        # Update the ball positions and other variables
        [ score, 
          self.first_paddle_y_position, self.second_paddle_y_position, 
          self.ball_x_position, self.ball_y_position, 
          self.ball_x_direction, self.ball_y_direction]   = update_ball(self.first_paddle_y_position, 
                                               self.second_paddle_y_position, 
                                               self.ball_x_position, self.ball_y_position, 
                                               self.ball_x_direction, self.ball_y_direction)
        
        # Draw the ball
        draw_ball(self.ball_x_position, self.ball_y_position)
        
        # Get pixels
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        
        # Update the window
        pygame.display.flip()
        
        # update the total score
        self.score += score
        # Return the score and image data
        return [score, image_data]


# In[15]:

# Qlearning  - CNN reads pixel data and we maximize the reward using DQN


# In[ ]:



