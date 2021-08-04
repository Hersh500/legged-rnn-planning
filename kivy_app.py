from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
import hopper
import numpy as np

dt = 0.01

class RootWidget(FloatLayout):
    def __init__(self, **kwargs):
        # make sure we aren't overriding any important functionality
        super(RootWidget, self).__init__(**kwargs)
        
        # simulation stuff
        self.constants = hopper.Constants()
        self.robot = hopper.Hopper(hopper.Constants())
        self.in_stance = False
        self.robot_state = [0, 1.2, 0, 0, 0, np.pi/2]
        self.foot_pos = [0, 1.2 - self.constants.L]
        self.terrain_func = lambda x: 0
        self.vel_des = 0
        self.cond = False
    
        self.robot_base = Ellipse(size = (10, 10), pos = (0, 0))

    def sim_to_plot(self, x, y):
        return x, y

    def reset(self):
        self.robot_state = [0, 1.2, 0, 0, 0, np.pi/2]
        self.foot_pos = [0, 1.2 - self.constants.L]
        self.in_stance = False
        self.vel_des = 0
        self.cond = False
        return
    
    def update(self):
        if self.in_stance:
            # do the stance dynamics
            derivs = np.array(self.robot.stanceDynamics(0, self.robot_state))
            self.robot_state = list(np.array(self.robot_state) + dt * derivs)
            if robot.checkStanceFailure(self.robot_state, self.foot_pos, self.terrain_func):
                self.reset()
            if self.robot_state[2] > self.robot.constants.L:
                # convert to flight state  
                self.robot_state = self.robot.stanceToFlight(self.robot_state, self.foot_pos)
                self.in_stance = False
                self.cond = False
        else:
            # do the flight dynamics
            cur_y_vel = self.robot_state[3]
            derivs = np.array(self.robot.flightDynamics(0, self.robot_state))
            self.robot_state = list(np.array(self.robot_state) + dt * derivs)
            # we have hit the apex
            if self.robot_state <= 0 and cur_y_vel >= 0:
                self.robot_state[5] = u
                self.cond = True
            self.foot_pos = self.robot.getFootXYInFlight(self.robot_state)
            if self.cond and self.foot_pos[1] <= self.terrain_func(self.foot_pos[0]):
                self.robot_state = self.robot.flightToStance(self.robot_state)
                self.in_stance = True
            if robot.checkFlightCollision(self.robot_state, self.terrain_func):
                self.reset()
        return


# the main loop should be:
# read the slider, set vel_des
# update the hopper state

class MainApp(App):

    def build(self):
        self.root = root = RootWidget()
        root.bind(size=self._update_rect, pos=self._update_rect)

        with root.canvas:
            Color(0.5, 0.5, 0.5, 1)  # green; colors range from 0-1 not 0-255
            self.rect = Rectangle(size=(1, 1), pos=(0, 0))
        return root

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

if __name__ == '__main__':
    MainApp().run()
