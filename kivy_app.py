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
    
    def sim_to_plot(self, x, y):
        return x, y

    def reset(self):
        self.robot_state = [0, 1.2, 0, 0, 0, np.pi/2]
        self.foot_pos = [0, 1.2 - self.constants.L]
        self.in_stance = False
        return
    
    def update(self):
        if self.in_stance:
            # do the stance dynamics
        else:
            # do the flight dynamics
        return


class MainApp(App):

    def build(self):
        self.root = root = RootWidget()
        root.bind(size=self._update_rect, pos=self._update_rect)

        with root.canvas.before:
            Color(0, 0, 0, 1)  # green; colors range from 0-1 not 0-255
            self.rect = Rectangle(size=root.size, pos=root.pos)
        return root

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

if __name__ == '__main__':
    MainApp().run()
