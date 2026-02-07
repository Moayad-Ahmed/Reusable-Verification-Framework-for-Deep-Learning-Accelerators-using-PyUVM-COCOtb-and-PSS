from pyuvm import *
from agent import My_Agent
from scoreboard import My_Scoreboard

class My_Env(uvm_env):

    def build_phase(self):
        self.My_Agent = My_Agent("My_Agent", self)
        self.My_Scoreboard = My_Scoreboard("My_Scoreboard", self)
   
    def connect_phase(self):
        self.My_Agent.agt_ap.connect(self.My_Scoreboard.score_export)