from pyuvm import *
from pyuvm_components.data_agent import DataAgent
from pyuvm_components.csb_agent import CsbAgent
from pyuvm_components.scoreboard import NVDLA_Scoreboard


class NVDLA_Env(uvm_env):

    def build_phase(self):
        self.data_agent = DataAgent("DataAgent", self)
        self.csb_agent  = CsbAgent("CsbAgent", self)
        self.scoreboard = NVDLA_Scoreboard("NVDLA_Scoreboard", self)

    def connect_phase(self):
        # CsbAgent monitor writes completed transactions (with actual_output_data)
        # to the scoreboard analysis FIFO for comparison against expected data
        self.csb_agent.agt_ap.connect(self.scoreboard.score_export)
