from pyuvm import *
from pyuvm_components.agent import NVDLA_Agent
from pyuvm_components.scoreboard import NVDLA_Scoreboard


class NVDLA_Env(uvm_env):
    def build_phase(self):
        self.agent = NVDLA_Agent("NVDLA_Agent", self)
        self.scoreboard = NVDLA_Scoreboard("NVDLA_Scoreboard", self)

    def connect_phase(self):
        self.agent.agt_ap.connect(self.scoreboard.score_export)