import time

from btgym.envs.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.utils import ROOT_PATH

from btgym.agent import Agent
import subprocess


class WatchTVEnv(object):
    agent_num = 1

    def __init__(self):
        self.comm = UnityCommunication()

        # launch simulator
        file_name = f'{ROOT_PATH}/../simulators/virtualhome/windows/VirtualHome.exe'
        self.simulator_process = subprocess.Popen(file_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.create_agents()
        self.create_behavior_lib()


    def run_script(self,script):
        self.comm.render_script(script, recording=True,skip_animation=False, frame_rate=10, camera_mode=["PERSON_FROM_BACK"],
                               find_solution=True)

    def reset(self):
        simulator_launched = False
        while not simulator_launched:
            try:
                self.comm.reset()
                simulator_launched = True
            except:
                pass

        self.comm.add_character('Chars/Female1')

    def step(self):
        for agent in self.agents:
            agent.bt.tick()
        return self.is_finished()

    def close(self):
        time.sleep(1)
        self.simulator_process.terminate()

    def is_finished(self):
        if "IsWatching(self,tv)" in self.agents[0].condition_set:
            return True
        else:
            return False


    def create_agents(self):

        agent = Agent()
        agent.env = self
        self.agents = [agent]


    def create_behavior_lib(self):
        behavior_lib_path = f"{ROOT_PATH}/envs/virtualhome/exec_lib"

        self.behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
