# Generate video for a program. Make sure you have the executable open

import time
from btgym.vh.simulation.unity_simulator.comm_unity import UnityCommunication


# script1 = ['<char0> [Walk] <tv> (1)','<char0> [switchon] <tv> (1)'] # Add here your script


script = ['<char0> [Walk] <tv> (1)','<char0> [switchon] <tv> (1)', '<char0> [Walk] <sofa> (1)', '<char0> [Sit] <sofa> (1)', '<char0> [Watch] <tv> (1)']

print('Starting Unity...')
comm = UnityCommunication()

print('Starting scene...')
comm.reset()
comm.add_character('Chars/Female1')

print('Generating video...')
for script_instruction in script:
    comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["PERSON_FROM_BACK"],find_solution=True)
# comm.render_script(script, recording=True,frame_rate=10,camera_mode=["PERSON_FROM_BACK"],find_solution=True)

# comm.render_script(script1, recording=True, find_solution=True)
# # time.sleep(2)
# comm.render_script(script2, recording=True, find_solution=True)

print('Generated, find video in simulation/unity_simulator/output/')
