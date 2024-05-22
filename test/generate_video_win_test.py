# Generate video for a program. Make sure you have the executable open

from btgym.envs.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from btgym.envs.virtualhome.tools import add_object_to_scene

# script1 = ['<char0> [Walk] <tv> (1)','<char0> [switchon] <tv> (1)'] # Add here your script

import subprocess

# file_name =  'D:\Workspace\BaiduSyncdisk\CXL_Storage\Code\windows_exec.v2.2.4\VirtualHome.exe'
file_name = 'D:\worktable\BTGym\simulators\\virtualhome\windows\VirtualHome.exe'

# 启动 exe 文件
process = subprocess.Popen(file_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# 准备早餐
script =  [
    # '<char0> [Walk] <chicken> (1)',
    # '<char0> [Grab] <chicken> (1)',
    # '<char0> [Walk] <stove> (1)',
    # '<char0> [Open] <stove> (1)',
    # '<char0> [PutIn]  <chicken>  (1) <stove> (1)',
    # '<char0> [Close] <stove> (1)',
    # '<char0> [SwitchOn] <stove> (1)',

    '<char0> [Walk] <apple> (1)',
    '<char0> [Grab] <apple> (1)',
    '<char0> [Walk] <fridge> (1)',
    '<char0> [Open] <fridge> (1)',
    '<char0> [PutIn]  <apple>  (1) <fridge> (1)',
    '<char0> [Close] <fridge> (1)',
    ]

print('Starting Unity...')
comm = UnityCommunication()

print('Starting scene...')
scene_name=0
simulator_launched = False
while not simulator_launched:
    try:
        comm.reset(18) # 0-49   40
        simulator_launched = True
    except:
        pass


_, env_g = comm.environment_graph()
# del_no = None
for no in env_g['nodes']:
    if no["class_name"]=="mug":
        print(no)
        break



# comm.add_character('Chars/Female1')
# comm.add_character('Chars/Male1',initial_room="kitchen")
comm.add_character('Chars/Male1')
#  Chars/Male6  micai   3
#  2 22  24 微波炉可以打开

print('Generating video...')
for script_instruction in script:
    # success, message = comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["FIRST_PERSON"],find_solution=True) #AUTO FIRST_PERSON
    # success, message = comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["AUTO"],find_solution=True) #AUTO FIRST_PERSON
    success, message = comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["FIRST_PERSON"],find_solution=True) #AUTO FIRST_PERSON

    # 检查指令是否成功执行            # PERSON_FROM_BACK
    if success:
        print(f"'Successfully: {script_instruction}'.")
    else:
        print(f"'Failed {script_instruction},{message}'.")


print('Generated, find video in simulation/unity_simulator/output/')
