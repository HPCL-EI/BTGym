# self状态:互斥状态映射
mutually_exclusive_states = {
    'IsLeftHandEmpty': 'IsLeftHolding',
    'IsLeftHolding': 'IsLeftHandEmpty',
    'IsRightHandEmpty': 'IsRightHolding',
    'IsRightHolding': 'IsRightHandEmpty',

    'IsSitting': 'IsStanding',
    'IsStanding': 'IsSitting',

}

# 物体状态: Mapping from state to anti-state
state_to_opposite = {
    'IsOpen': 'IsClose',
    'IsClose': 'IsOpen',
    'IsSwitchedOff': 'IsSwitchedOn',
    'IsSwitchedOn': 'IsSwitchedOff',
    'IsPlugged': 'IsUnplugged',
}


import re
def extract_argument(state):
    match = re.search(r'\((.*?)\)', state)
    if match:
        return match.group(1)
    return None


def update_state(c, state_dic):
    for state, opposite in state_to_opposite.items():
        if state in c:
            obj = extract_argument(c)
            if obj in state_dic:
                # 如果对象已经有一个反状态，返回False表示冲突
                if state_dic[obj] == opposite:
                    return False
            # 更新状态字典
            state_dic[obj] = state
            break
    return True


# conds = {'IsRightHolding(self,wineglass)', 'IsStanding(self)', 'IsNear(self,candle)',
#          'IsRightHandEmpty(self)', 'IsOn(plate,kitchentable)'}
conds = {'IsRightHandEmpty(self)','IsRightHolding(self,wineglass)','IsStanding(self)', 'IsNear(self,candle)',
         'IsOn(plate,kitchentable)'}
obj_state_dic = {}
self_state_dic={}
self_state_dic['self']=set()
is_near = False
FF=False
for c in conds:
    if "IsNear" in c and is_near:
        FF= True
    elif "IsNear" in c:
        is_near = True
    # Cannot be updated, the value already exists in the past
    if not update_state(c, obj_state_dic):
        FF= True

    # Check for mutually exclusive states without obj
    for state, opposite in mutually_exclusive_states.items():
        if state in c and opposite in self_state_dic['self']:
            FF= True
        elif state in c:
            self_state_dic['self'].add(state)
            break
# 检查是否同时具有 'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'
required_states = {'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
if all(state in conds for state in required_states):
    FF= True

print(FF)