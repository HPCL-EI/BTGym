import re
from dataset import read_dataset


def get_parts(_example):
    _Instruction = _example['Instruction']
    _Goals = _example['Goals']
    _Actions = _example['Actions']
    return _Instruction, _Goals, _Actions


def split_set(data_set, data_list):
    result = []
    temp = []
    for i, v in enumerate(data_list):
        if v in data_set:
            if i != 0:
                result.append(temp)
            temp = []
        temp.append(v)
    if temp:
        result.append(temp)
    return result


def split_string_by_underscores(s):
    # 计算下划线的数量
    underscore_count = s.count('_')
    # 根据下划线数量分割字符串
    if underscore_count == 1:
        # 一个下划线，分割成两部分
        return [s[:s.index('_')], s[s.index('_') + 1:]]
    elif underscore_count == 2:
        # 两个下划线，分割成三部分
        first_part, second_part = s.split('_', 1)  # 先按第一个下划线分割
        return [first_part, second_part[:second_part.index('_')], second_part[second_part.index('_') + 1:]]
    else:
        # 下划线数量不是1也不是2，抛出异常
        print(s)
        raise ValueError("Unexpected number of underscores in string")


Condition = {'IsNear', 'IsOn', 'IsIn', 'IsOpen', 'IsClose', 'IsSwitchedOn', 'IsSwitchedOff'}
Action = {'Walk', 'RightGrab', 'LeftGrab', 'RightPut', 'LeftPut', 'RightPutIn',
         'LeftPutIn', 'RightGrabFrom', 'LeftGrabFrom', 'Open', 'Close', 'SwitchOn', 'SwitchOff'}

SurfacePlaces = {"floor","bathroomcabinet","bathroomcounter","towelrack","rug", "plate", "tvstand", "nightstand", "kitchentable", "kitchencabinet", "kitchencounter", "fryingpan", "stove","oventray", "mat","tray","bookshelf", "desk", "cabinet", "chair", "bench","sofa", "bed", "mousemat","radio", "boardgame", "couch","table","filingcabinet","mousepad","bathtub"}
SittablePlaces = {"Bathtub", "bed", "Bench", "chair", "rug", "sofa", "toilet"}
CanOpenPlaces = {"bathroomcabinet", "book", "bookshelf", "box", "coffeepot", "cabinet", "closet", "clothespile", "coffeemaker", "cookingpot", "curtains", "desk", "dishwasher", "door", "folder", "fridge", "garbagecan", "hairproduct", "journal", "lotionbottle", "magazine", "microwave", "milk", "nightstand", "printer", "radio", "stove", "toilet", "toothpaste", "beer", "washingmachine"}
CanPutInPlaces = {"bathroomcabinet", "bookshelf", "box", "coffeepot", "cabinet", "closet", "clothespile","coffeemaker", "cookingpot", "desk", "dishwasher", "folder", "fridge", "garbagecan", "microwave", "nightstand", "printer", "stove", "toilet", "toothpaste", "washingmachine"}
Objects = {"apple", "sportsball", "bananas", "barsoap", "bellpepper", "boardgame", "book", "box","bread", "breadslice", "broom", "bucket", "carrot", "cat", "cards", "cellphone", "chinesefood","coffeepot", "crayons", "chair", "candle",  "chefknife", "chicken", "chocolatesyrup", "clock", "clothespants", "clothespile", "clothesshirt", "condimentbottle",  "condimentshaker","cookingpot", "candybar", "crackers", "cereal", "creamybuns", "chips", "cucumber", "cupcake", "cutleryknife",  "cutleryfork", "cutlets", "cuttingboard", "dishwashingliquid", "dishbowl","plate", "dustpan", "facecream", "folder",  "fryingpan", "globe", "glasses", "hairproduct", "hanger","juice", "journal", "keyboard", "lemon", "lime", "lotionbottle",  "kettle", "magazine", "milk","milkshake", "mincedmeat", "mouse", "mug", "napkin", "notes", "orange", "pancake", "paper",  "papertowel", "pear", "pen", "pie", "pillow", "plum", "potato", "poundcake", "pudding", "radio","remotecontrol", "rug", "salad", "salmon", "slippers", "washingsponge", "spoon", "sundae","teddybear", "toy", "toiletpaper", "tomato", "toothbrush",   "toothpaste", "towel", "towelrack","wallpictureframe", "wallphone", "waterglass", "watermelon", "whippedcream", "wineglass","alcohol", "beer", "wine"}
HasSwitchObjects = {"cellphone", "candle", "clock", "coffeemaker", "dishwasher", "fridge", "lightswitch", "kettle","microwave", "pc", "printer", "radio", "remotecontrol", "stove", "tv", "toaster", "walltv","wallphone", "computer", "washingmachine"}
Object = SurfacePlaces | SittablePlaces | Objects | CanOpenPlaces | CanPutInPlaces | HasSwitchObjects

dataset = read_dataset('E:/BTGym_yang/more_objects/dataset.txt')

# 逐行处理
for index, example in enumerate(dataset):
    # print(index, example)
    # if index != 16:
    #     continue
    Instruction, Goals, Actions = get_parts(example)
    print(Instruction, Goals, Actions)
    print()
    pattern = r"\w+"
    # instruction = re.findall(pattern, Instruction)
    goals = re.findall(pattern, Goals)
    actions = re.findall(pattern, Actions)

    goals = split_set(Condition, goals)
    print(goals)
    actions = split_set(Action, actions)
    print(actions)

    Goals_output = 'Goals:'
    Actions_output = 'Actions:'

    # todo: 输出1
    # Goals: IsOn_bananas_kitchentable & IsOn_milk_kitchentable & IsOn_cupcake_kitchentable
    # Actions: Walk_bananas, RightPut_bananas_kitchentable, Walk_milk, RightPut_milk_kitchentable,
    #          Walk_cupcake, RightPut_cupcake_kitchentable, Walk_toaster, SwitchOn_toaster, Walk_tv, SwitchOn_tv

    # for i, goal in enumerate(goals):
    #     a, b, c = goal[0], goal[1], ''
    #     if len(goal) == 3:
    #         c = goal[2]
    #     Goals_output += ' ' + str(a) + '_' + str(b) + str('_' + c if c != ''else '') + ' '
    #     if i != len(goals) - 1:
    #         Goals_output += '&'
    # print(Goals_output)
    # for i, action in enumerate(actions):
    #     a, b, c = action[0], action[1], ''
    #     if len(action) == 3:
    #         c = action[2]
    #     Actions_output += ' ' + str(a) + '_' + str(b) + str('_' + c if c != ''else '')
    #     if i != len(actions) - 1:
    #         Actions_output += ','
    # print(Actions_output)


    # todo：输出2
    # Instruction: Place the bananas, milk, and cupcake on the kitchentable, switch on the toaster and the tv.
    # Goals: [{'IsSwitchedOn(toaster)', 'IsOn(bananas,kitchentable)', 'IsOn(milk,kitchentable)',
    #          'IsOn(cupcake,kitchentable)', 'IsSwitchedOn(tv)'}]
    # Actions: ['Walk(bananas)', 'RightGrab(bananas)', 'Walk(milk)', 'LeftGrab(milk)',
    #           'Walk(kitchentable)', 'RightPut(bananas,kitchentable)',  'LeftPut(milk,kitchentable)',
    #           'Walk(cupcake)', 'RightGrab(cupcake)', 'Walk(kitchentable)', 'RightPut(cupcake,kitchentable)',
    #           'Walk(toaster)', 'SwitchOn(toaster)', 'Walk(tv)', 'SwitchOn(tv)']

    Goals_output += ' [{'

    if '_' in goals[0][0]:
        goals = goals[0]
        print(goals)
        goals = [split_string_by_underscores(item) for item in goals]
    if '_' in actions[0][0]:
        actions = actions[0]
        print(actions)
        actions = [split_string_by_underscores(item) for item in actions]



    for i, goal in enumerate(goals):
        print('--', goal)
        Goals_output += "'" + goal[0] + '(' + goal[1]
        if len(goal) == 3:
            Goals_output += ',' + goal[2]
        Goals_output += ')' + "'"
        if i != len(goals) - 1:
            Goals_output += ','
    Goals_output += '}]'
    print(Goals_output)
    Actions_output += ' ['

    for i, action in enumerate(actions):
        # print(action)
        Actions_output += "'" + action[0] + '(' + action[1]
        if len(action) == 3:
            Actions_output += ',' + action[2]
        Actions_output += ')' + "'"
        if i != len(actions) - 1:
            Actions_output += ','
    Actions_output += ']'
    print(Actions_output)

    with open('E:/BTGym_yang/more_objects/new_dataset.txt', 'a') as new_dataset:
        new_dataset.write(str(index+1) + '.' + '\n')
        new_dataset.write('Instruction:' + Instruction + '\n')
        new_dataset.write(Goals_output + '\n')
        new_dataset.write(Actions_output + '\n\n')

