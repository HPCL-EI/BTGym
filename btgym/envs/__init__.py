
env_map = {}

from btgym.envs.virtualhome.watch_tv_env import WatchTVEnv as VHWatchTVEnv
from btgym.envs.virtualhometext.watch_tv_env import WatchTVEnv as VHTWatchTVEnv

from btgym.envs.virtualhome.put_milk_in_fridge import MilkFridgeEnv as MilkFridgeEnv

vh_env_map = {
    "VH-WatchTV": VHWatchTVEnv,
    "VHT-WatchTV": VHTWatchTVEnv,
    "VH-PutMilkInFridge":MilkFridgeEnv
}

env_map.update(vh_env_map)