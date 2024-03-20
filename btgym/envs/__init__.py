
env_map = {}

from btgym.envs.virtualhome.watch_tv_env import WatchTVEnv as VHWatchTVEnv
from btgym.envs.virtualhometext.watch_tv_env import WatchTVEnv as VHTWatchTVEnv

vh_env_map = {
    "VH-WatchTV": VHWatchTVEnv,
    "VHT-WatchTV": VHTWatchTVEnv
}

env_map.update(vh_env_map)