
env_map = {}

from btgym.envs.virtualhome.watch_tv_env import WatchTVEnv

vh_env_map = {
    "VH-WatchTV": WatchTVEnv
}

env_map.update(vh_env_map)