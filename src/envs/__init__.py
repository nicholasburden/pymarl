from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

from .starcraft2 import StarCraft2Env
REGISTRY["sc2"] = partial(env_fn,
                          env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))



