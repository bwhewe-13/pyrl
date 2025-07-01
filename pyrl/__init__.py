from gymnasium.envs.registration import register

from . import envs, modeling

__all__ = ["modeling", "envs"]

register(id="envs/PrisonGuard01-v0", entry_point="pyrl.envs.prisonguard:PrisonGuard01")
