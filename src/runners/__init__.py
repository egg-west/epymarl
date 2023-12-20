REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .mtrl_episode_runner import MTRLRunner
REGISTRY["mtrl"] = MTRLRunner