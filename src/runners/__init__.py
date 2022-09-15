REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .gym_runner import GymRunner
REGISTRY['gym'] = GymRunner

from .parallel_coop_runner import ParallelCoopRunner
REGISTRY['parallel_coop'] = ParallelCoopRunner
