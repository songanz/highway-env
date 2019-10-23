from stable_baselines.surprise.dqn import DQN
from stable_baselines.surprise.sac import SAC
from stable_baselines.surprise.ddpg import DDPG

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines.surprise.trpo_mpi import TRPO_MPI
del mpi4py

__version__ = "2.7.0"
