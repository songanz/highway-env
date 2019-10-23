from stable_baselines.surprise_off_po.dqn import DQN
from stable_baselines.surprise_off_po.sac import SAC
from stable_baselines.surprise_off_po.ddpg import DDPG

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines.surprise_off_po.trpo_mpi import TRPO_MPI
del mpi4py

__version__ = "2.7.0"
