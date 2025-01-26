from pathlib import Path
from setuptools import setup

# This is where you add any fancy path resolution to the local lib:
project_root: Path = Path(__file__).parent

setup(
    install_requires=[
        "jax", 
        "diffrax>=0.6.2", 
        "networkx>=3.3",
        "equinox>=0.11.11",
        "frozendict",
        "tqdm",
        f"sigkerax @ {(project_root / 'sigkerax').as_uri()}",
    ]
)