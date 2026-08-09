"""Reading the path files the OMPL demos write.

A single file holds several unrelated motions, and the seams between them are not
motions at all. Anything that replays or audits one of these files has to agree on
where those seams are, so the splitting lives here rather than in either script.
"""

from __future__ import annotations

import numpy as np


MOTION_MARKER = "# motion"


def load_runs(path: str) -> list[np.ndarray]:
    """Read a path file as the separate motions it holds.

    `demo_UR5MBMBenchmark` labels each motion with a `# motion <scene> <index>` line, and
    those markers are authoritative: its file holds one motion per benchmark problem, from
    unrelated start states, so nothing about the configurations themselves marks the seam.
    Files without markers (`demo_UR5PyBulletScene`) fall back to `split_runs`.
    """
    runs: list[list[list[float]]] = []
    marked = False
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(MOTION_MARKER):
                marked = True
                runs.append([])
            elif line and not line.startswith("#"):
                if not runs:
                    runs.append([])
                runs[-1].append([float(v) for v in line.split()])
    rows = [np.asarray(run, dtype=float) for run in runs if len(run) > 1]
    if not rows:
        raise ValueError(f"{path} holds no configurations")
    return rows if marked else split_runs(rows[0])


def split_runs(path: np.ndarray) -> list[np.ndarray]:
    """Split the file into the separate motions it actually holds.

    `demo_UR5PyBulletScene` writes one path per goal into a single file, each
    re-prefixed with the scene's start configuration. Between the end of one goal's
    motion and the start of the next there is a *jump*: the arm is back at home, having
    never travelled there. Treating that seam as an edge -- densifying across it, or
    animating along it -- invents states along a straight line from deep inside a shelf
    back to the home pose, which collide with everything and belong to no trajectory the
    planner produced. Splitting on repeats of the first configuration recovers the real
    motions.
    """
    start = path[0]
    breaks = [i for i in range(len(path)) if np.array_equal(path[i], start)]
    bounds = breaks + [len(path)]
    runs = [path[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1]
    return runs or [path]
