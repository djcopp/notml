"""Physics: one O(N^2) gravitational acceleration step for 4096 bodies."""

import numpy as np
import torch

from lab.harness import run_experiment

N, EPS = 4096, 1e-2


class Accel(torch.nn.Module):
    def forward(self, pos, mass):
        d = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 3]
        inv = torch.rsqrt((d * d).sum(-1) + EPS) ** 3  # [N, N]
        return (d * (inv * mass.unsqueeze(0)).unsqueeze(-1)).sum(1)


def numpy_accel(pos, mass):
    d = pos[None, :, :] - pos[:, None, :]
    inv = (np.sum(d * d, -1) + EPS) ** -1.5
    return np.einsum("ijk,ij->ik", d, inv * mass[None, :])


def main():
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((N, 3)).astype(np.float32)
    mass = rng.random(N).astype(np.float32)
    run_experiment(
        "nbody_step",
        Accel(),
        (pos, mass),
        ["pos", "mass"],
        baselines={"numpy": numpy_accel},
        check=lambda out, ref: (
            np.abs(out - ref).max() / np.abs(ref).max() < 2e-3
        ),  # rel. to max magnitude
        size_note=f"{N} bodies",
    )


if __name__ == "__main__":
    main()
