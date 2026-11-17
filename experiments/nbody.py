"""Physics: one O(N²) gravitational acceleration step."""

import numpy as np
import torch

from lab.case import Case, Tolerance

SOFTENING = 1e-2


class Acceleration(torch.nn.Module):
    def forward(self, pos, mass):
        d = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 3]
        inv_r3 = torch.rsqrt((d * d).sum(-1) + SOFTENING) ** 3  # [N, N]
        return (d * (inv_r3 * mass.unsqueeze(0)).unsqueeze(-1)).sum(1)


def numpy_acceleration(pos, mass):
    d = pos[None, :, :] - pos[:, None, :]
    inv_r3 = (np.sum(d * d, -1) + SOFTENING) ** -1.5
    return np.einsum("ijk,ij->ik", d, inv_r3 * mass[None, :])


def cases(small: bool = False) -> list[Case]:
    n = 256 if small else 4096
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((n, 3)).astype(np.float32)
    mass = rng.random(n).astype(np.float32)
    return [
        Case(
            "nbody",
            Acceleration(),
            {"pos": pos, "mass": mass},
            {"numpy": numpy_acceleration},
            # a few accelerations nearly cancel to zero, so compare against the largest magnitude
            Tolerance("max_rel", rtol=2e-3),
            f"{n:,} bodies",
        )
    ]
