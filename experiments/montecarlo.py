"""Finance: price an arithmetic-average Asian call by Monte Carlo (GBM), randomness generated in-graph."""

import numpy as np
import torch

from lab.harness import run_experiment

PATHS, STEPS = 1_000_000, 64
S0, K_STRIKE, R, SIGMA, T = 100.0, 100.0, 0.03, 0.25, 1.0
DT = T / STEPS
DRIFT = (R - 0.5 * SIGMA**2) * DT
VOL = SIGMA * np.sqrt(DT)


class AsianCall(torch.nn.Module):
    def forward(self, s0):
        z = torch.randn(PATHS, STEPS, device=s0.device)
        logp = torch.cumsum(DRIFT + VOL * z, dim=1)
        avg = (s0 * torch.exp(logp)).mean(dim=1)
        return (torch.clamp(avg - K_STRIKE, min=0.0).mean() * np.exp(-R * T)).reshape(1)


def numpy_price(s0):
    z = np.random.default_rng().standard_normal((PATHS, STEPS), dtype=np.float32)
    logp = np.cumsum(DRIFT + VOL * z, axis=1)
    avg = (s0 * np.exp(logp)).mean(axis=1)
    return np.array([np.maximum(avg - K_STRIKE, 0).mean() * np.exp(-R * T)])


def main():
    s0 = np.array([S0], np.float32)
    run_experiment(
        "montecarlo_asian",
        AsianCall(),
        (s0,),
        ["s0"],
        baselines={"numpy": numpy_price},
        check=lambda out, ref: (
            abs(float(out.ravel()[0]) - float(ref.ravel()[0])) / float(ref.ravel()[0])
            < 0.01
        ),
        size_note=f"{PATHS:,} paths x {STEPS} steps",
    )


if __name__ == "__main__":
    main()
