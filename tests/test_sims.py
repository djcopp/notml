"""Each simulation graph must match a straightforward numpy version of the same rules."""

import numpy as np
import onnxruntime as ort

from notml import sims


def run(model, **feeds):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def numpy_life(x, steps):
    for _ in range(steps):
        n = sum(np.roll(np.roll(x, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - x
        x = ((n == 3) | ((x == 1) & (n == 2))).astype(np.float32)
    return x


def test_life():
    x = (np.random.default_rng(0).random((48, 64)) < 0.3).astype(np.float32)
    cells, image = run(sims.life(48, 64, steps=8), cells=x[None, None])
    expected = numpy_life(x, 8)
    np.testing.assert_array_equal(cells[0, 0], expected)
    np.testing.assert_array_equal(image, (expected * 255).astype(np.uint8))


def test_glider_moves_across_the_wrap():
    x = np.zeros((8, 8), np.float32)
    x[[0, 1, 2, 2, 2], [1, 2, 0, 1, 2]] = 1  # glider heading down-right
    cells, _ = run(sims.life(8, 8, steps=32), cells=x[None, None])
    np.testing.assert_array_equal(cells[0, 0], x)  # 32 steps = 8 cells diagonally = back where it started


def numpy_gray_scott(uv, steps, feed, kill, du, dv):
    def lap(a):
        out = -a.copy()
        for (dy, dx), w in {
            (0, 1): 0.2,
            (0, -1): 0.2,
            (1, 0): 0.2,
            (-1, 0): 0.2,
            (1, 1): 0.05,
            (1, -1): 0.05,
            (-1, 1): 0.05,
            (-1, -1): 0.05,
        }.items():
            out += w * np.roll(np.roll(a, dy, 0), dx, 1)
        return out

    u, v = uv[0].astype(np.float64), uv[1].astype(np.float64)
    for _ in range(steps):
        uvv = u * v * v
        u, v = u + du * lap(u) - uvv + feed * (1 - u), v + dv * lap(v) + uvv - (feed + kill) * v
    return np.stack([u, v])


def test_gray_scott():
    rng = np.random.default_rng(1)
    uv = np.stack([np.ones((32, 40)), np.zeros((32, 40))]).astype(np.float32)
    uv[1, 10:20, 12:22] = 0.5 + 0.1 * rng.random((10, 10))
    out, image = run(sims.gray_scott(32, 40, steps=16), uv=uv[None])
    np.testing.assert_allclose(out[0], numpy_gray_scott(uv, 16, **sims.GRAY_SCOTT), atol=1e-4)
    assert image.dtype == np.uint8 and image.shape == (32, 40)


def numpy_particle_life(pos, vel, kind, rules, radius=0.08, beta=0.3, friction=0.707, dt=0.02, force=10.0):
    attraction = kind @ rules @ kind.T
    d = pos[None, :, :] - pos[:, None, :]
    d -= np.round(d)
    r = np.sqrt((d**2).sum(-1) + 1e-9)
    x = r / radius
    strength = np.where(
        x < beta, x / beta - 1, np.where(x < 1, attraction * (1 - np.abs(2 * x - 1 - beta) / (1 - beta)), 0)
    )
    acc = (d / r[..., None] * strength[..., None]).sum(1) * force * radius
    vel = vel * friction + acc * dt
    pos = pos + vel * dt
    return pos - np.floor(pos), vel


def test_particle_life():
    rng = np.random.default_rng(2)
    n, types = 300, 4
    pos = rng.random((n, 2)).astype(np.float32)
    vel = np.zeros((n, 2), np.float32)
    kind = np.eye(types, dtype=np.float32)[rng.integers(0, types, n)]
    rules = rng.uniform(-1, 1, (types, types)).astype(np.float32)
    new_pos, new_vel = run(sims.particle_life(n, types), pos=pos, vel=vel, kind=kind, rules=rules)
    ref_pos, ref_vel = numpy_particle_life(pos, vel, kind, rules)
    np.testing.assert_allclose(new_vel, ref_vel, atol=1e-4)
    np.testing.assert_allclose(new_pos, ref_pos, atol=1e-5)
