"""The ONNX CHIP-8 must end every frame in exactly the same state as the Python reference emulator."""

import numpy as np
import onnxruntime as ort
import pytest

from notml.chip8 import asm, machine, reference

CYCLES = 8


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(machine.build(CYCLES).SerializeToString(), providers=["CPUExecutionProvider"])


def run_both(session, roms, frames, keys_for_frame, seeds=None):
    state = machine.initial_state(roms, seeds)
    refs = [reference.Chip8(rom, seed=s) for rom, s in zip(roms, state["ctl"][:, machine.RNG], strict=True)]
    for f in range(frames):
        keys = keys_for_frame(f)
        outs = session.run(None, {**state, "keys": keys})
        state = dict(zip(machine.SHAPES, outs[:5], strict=True))
        for k, ref in enumerate(refs):
            ref.frame(keys[k], CYCLES)
    return state, refs


def assert_same(state, refs):
    for k, ref in enumerate(refs):
        ctl = state["ctl"][k]
        assert ctl.tolist() == [ref.pc, ref.i, ref.sp, ref.dt, ref.st, ref.rng], f"console {k}"
        np.testing.assert_array_equal(state["v"][k], ref.v, err_msg=f"console {k} registers")
        np.testing.assert_array_equal(state["stack"][k], ref.stack, err_msg=f"console {k} stack")
        np.testing.assert_array_equal(state["mem"][k], ref.mem, err_msg=f"console {k} memory")
        np.testing.assert_array_equal(state["screen"][k], ref.screen.ravel(), err_msg=f"console {k} screen")


def random_rom(rng, length=600):
    """Random instructions, biased towards valid ones, with jumps kept inside the program."""
    words = []
    for _ in range(length // 2):
        top = int(rng.integers(0, 16))
        word = top << 12 | int(rng.integers(0, 4096))
        if top in (1, 2, 0xB):
            word = top << 12 | (0x200 + 2 * int(rng.integers(0, length // 2)))
        if top == 0:
            word = int(rng.choice([0x00E0, 0x00EE, 0x0123]))
        if top == 0xE:
            word = 0xE000 | int(rng.integers(0, 16)) << 8 | int(rng.choice([0x9E, 0xA1, 0x55]))
        if top == 0xF:
            word = (
                0xF000
                | int(rng.integers(0, 16)) << 8
                | int(rng.choice([7, 0xA, 0x15, 0x18, 0x1E, 0x29, 0x33, 0x55, 0x65]))
            )
        words.append(word)
    return bytes(b for w in words for b in (w >> 8, w & 255))


def test_random_programs_match_reference(session):
    rng = np.random.default_rng(0)
    roms = [random_rom(rng) for _ in range(64)]
    key_rng = np.random.default_rng(1)
    keys = [(key_rng.random((64, 16)) < 0.1).astype(np.int32) for _ in range(40)]
    state, refs = run_both(session, roms, 40, lambda f: keys[f])
    assert_same(state, refs)


def test_assembled_program_draws_digits(session):
    rom = asm.assemble("""
        LD V0, 123
        LD I, 0x300
        LD B, V0          ; 1, 2, 3 at 0x300
        LD V2, [I]        ; V0..V2 = 1, 2, 3
        LD V5, 4
        LD F, V1
        DRW V5, V5, 5     ; draw "2"
    loop: JP loop
    """)
    state, refs = run_both(session, [rom], 2, lambda f: np.zeros((1, 16), np.int32))
    assert_same(state, refs)
    assert refs[0].v[:3].tolist() == [1, 2, 3]
    assert refs[0].screen.sum() > 0


def test_wait_for_key(session):
    rom = asm.assemble("LD V3, K\nloop: JP loop")
    no_keys = np.zeros((1, 16), np.int32)
    key_9 = no_keys.copy()
    key_9[0, 9] = 1
    state, refs = run_both(session, [rom], 3, lambda f: key_9 if f == 2 else no_keys)
    assert_same(state, refs)
    assert refs[0].v[3] == 9


def test_catch_game_plays_the_same(session):
    from notml.chip8 import program

    rom = program("catch")
    rng = np.random.default_rng(3)
    keys = []
    for _ in range(300):  # each console holds left, right or nothing for a while
        k = np.zeros((16, 16), np.int32)
        choice = rng.integers(0, 3, 16)
        k[choice == 1, 4] = 1
        k[choice == 2, 6] = 1
        keys.append(k)
    state, refs = run_both(session, [rom] * 16, 300, lambda f: keys[f // 10])
    assert_same(state, refs)
    assert len({tuple(r.screen.ravel()) for r in refs}) > 1  # different keys, different games
