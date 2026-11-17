"""Every case, at a small size, must give the same answer through onnxruntime as its reference baseline."""

import numpy as np
import pytest

import experiments
from experiments import montecarlo, vm
from lab.harness import onnx_runner, save_onnx

CASES = experiments.all_cases(small=True)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_onnx_matches_reference(case, tmp_path):
    if case.known_bad:
        pytest.skip(case.known_bad)
    path = tmp_path / f"{case.name}.onnx"
    save_onnx(case, path)
    out = onnx_runner(case, path, "CPUExecutionProvider")()
    ref = case.reference(*case.inputs.values())
    assert case.tolerance.check(out, ref)


def test_vm_matches_python_interpreter(tmp_path):
    (case,) = vm.cases(small=True)
    path = tmp_path / "vm.onnx"
    save_onnx(case, path)
    regs = onnx_runner(case, path, "CPUExecutionProvider")()
    rng = np.random.default_rng(1)
    for lane in rng.choice(len(regs), 200, replace=False):
        program = vm.GCD if lane % 2 == 0 else vm.COLLATZ
        expected, _ = vm.python_vm(program, case.inputs["regs"][lane].tolist(), vm.CALLS * vm.CYCLES_PER_CALL)
        assert regs[lane].tolist() == expected


def test_vm_computes_gcd():
    (case,) = vm.cases(small=True)
    regs, _ = vm.numpy_vm(**case.inputs, cycles=vm.CALLS * vm.CYCLES_PER_CALL)
    start = case.inputs["regs"]
    assert np.array_equal(regs[0::2, 0], np.gcd(start[0::2, 0], start[0::2, 1]))


def test_pcg_normals_look_normal():
    q = montecarlo.normal_quality("pcg")
    assert abs(q["mean"]) < 0.01
    assert abs(q["std"] - 1) < 0.01
    assert abs(q["kurtosis"] - 3) < 0.05
    assert abs(q["lag1_autocorr"]) < 0.01
