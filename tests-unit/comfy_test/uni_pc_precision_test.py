"""UniPC corrector weights on a real shift-8 schedule must match a float64 run on every device.

High-shift flow schedules (Wan 2.2 uses shift 8) take tiny log-SNR steps early on, where the
B(h) recurrence cancels catastrophically in float32: the weights drift on CPU/CUDA and blow up
on MPS, whose expm1 is less precise.
"""
import pytest
import torch

import comfy.model_sampling
import comfy.samplers
from comfy.extra_samplers import uni_pc


class _FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


def _shift8_sigmas(steps=20):
    ms = _FlowSampling(None)
    ms.set_parameters(shift=8.0, multiplier=1000)
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", steps + 1)
    return torch.cat([sigmas[:-2], sigmas[-1:]])  # KSampler drops the penultimate sigma for uni_pc


def _solve_calls(monkeypatch, device, dtype, variant):
    calls = []
    real_solve = torch.linalg.solve

    def spy(R, b):
        rhos = real_solve(R, b)
        calls.append((b.cpu().to(torch.float64), rhos.cpu().to(torch.float64)))
        return rhos

    monkeypatch.setattr(torch.linalg, "solve", spy)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(1, 4, 4, 4, generator=g, dtype=torch.float64).to(device, dtype)
    noise = torch.randn(1, 4, 4, 4, generator=g, dtype=torch.float64).to(device, dtype)
    sigmas = _shift8_sigmas().to(device, dtype)

    def model(x, sigma, **kwargs):
        return x0 * torch.cos(20 * sigma.view(-1, 1, 1, 1)) + 0.3 * torch.sin(x)

    uni_pc.sample_unipc(model, noise * sigmas[0], sigmas.clone(), extra_args={}, disable=True, variant=variant)
    monkeypatch.undo()
    return calls


needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


@pytest.mark.parametrize("variant", ["bh1", "bh2"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=needs_mps)])
def test_corrector_weights_match_float64(monkeypatch, device, variant):
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # uni_pc builds rks with torch.tensor(); float64 reference needs float64 defaults
    try:
        reference = _solve_calls(monkeypatch, "cpu", torch.float64, variant)
    finally:
        torch.set_default_dtype(default)

    actual = _solve_calls(monkeypatch, device, torch.float32, variant)

    assert len(actual) == len(reference) > 0
    for (b, rhos), (b_ref, rhos_ref) in zip(actual, reference):
        torch.testing.assert_close(b, b_ref, rtol=1e-5, atol=0)
        torch.testing.assert_close(rhos, rhos_ref, rtol=0, atol=1e-3)
