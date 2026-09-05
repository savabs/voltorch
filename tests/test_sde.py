"""Tests for agent.quant.sde — Differentiable SDE Module (Math Stack M1)."""

import pytest
import torch

# voltorch.sde is behind the optional [sde] extra: it needs torchsde, which
# pulls a solver stack that users of the pricers never touch. Skipping here
# rather than failing collection is what makes `pip install voltorch && pytest`
# a green run for someone who only wanted the pricers.
pytest.importorskip("torchsde", reason="install voltorch[sde] to run these")

from voltorch.sde import GBM, HestonSDE, SDEConfig, make_time_grid


class TestGBM:
    def test_smoke_simulate(self):
        gbm = GBM()
        y0 = torch.full((3, 1), 100.0)
        ts = make_time_grid(T=1.0, n_steps=63)
        ys = gbm.simulate(y0, ts)
        assert ys.shape == (64, 3, 1)
        assert not torch.isnan(ys).any()
        assert (ys > 0).all()

    def test_gradient_flows_to_mu(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        y0 = torch.full((4, 1), 100.0)
        ts = make_time_grid(T=0.5, n_steps=126)
        ys = gbm.simulate(y0, ts)
        loss = ys[-1].mean()
        loss.backward()
        assert gbm.mu.grad is not None
        assert gbm.sigma.grad is not None
        assert gbm.mu.grad.item() != 0.0
        assert gbm.sigma.grad.item() != 0.0

    def test_gradient_flows_to_sigma(self):
        gbm = GBM()
        y0 = torch.full((2, 1), 100.0)
        ts = make_time_grid(T=1.0, n_steps=252)
        ys = gbm.simulate(y0, ts)
        loss = ys[-1].std()
        loss.backward()
        assert gbm.sigma.grad is not None
        assert gbm.sigma.grad.item() != 0.0

    def test_frozen_parameters_no_grad(self):
        gbm = GBM(mu=0.05, sigma=0.20, learnable=False)
        y0 = torch.full((2, 1), 100.0)
        ts = make_time_grid(T=1.0, n_steps=63)
        ys = gbm.simulate(y0, ts)
        assert not ys.requires_grad
        assert gbm.mu.grad is None
        assert gbm.sigma.grad is None

    def test_n_samples(self):
        gbm = GBM(learnable=False)
        y0 = torch.full((2, 1), 100.0)
        ts = make_time_grid(T=1.0, n_steps=21)
        ys = gbm.simulate(y0, ts, n_samples=5)
        assert ys.shape == (22, 10, 1)

    def test_adjoint_mode(self):
        gbm = GBM()
        cfg = SDEConfig(adjoint=True)
        y0 = torch.full((2, 1), 100.0)
        ts = make_time_grid(T=0.25, n_steps=63)
        ys = gbm.simulate(y0, ts, config=cfg)
        loss = ys[-1].mean()
        loss.backward()
        assert gbm.mu.grad is not None
        assert gbm.mu.grad.item() != 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu(self):
        gbm = GBM().cuda()
        y0 = torch.full((4, 1), 100.0).cuda()
        ts = make_time_grid(T=1.0, n_steps=63, device=torch.device("cuda"))
        ys = gbm.simulate(y0, ts)
        loss = ys[-1].mean()
        loss.backward()
        assert gbm.mu.grad is not None
        assert gbm.mu.grad.device.type == "cuda"


class TestHestonSDE:
    def test_smoke_simulate(self):
        model = HestonSDE()
        y0 = torch.tensor([[100.0, 0.04], [95.0, 0.06]])
        ts = make_time_grid(T=1.0, n_steps=63)
        ys = model.simulate(y0, ts)
        assert ys.shape == (64, 2, 2)
        assert not torch.isnan(ys).any()
        assert (ys[:, :, 0] > 0).all()

    def test_variance_nonnegative(self):
        model = HestonSDE(kappa=5.0, theta=0.04, xi=0.10)
        y0 = torch.tensor([[100.0, 0.04]])
        ts = make_time_grid(T=1.0, n_steps=252)
        ys = model.simulate(y0, ts)
        V = ys[:, :, 1]
        assert (V >= -1e-6).all()

    def test_gradient_flows_all_params(self):
        model = HestonSDE()
        y0 = torch.tensor([[100.0, 0.04]])
        ts = make_time_grid(T=0.5, n_steps=126)
        ys = model.simulate(y0, ts)
        loss = ys[-1, :, 0].mean()
        loss.backward()
        for p in ["mu", "kappa", "theta", "xi", "rho"]:
            g = getattr(model, p).grad
            assert g is not None, f"{p} grad is None"
            assert g.item() != 0.0, f"{p} grad is zero"

    def test_frozen_parameters_no_grad(self):
        model = HestonSDE(learnable=False)
        y0 = torch.tensor([[100.0, 0.04]])
        ts = make_time_grid(T=0.25, n_steps=63)
        ys = model.simulate(y0, ts)
        assert not ys.requires_grad
        for p in ["mu", "kappa", "theta", "xi", "rho"]:
            assert getattr(model, p).grad is None, f"{p} should have no grad"

    def test_rho_clamped(self):
        model = HestonSDE(rho=2.0)
        y0 = torch.tensor([[100.0, 0.04]])
        ts = make_time_grid(T=0.25, n_steps=21)
        ys = model.simulate(y0, ts)
        assert not torch.isnan(ys).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu(self):
        model = HestonSDE().cuda()
        y0 = torch.tensor([[100.0, 0.04]]).cuda()
        ts = make_time_grid(T=0.25, n_steps=63, device=torch.device("cuda"))
        ys = model.simulate(y0, ts)
        loss = ys[-1, :, 0].mean()
        loss.backward()
        assert model.mu.grad is not None


class TestMakeTimeGrid:
    def test_default(self):
        ts = make_time_grid()
        assert ts.shape == (253,)
        assert ts[0].item() == 0.0
        assert ts[-1].item() == 1.0

    def test_custom(self):
        ts = make_time_grid(T=2.0, n_steps=100)
        assert ts.shape == (101,)
        assert ts[-1].item() == 2.0
