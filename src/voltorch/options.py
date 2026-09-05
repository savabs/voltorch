"""
TirraMind — Options Pricing & Greeks Module (Math Stack M2)

Provides vectorized, differentiable option pricing and sensitivity calculations (Greeks)
under the Black-Scholes-Merton and Barone-Adesi Whaley frameworks in PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Stable, differentiable cumulative standard normal distribution CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    """Stable, differentiable standard normal probability density function PDF."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


class BlackScholes(nn.Module):
    """Vectorized and fully differentiable Black-Scholes-Merton pricing engine."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: torch.Tensor,
        sigma: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        is_call: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        """Calculate European option prices under Black-Scholes-Merton.

        Args:
            S: Spot price tensor.
            K: Strike price tensor.
            T: Time to expiration tensor (in years).
            r: Continuous risk-free interest rate tensor.
            sigma: Continuous volatility tensor.
            q: Continuous dividend yield tensor (optional, defaults to zero).
            is_call: Boolean or boolean tensor indicating option type (True for Call, False for Put).

        Returns:
            Option price tensor of shape matching inputs.
        }
        """
        # Enforce stable boundaries to prevent division by zero or NaN gradients
        S_clamped = torch.clamp(S, min=1e-6)
        K_clamped = torch.clamp(K, min=1e-6)
        T_clamped = torch.clamp(T, min=1e-5)
        sigma_clamped = torch.clamp(sigma, min=1e-4)

        if q is None:
            q = torch.zeros_like(S_clamped)

        # Compute d1 and d2
        d1 = (
            torch.log(S_clamped / K_clamped)
            + (r - q + 0.5 * sigma_clamped**2) * T_clamped
        ) / (sigma_clamped * torch.sqrt(T_clamped))
        d2 = d1 - sigma_clamped * torch.sqrt(T_clamped)

        # Option type mask
        if isinstance(is_call, torch.Tensor):
            call_mask = is_call.float()
            put_mask = 1.0 - call_mask
        else:
            call_mask = 1.0 if is_call else 0.0
            put_mask = 0.0 if is_call else 1.0

        # Calculate call and put prices
        c_price = S_clamped * torch.exp(-q * T_clamped) * _norm_cdf(
            d1
        ) - K_clamped * torch.exp(-r * T_clamped) * _norm_cdf(d2)
        p_price = K_clamped * torch.exp(-r * T_clamped) * _norm_cdf(
            -d2
        ) - S_clamped * torch.exp(-q * T_clamped) * _norm_cdf(-d1)

        return call_mask * c_price + put_mask * p_price

    def analytical_greeks(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: torch.Tensor,
        sigma: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        is_call: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        """Compute analytical Greeks for European options.

        Returns a dictionary with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.
        """
        S_clamped = torch.clamp(S, min=1e-6)
        K_clamped = torch.clamp(K, min=1e-6)
        T_clamped = torch.clamp(T, min=1e-5)
        sigma_clamped = torch.clamp(sigma, min=1e-4)

        if q is None:
            q = torch.zeros_like(S_clamped)

        sqrt_T = torch.sqrt(T_clamped)
        d1 = (
            torch.log(S_clamped / K_clamped)
            + (r - q + 0.5 * sigma_clamped**2) * T_clamped
        ) / (sigma_clamped * sqrt_T)
        d2 = d1 - sigma_clamped * sqrt_T

        nd1 = _norm_pdf(d1)
        Nd1 = _norm_cdf(d1)
        Nd2 = _norm_cdf(d2)
        Nmd1 = _norm_cdf(-d1)
        Nmd2 = _norm_cdf(-d2)

        if isinstance(is_call, torch.Tensor):
            call_mask = is_call.float()
            put_mask = 1.0 - call_mask
        else:
            call_mask = 1.0 if is_call else 0.0
            put_mask = 0.0 if is_call else 1.0

        # Delta
        delta = (
            call_mask * torch.exp(-q * T_clamped) * Nd1
            - put_mask * torch.exp(-q * T_clamped) * Nmd1
        )

        # Gamma (same for call and put)
        gamma = torch.exp(-q * T_clamped) * nd1 / (S_clamped * sigma_clamped * sqrt_T)

        # Vega (same for call and put)
        vega = S_clamped * torch.exp(-q * T_clamped) * nd1 * sqrt_T

        # Theta (time decay, negative for long options)
        theta_call = (
            -S_clamped
            * torch.exp(-q * T_clamped)
            * nd1
            * sigma_clamped
            / (2.0 * sqrt_T)
            - q * S_clamped * torch.exp(-q * T_clamped) * Nd1
            - r * K_clamped * torch.exp(-r * T_clamped) * Nd2
        )
        theta_put = (
            -S_clamped
            * torch.exp(-q * T_clamped)
            * nd1
            * sigma_clamped
            / (2.0 * sqrt_T)
            + q * S_clamped * torch.exp(-q * T_clamped) * Nmd1
            - r * K_clamped * torch.exp(-r * T_clamped) * Nmd2
        )
        theta = call_mask * theta_call + put_mask * theta_put

        # Rho
        rho_call = K_clamped * T_clamped * torch.exp(-r * T_clamped) * Nd2
        rho_put = -K_clamped * T_clamped * torch.exp(-r * T_clamped) * Nmd2
        rho = call_mask * rho_call + put_mask * rho_put

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    def autograd_greeks(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: torch.Tensor,
        sigma: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        is_call: bool | torch.Tensor = True,
    ) -> dict[str, torch.Tensor]:
        """Compute Greeks via PyTorch autograd for full differentiability.

        Returns a dictionary with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.
        """
        S_in = S.clone().requires_grad_(True)
        K_in = K.clone().requires_grad_(True)
        T_in = T.clone().requires_grad_(True)
        r_in = r.clone().requires_grad_(True)
        sigma_in = sigma.clone().requires_grad_(True)

        price = self.forward(S_in, K_in, T_in, r_in, sigma_in, q, is_call)
        price_sum = price.sum()

        # Delta
        delta = torch.autograd.grad(price_sum, S_in, create_graph=True)[0]

        # Gamma (second derivative w.r.t S)
        grad_delta = torch.autograd.grad(delta.sum(), S_in, create_graph=True)[0]
        gamma = grad_delta

        # Vega
        vega = torch.autograd.grad(price_sum, sigma_in, create_graph=True)[0]

        # Theta (negative of derivative w.r.t T)
        theta = -torch.autograd.grad(price_sum, T_in, create_graph=True)[0]

        # Rho
        rho = torch.autograd.grad(price_sum, r_in, create_graph=True)[0]

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }


def implied_volatility(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    market_price: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    is_call: bool = True,
    max_iters: int = 20,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Robust, vectorized, differentiable Newton-Raphson implied volatility solver.

    Uses a fixed number of iterations with clamping to maintain continuous gradients
    suitable for GPU execution and backpropagation.
    """
    bs = BlackScholes()

    # Initial guess: Brenner-Subrahmanyam approximation
    S_clamped = torch.clamp(S, min=1e-6)
    K_clamped = torch.clamp(K, min=1e-6)
    T_clamped = torch.clamp(T, min=1e-5)

    # For ATM options, initial guess is sqrt(2*pi/T) * (price / S)
    # For others, use the log-moneyness heuristic
    moneyness = torch.log(S_clamped / K_clamped)
    atm_approx = torch.abs(market_price / S_clamped) * torch.sqrt(
        2.0 * math.pi / T_clamped
    )
    sigma = torch.where(
        torch.abs(moneyness) < 0.1,
        atm_approx,
        torch.sqrt(torch.abs(moneyness) / T_clamped),
    )
    sigma = torch.clamp(sigma, min=1e-4, max=5.0)

    for _ in range(max_iters):
        price = bs(S, K, T, r, sigma, q, is_call)
        vega = bs.analytical_greeks(S, K, T, r, sigma, q, is_call)["vega"]

        # Avoid division by zero when vega is tiny
        vega_safe = torch.where(
            torch.abs(vega) < 1e-10, torch.ones_like(vega) * 1e-10, vega
        )

        diff = price - market_price
        update = diff / vega_safe
        sigma = sigma - update

        # Clamp to prevent negative or exploding volatility
        sigma = torch.clamp(sigma, min=1e-4, max=5.0)

    return sigma


class BaroneAdesiWhaley(nn.Module):
    """Vectorized, differentiable American option pricing via Barone-Adesi-Whaley (1987) approximation.

    This model prices American calls and puts by decomposing the price into the
    European Black-Scholes price plus an early exercise premium.
    """

    def __init__(self):
        super().__init__()
        self.bs = BlackScholes()

    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: torch.Tensor,
        sigma: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        is_call: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        """Price American options using the Barone-Adesi-Whaley quadratic approximation.

        Args:
            S: Spot price tensor.
            K: Strike price tensor.
            T: Time to expiration tensor (in years).
            r: Continuous risk-free interest rate tensor.
            sigma: Continuous volatility tensor.
            q: Continuous dividend yield tensor (optional, defaults to zero).
            is_call: Boolean or boolean tensor indicating option type.

        Returns:
            American option price tensor.
        """
        S_clamped = torch.clamp(S, min=1e-6)
        K_clamped = torch.clamp(K, min=1e-6)
        T_clamped = torch.clamp(T, min=1e-5)
        sigma_clamped = torch.clamp(sigma, min=1e-4)

        if q is None:
            q = torch.zeros_like(S_clamped)

        b = r - q  # Cost of carry

        # Compute European price
        eur_price = self.bs(
            S_clamped, K_clamped, T_clamped, r, sigma_clamped, q, is_call
        )

        if isinstance(is_call, torch.Tensor):
            call_mask = is_call.float()
            put_mask = 1.0 - call_mask
        else:
            call_mask = 1.0 if is_call else 0.0
            put_mask = 0.0 if is_call else 1.0

        # --- Put early exercise premium (always computed) ---
        sigma2 = sigma_clamped**2
        b_safe = torch.where(torch.abs(b) < 1e-10, torch.ones_like(b) * 1e-10, b)
        exp_bT = torch.exp(-b_safe * T_clamped)
        denominator = 1.0 - exp_bT
        denominator_safe = torch.where(
            torch.abs(denominator) < 1e-10,
            torch.ones_like(denominator) * 1e-10,
            denominator,
        )

        c = -b_safe / denominator_safe
        a_coeff = 0.5 * sigma2
        b_coeff = b_safe - 0.5 * sigma2

        discriminant = b_coeff**2 - 4.0 * a_coeff * c
        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)

        # q1 is the negative root (used for puts)
        q1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)

        # Critical spot price S* for puts
        r_b_diff = r - b_safe
        r_b_diff_safe = torch.where(
            torch.abs(r_b_diff) < 1e-10,
            torch.ones_like(r_b_diff) * 1e-10,
            r_b_diff,
        )
        # Find critical spot price S* for puts via Newton-Raphson
        # Target: f(S*) = P_Eur(S*) + A1*(S*/S*)^q1 - (K - S*) = 0
        # Start with a better initial guess
        S_star_put = K_clamped * (1.0 - (r / r_b_diff_safe) * sigma2 * T_clamped / 2.0)
        S_star_put = torch.clamp(S_star_put, min=K_clamped * 0.1)

        for _ in range(5):
            d1_Sstar_put = (
                torch.log(S_star_put / K_clamped) + (b_safe + 0.5 * sigma2) * T_clamped
            ) / (sigma_clamped * torch.sqrt(T_clamped))
            A1_iter = -(S_star_put / q1) * (
                1.0 - torch.exp((b_safe - r) * T_clamped) * _norm_cdf(-d1_Sstar_put)
            )
            put_premium_iter = A1_iter * ((S_clamped / S_star_put) ** q1)
            # f(S*) = P_Eur(S*) + put_premium - (K - S*)
            f = (
                self.bs(
                    S_star_put, K_clamped, T_clamped, r, sigma_clamped, q, is_call=False
                )
                + put_premium_iter
                - (K_clamped - S_star_put)
            )
            # f'(S*) = delta_put(S*) + d(put_premium)/dS + 1
            # d(put_premium)/dS = A1 * q1 * (S/S*)^(q1-1) * (1/S*)
            # At S=S*: d(put_premium)/dS = A1 * q1 / S*
            delta_put = self.bs.analytical_greeks(
                S_star_put, K_clamped, T_clamped, r, sigma_clamped, q, is_call=False
            )["delta"]
            d_premium_dS = A1_iter * q1 / S_star_put
            fp = delta_put + d_premium_dS + 1.0
            fp_safe = torch.where(
                torch.abs(fp) < 1e-10, torch.ones_like(fp) * 1e-10, fp
            )
            S_star_put = S_star_put - f / fp_safe
            S_star_put = torch.clamp(S_star_put, min=K_clamped * 0.1)

        d1_Sstar_put = (
            torch.log(S_star_put / K_clamped) + (b_safe + 0.5 * sigma2) * T_clamped
        ) / (sigma_clamped * torch.sqrt(T_clamped))
        A1 = -(S_star_put / q1) * (
            1.0 - torch.exp((b_safe - r) * T_clamped) * _norm_cdf(-d1_Sstar_put)
        )

        put_premium = A1 * ((S_clamped / S_star_put) ** q1)
        put_exercise = K_clamped - S_clamped
        put_am = eur_price + put_premium
        put_am = torch.where(S_clamped <= S_star_put, put_exercise, put_am)

        # --- Call early exercise premium (only when b < r, i.e., q > 0) ---
        # When b >= r, American call = European call
        b_lt_r = (b < r).float()

        # Compute q2 only where needed to avoid division by zero in S_star_call
        # For safety, compute q2 everywhere but mask it out when b >= r
        q2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)
        S_star_call_raw = K_clamped * (
            1.0 + (r / r_b_diff_safe) * sigma2 * T_clamped / 2.0
        )
        # Only compute early exercise premium when b < r; otherwise set S* = K to avoid NaN
        S_star_call = torch.where(b_lt_r > 0.5, S_star_call_raw, K_clamped)

        d1_Sstar_call = (
            torch.log(S_star_call / K_clamped) + (b_safe + 0.5 * sigma2) * T_clamped
        ) / (sigma_clamped * torch.sqrt(T_clamped))
        A2_raw = (S_star_call / q2) * (
            1.0 - torch.exp((b_safe - r) * T_clamped) * _norm_cdf(d1_Sstar_call)
        )
        # Mask A2 to avoid inf*0 = NaN when b >= r
        A2 = torch.where(b_lt_r > 0.5, A2_raw, torch.zeros_like(eur_price))

        call_premium = A2 * ((S_clamped / S_star_call) ** q2)
        # Avoid NaN propagation when b >= r by using torch.where
        call_premium_safe = torch.where(
            b_lt_r > 0.5, call_premium, torch.zeros_like(eur_price)
        )
        call_exercise = S_clamped - K_clamped
        call_am = eur_price + call_premium_safe
        call_am = torch.where(
            (S_clamped >= S_star_call) & (b_lt_r > 0.5), call_exercise, call_am
        )

        return call_mask * call_am + put_mask * put_am


# ═══════════════════════════════════════════════════════════════
# Fourier-Cosine (COS) Expansion Pricing Engine (Math Stack M3)
# ═══════════════════════════════════════════════════════════════


def _chi_k(
    k: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """Analytical integral of exp(y) * cos(k * pi * (y - a) / (b - a)) over [c, d]."""
    k_pi = k * math.pi
    b_a = b - a
    term = k_pi / b_a

    cos_d = torch.cos(term * (d - a))
    cos_c = torch.cos(term * (c - a))
    sin_d = torch.sin(term * (d - a))
    sin_c = torch.sin(term * (c - a))

    exp_d = torch.exp(d)
    exp_c = torch.exp(c)

    val = (1.0 / (1.0 + term**2)) * (
        exp_d * cos_d - exp_c * cos_c + term * exp_d * sin_d - term * exp_c * sin_c
    )
    val_k0 = exp_d - exp_c
    return torch.where(k == 0, val_k0, val)


def _psi_k(
    k: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """Analytical integral of cos(k * pi * (y - a) / (b - a)) over [c, d]."""
    k_pi = k * math.pi
    b_a = b - a
    term = k_pi / b_a
    term_safe = torch.where(k == 0, torch.ones_like(term), term)

    sin_d = torch.sin(term_safe * (d - a))
    sin_c = torch.sin(term_safe * (c - a))

    val = (1.0 / term_safe) * (sin_d - sin_c)
    val_k0 = d - c
    return torch.where(k == 0, val_k0, val)


class FourierCOS(nn.Module):
    """Base class for European option pricing via Fourier-Cosine (COS) expansion.

    Subclasses must implement:
        - characteristic_function(u, T, r, q) -> complex tensor
        - cumulants(T, r, q) -> (c1, c2) tensors
    """

    def __init__(self, n_cos: int = 128, L: float = 10.0):
        super().__init__()
        self.n_cos = n_cos
        self.L = L

    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        is_call: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        """Price European options under the subclass specific risk-neutral model."""
        S_clamped = torch.clamp(S, min=1e-6)
        K_clamped = torch.clamp(K, min=1e-6)
        T_clamped = torch.clamp(T, min=1e-5)

        if q is None:
            q = torch.zeros_like(S_clamped)

        # Truncation interval [a, b] from cumulants
        c1, c2 = self.cumulants(T_clamped, r, q)
        b_minus_a = 2.0 * self.L * torch.sqrt(torch.abs(c2))
        a = c1 - self.L * torch.sqrt(torch.abs(c2))
        b = c1 + self.L * torch.sqrt(torch.abs(c2))

        # Shape formatting to column vectors for broadcasting
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        b_minus_a = b_minus_a.unsqueeze(-1)
        T_c = T_clamped.unsqueeze(-1)
        r_c = r.unsqueeze(-1)
        q_c = q.unsqueeze(-1)

        # Log moneyness x = log(S / K)
        x = torch.log(S_clamped / K_clamped).unsqueeze(-1)

        # k term tensor (1, N)
        device = S_clamped.device
        k = torch.arange(self.n_cos, dtype=torch.float32, device=device).unsqueeze(0)

        # Frequencies u_k = k * pi / (b - a)
        u = k * math.pi / b_minus_a

        # Model-specific characteristic function evaluation
        phi = self.characteristic_function(u, T_c, r_c, q_c)

        if isinstance(is_call, torch.Tensor):
            call_mask = is_call.float().unsqueeze(-1)
            put_mask = 1.0 - call_mask
        else:
            call_mask = 1.0 if is_call else 0.0
            put_mask = 0.0 if is_call else 1.0

        # Payoff coefficients H_k
        zero = torch.zeros_like(a)
        H_k_call = (2.0 / b_minus_a) * (
            _chi_k(k, a, b, zero, b) - _psi_k(k, a, b, zero, b)
        )
        H_k_put = (2.0 / b_minus_a) * (
            -_chi_k(k, a, b, a, zero) + _psi_k(k, a, b, a, zero)
        )
        H_k = call_mask * H_k_call + put_mask * H_k_put

        # Re{ phi(u) * exp(i * u * (x - a)) } expansion sum
        u_x_a = u * (x - a)
        cos_term = torch.cos(u_x_a)
        sin_term = torch.sin(u_x_a)

        re_sum = phi.real * cos_term - phi.imag * sin_term

        # First term weight scaling (k=0 gets 0.5)
        scale_mask = torch.ones_like(k)
        scale_mask[0, 0] = 0.5

        terms = re_sum * H_k * scale_mask
        option_val = K_clamped * torch.exp(-r * T_clamped) * torch.sum(terms, dim=-1)

        return torch.clamp(option_val, min=0.0)

    def characteristic_function(
        self, u: torch.Tensor, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def cumulants(
        self, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class HestonCOS(FourierCOS):
    """European option pricing under the Heston stochastic volatility model using COS."""

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.30,
        rho: float = -0.70,
        v0: float = 0.04,
        n_cos: int = 128,
        L: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__(n_cos=n_cos, L=L)
        for name, val in [
            ("kappa", kappa),
            ("theta", theta),
            ("xi", xi),
            ("rho", rho),
            ("v0", v0),
        ]:
            t = torch.tensor(val, dtype=torch.float32)
            if learnable:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def characteristic_function(
        self, u: torch.Tensor, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        u_c = torch.complex(u, torch.zeros_like(u))

        kappa = torch.clamp(self.kappa, min=1e-3)
        theta = torch.clamp(self.theta, min=1e-4)
        xi = torch.clamp(self.xi, min=1e-4)
        rho = torch.clamp(self.rho, min=-0.999, max=0.999)
        v0 = torch.clamp(self.v0, min=1e-4)

        d = torch.sqrt((kappa - 1j * rho * xi * u_c) ** 2 + xi**2 * (u_c**2 + 1j * u_c))
        g = (kappa - 1j * rho * xi * u_c - d) / (kappa - 1j * rho * xi * u_c + d)

        # Little trap branch-cut stable formulation (Albrecher et al., 2007)
        exp_dT = torch.exp(-d * T)
        exp1 = (
            (v0 / xi**2)
            * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
            * (kappa - 1j * rho * xi * u_c - d)
        )

        log_term = torch.log((1.0 - g * exp_dT) / (1.0 - g))
        exp2 = (kappa * theta / xi**2) * (
            T * (kappa - 1j * rho * xi * u_c - d) - 2.0 * log_term
        )

        # Complete characteristic function including the risk-neutral drift
        phi_drift = torch.exp(1j * u_c * (r - q) * T)

        return phi_drift * torch.exp(exp1 + exp2)

    def cumulants(
        self, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kappa = torch.clamp(self.kappa, min=1e-3)
        theta = torch.clamp(self.theta, min=1e-4)
        xi = torch.clamp(self.xi, min=1e-4)
        v0 = torch.clamp(self.v0, min=1e-4)

        c1 = (
            (r - q) * T
            + ((1.0 - torch.exp(-kappa * T)) / (2.0 * kappa)) * (theta - v0)
            - 0.5 * theta * T
        )
        c2 = (
            theta * T
            + (v0 / kappa) * (1.0 - torch.exp(-kappa * T))
            + (xi**2 * theta * T) / (4.0 * kappa**2)
        )
        return c1, c2


class BatesCOS(FourierCOS):
    """European option pricing under the Bates model (Heston + log-normal jumps)."""

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.30,
        rho: float = -0.70,
        v0: float = 0.04,
        lambda_j: float = 0.10,
        mu_j: float = -0.05,
        sigma_j: float = 0.15,
        n_cos: int = 128,
        L: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__(n_cos=n_cos, L=L)
        self.heston = HestonCOS(kappa, theta, xi, rho, v0, n_cos, L, learnable)
        for name, val in [
            ("lambda_j", lambda_j),
            ("mu_j", mu_j),
            ("sigma_j", sigma_j),
        ]:
            t = torch.tensor(val, dtype=torch.float32)
            if learnable:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def characteristic_function(
        self, u: torch.Tensor, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        phi_heston = self.heston.characteristic_function(u, T, r, q)

        u_c = torch.complex(u, torch.zeros_like(u))
        lambda_j = torch.clamp(self.lambda_j, min=0.0)
        sigma_j = torch.clamp(self.sigma_j, min=1e-4)

        # Martingale correction: k_j = E[exp(J)] - 1
        k_j = torch.exp(self.mu_j + 0.5 * sigma_j**2) - 1.0

        # Merton log-normal jump characteristic component
        jump_exponent = (
            lambda_j
            * T
            * (
                torch.exp(1j * u_c * self.mu_j - 0.5 * u_c**2 * sigma_j**2)
                - 1.0
                - 1j * u_c * k_j
            )
        )
        phi_jump = torch.exp(jump_exponent)

        return phi_heston * phi_jump

    def cumulants(
        self, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c1_h, c2_h = self.heston.cumulants(T, r, q)
        lambda_j = torch.clamp(self.lambda_j, min=0.0)
        sigma_j = torch.clamp(self.sigma_j, min=1e-4)

        c1 = c1_h + lambda_j * T * self.mu_j
        c2 = c2_h + lambda_j * T * (self.mu_j**2 + sigma_j**2)
        return c1, c2


class MertonCOS(FourierCOS):
    """European option pricing under the Merton Jump-Diffusion model using COS."""

    def __init__(
        self,
        sigma: float = 0.20,
        lambda_j: float = 0.10,
        mu_j: float = -0.05,
        sigma_j: float = 0.15,
        n_cos: int = 128,
        L: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__(n_cos=n_cos, L=L)
        for name, val in [
            ("sigma", sigma),
            ("lambda_j", lambda_j),
            ("mu_j", mu_j),
            ("sigma_j", sigma_j),
        ]:
            t = torch.tensor(val, dtype=torch.float32)
            if learnable:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def characteristic_function(
        self, u: torch.Tensor, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        u_c = torch.complex(u, torch.zeros_like(u))
        sigma = torch.clamp(self.sigma, min=1e-4)
        lambda_j = torch.clamp(self.lambda_j, min=0.0)
        sigma_j = torch.clamp(self.sigma_j, min=1e-4)

        k_j = torch.exp(self.mu_j + 0.5 * sigma_j**2) - 1.0

        phi_drift = torch.exp(1j * u_c * (r - q - 0.5 * sigma**2) * T)
        phi_diffusion = torch.exp(-0.5 * u_c**2 * sigma**2 * T)

        jump_exponent = (
            lambda_j
            * T
            * (
                torch.exp(1j * u_c * self.mu_j - 0.5 * u_c**2 * sigma_j**2)
                - 1.0
                - 1j * u_c * k_j
            )
        )
        phi_jump = torch.exp(jump_exponent)

        return phi_drift * phi_diffusion * phi_jump

    def cumulants(
        self, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.clamp(self.sigma, min=1e-4)
        lambda_j = torch.clamp(self.lambda_j, min=0.0)
        sigma_j = torch.clamp(self.sigma_j, min=1e-4)

        c1 = (r - q) * T - 0.5 * sigma**2 * T + lambda_j * T * self.mu_j
        c2 = sigma**2 * T + lambda_j * T * (self.mu_j**2 + sigma_j**2)
        return c1, c2


class VarianceGammaCOS(FourierCOS):
    """European option pricing under the Variance Gamma (VG) model using COS."""

    def __init__(
        self,
        sigma: float = 0.20,
        nu: float = 0.20,
        theta: float = -0.10,
        n_cos: int = 128,
        L: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__(n_cos=n_cos, L=L)
        for name, val in [
            ("sigma", sigma),
            ("nu", nu),
            ("theta", theta),
        ]:
            t = torch.tensor(val, dtype=torch.float32)
            if learnable:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def characteristic_function(
        self, u: torch.Tensor, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        u_c = torch.complex(u, torch.zeros_like(u))
        sigma = torch.clamp(self.sigma, min=1e-4)
        nu = torch.clamp(self.nu, min=1e-4)

        # Martingale correction: omega = -1/nu * log(1 - theta*nu - 0.5 * sigma^2 * nu)
        omega = -(1.0 / nu) * torch.log(1.0 - self.theta * nu - 0.5 * sigma**2 * nu)

        phi_drift = torch.exp(1j * u_c * (r - q) * T)
        vg_cf = (
            1.0 - 1j * u_c * self.theta * nu + 0.5 * sigma**2 * nu * u_c**2
        ) ** (-T / nu)
        phi_martingale = torch.exp(-1j * u_c * omega * T)

        return phi_drift * vg_cf * phi_martingale

    def cumulants(
        self, T: torch.Tensor, r: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.clamp(self.sigma, min=1e-4)
        nu = torch.clamp(self.nu, min=1e-4)

        c1 = (r - q) * T + self.theta * T
        c2 = sigma**2 * T + self.theta**2 * nu * T
        return c1, c2
