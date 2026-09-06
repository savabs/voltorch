"""Differentiable option pricing in PyTorch.

Every model here is an `nn.Module` and every price is differentiable with
respect to the spot, the strike, the maturity AND the model parameters. That is
the whole point: greeks come from autograd rather than from finite differences,
calibration is gradient descent rather than a derivative-free optimiser, and a
pricer can sit inside a network and receive gradients through it.

The hard part of this library is not the formulae, which are in the papers. It
is the handful of numerical failures that silently poison a naive
implementation -- a branch cut that flips sign mid-integration, a 0/0 that is
finite in the limit but NaN in floating point, a drift term whose omission
looks like a small pricing error and is actually a lost martingale property.
Each guard is documented at its site with the symptom it prevents.
"""

from .options import (
    BlackScholes,
    BaroneAdesiWhaley,
    FourierCOS,
    HestonCOS,
    BatesCOS,
    MertonCOS,
    VarianceGammaCOS,
    implied_volatility,
)
from .volatility import (
    SVIParameterization,
    SABRModel,
    ImpliedVolatilitySurface,
)
from .rough import RoughBergomiModel, estimate_hurst_exponent
from .ssvi import SSVI
from .chain import FitReport, fit_chain, otm
from .arbitrage import butterfly_violations, calendar_violations, durrleman_g, vertical_violations

__version__ = "0.1.0"

__all__ = [
    "SSVI", "fit_chain", "FitReport", "otm", "butterfly_violations", "calendar_violations", "durrleman_g", "vertical_violations",
    "BlackScholes",
    "BaroneAdesiWhaley",
    "FourierCOS",
    "HestonCOS",
    "BatesCOS",
    "MertonCOS",
    "VarianceGammaCOS",
    "implied_volatility",
    "SVIParameterization",
    "SABRModel",
    "ImpliedVolatilitySurface",
    "RoughBergomiModel",
    "estimate_hurst_exponent",
    "__version__",
]

# `voltorch.sde` is deliberately NOT imported here: it needs torchsde, which
# pulls in a solver stack most users of the pricers never touch. Install with
# `pip install voltorch[sde]` and import it directly.
