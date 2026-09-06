import pytest
import torch

from voltorch.options import BlackScholes

# Recorded from Deribit on 2026-09-06 (get_book_summary_by_currency): instrument,
# days to expiry, underlying_price (F), strike, put?, mark_iv, mark_price in BTC.
RECORDED = [
    ("BTC-25DEC26-250000-P", 110, 80913.0, 250000.0, True, None, 2.08998),
    ("BTC-30OCT26-85000-P", 54, 80278.0, 85000.0, True, None, 0.09376),
    ("BTC-25JUN27-96000-C", 292, 82771.0, 96000.0, False, None, 0.09261),
]


@pytest.mark.parametrize("name,days,F,K,is_put,_,mark_btc", RECORDED)
def test_price_in_coin_times_forward_is_black76(name, days, F, K, is_put, _, mark_btc):
    """The convention the loader relies on. If Deribit changes it, this fails
    loudly instead of biasing every implied vol by the basis."""
    from voltorch.options import implied_volatility_bisect
    T = torch.tensor([days / 365.0], dtype=torch.float64)
    Ft, Kt, r = torch.tensor([F], dtype=torch.float64), torch.tensor([K], dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    usd = torch.tensor([mark_btc * F], dtype=torch.float64)
    sigma = implied_volatility_bisect(Ft, Kt, T, r, usd, is_call=not is_put)
    back = BlackScholes()(Ft, Kt, T, r, sigma, is_call=not is_put)
    assert abs(float(back) / float(usd) - 1) < 1e-6
