import numpy as np
import pytest

from nvsim.constants import D_GS_HZ, GAMMA_E_HZ_PER_T
from nvsim.odmr import odmr_spectrum


def _dip_freqs(f, s):
    """Frequencies of local minima of s."""
    idx = np.where((s[1:-1] < s[:-2]) & (s[1:-1] < s[2:]))[0] + 1
    return f[idx]


def test_single_dip_at_d_for_zero_field_zero_strain():
    f = np.linspace(2.80e9, 2.94e9, 4001)
    s = odmr_spectrum(f)
    dips = _dip_freqs(f, s)
    assert len(dips) == 1
    assert dips[0] == pytest.approx(D_GS_HZ, abs=f[1] - f[0])


def test_two_dips_split_linearly_in_axial_field():
    f = np.linspace(2.70e9, 3.04e9, 8001)
    for bz in (1e-3, 2e-3, 4e-3):
        s = odmr_spectrum(f, b_nv_t=(0, 0, bz))
        dips = _dip_freqs(f, s)
        assert len(dips) == 2
        assert dips[1] - dips[0] == pytest.approx(
            2 * GAMMA_E_HZ_PER_T * bz, abs=2 * (f[1] - f[0])
        )


def test_contrast_at_dip():
    f = np.linspace(2.86e9, 2.88e9, 2001)
    s = odmr_spectrum(f, contrast=0.15)
    # both transitions degenerate at D -> dips add; each carries `contrast`
    assert s.min() == pytest.approx(1 - 2 * 0.15, rel=1e-3)
    assert s.max() <= 1.0
