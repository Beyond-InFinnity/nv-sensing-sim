import numpy as np
import pytest

from nvsim.constants import A_PAR_N14_HZ, D_GS_HZ, GAMMA_E_HZ_PER_T
from nvsim.odmr import odmr_spectrum, odmr_spectrum_n14


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


def test_n14_triplet_spacing_2p16_mhz():
    bz = 2e-3  # separate the two electronic transitions cleanly
    f = np.linspace(2.80e9, 2.83e9, 60001)  # around f_minus
    s = odmr_spectrum_n14(f, b_nv_t=(0, 0, bz), fwhm_hz=0.4e6)
    dips = _dip_freqs(f, s)
    assert len(dips) == 3
    df = f[1] - f[0]
    assert dips[1] - dips[0] == pytest.approx(abs(A_PAR_N14_HZ), abs=2 * df)
    assert dips[2] - dips[1] == pytest.approx(abs(A_PAR_N14_HZ), abs=2 * df)


def test_triplet_contrast_is_one_third_each():
    f = np.linspace(2.80e9, 2.83e9, 60001)
    s = odmr_spectrum_n14(f, b_nv_t=(0, 0, 2e-3), contrast=0.24, fwhm_hz=0.4e6)
    # well-separated lines: each dip depth ~ contrast/3
    assert 1 - s.min() == pytest.approx(0.24 / 3, rel=0.05)


def test_power_broadening_sqrt_1_plus_s():
    f = np.linspace(2.865e9, 2.875e9, 40001)

    def fwhm_of_center_line(s_par):
        s = odmr_spectrum_n14(f, fwhm_hz=0.5e6, a_par_hz=20e6, saturation=s_par)
        # a_par 20 MHz (unphysical, test-only) isolates the mI=0 line at D
        depth = 1 - s.min()
        half = 1 - depth / 2
        below = f[s < half]
        return below.max() - below.min()

    w1, w4 = fwhm_of_center_line(1.0), fwhm_of_center_line(4.0)
    assert w4 / w1 == pytest.approx(np.sqrt(5) / np.sqrt(2), rel=0.03)


def test_saturation_scales_contrast():
    # a_par 50 MHz (unphysical, test-only) isolates the mI=0 line at D so the
    # dip depth is free of neighbor-line overlap, which grows with broadening
    f = np.linspace(2.86e9, 2.88e9, 20001)
    depths = []
    for s_par in (0.5, 2.0, 8.0):
        s = odmr_spectrum_n14(f, contrast=0.3, a_par_hz=50e6, saturation=s_par)
        depths.append(1 - s.min())
    sats = np.array([0.5, 2.0, 8.0])
    ratios = np.array(depths) / (0.3 * sats / (1 + sats))
    np.testing.assert_allclose(ratios, ratios[0], rtol=0.1)  # ∝ s/(1+s)
