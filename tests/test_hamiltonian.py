import numpy as np
import pytest

from nvsim.constants import D_GS_HZ, GAMMA_E_HZ_PER_T, NV_AXES
from nvsim.hamiltonian import b_lab_to_nv, h_gs, transition_frequencies


def test_zero_field_splitting_at_2p870_ghz():
    f_minus, f_plus = transition_frequencies(h_gs())
    assert f_minus == pytest.approx(2.870e9, rel=1e-12)
    assert f_plus == pytest.approx(2.870e9, rel=1e-12)


def test_strain_splits_by_2e():
    e = 5e6
    f_minus, f_plus = transition_frequencies(h_gs(e_hz=e))
    assert f_plus - f_minus == pytest.approx(2 * e, rel=1e-9)
    assert (f_plus + f_minus) / 2 == pytest.approx(D_GS_HZ, rel=1e-12)


def test_axial_zeeman_splitting_slope_28p02_ghz_per_t():
    for bz in (0.5e-3, 1e-3, 3e-3):
        f_minus, f_plus = transition_frequencies(h_gs(b_nv_t=(0, 0, bz)))
        assert f_plus - f_minus == pytest.approx(2 * GAMMA_E_HZ_PER_T * bz, rel=1e-6)


def test_transverse_field_is_second_order():
    bx = 1e-3
    f_minus, f_plus = transition_frequencies(h_gs(b_nv_t=(bx, 0, 0)))
    # splitting from transverse field is O((γB)²/D), not linear
    assert f_plus - f_minus < 2 * GAMMA_E_HZ_PER_T * bx * 0.05


def test_nv_axes_are_unit_111_directions():
    assert NV_AXES.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(NV_AXES, axis=1), 1.0, rtol=1e-12)
    # pairwise angle between distinct <111> axes: cos = -1/3
    for i in range(4):
        for j in range(i + 1, 4):
            assert NV_AXES[i] @ NV_AXES[j] == pytest.approx(-1 / 3, rel=1e-9)


def test_b_lab_to_nv_projection():
    b_lab = 1e-3 * NV_AXES[0]  # field along orientation 0's axis
    b0 = b_lab_to_nv(b_lab, 0)
    assert b0[2] == pytest.approx(1e-3, rel=1e-12)  # fully axial for orientation 0
    assert np.hypot(b0[0], b0[1]) == pytest.approx(0, abs=1e-15)
    for k in (1, 2, 3):  # cos(theta) = -1/3 for the other three orientations
        bk = b_lab_to_nv(b_lab, k)
        assert bk[2] == pytest.approx(-1e-3 / 3, rel=1e-9)
        assert np.linalg.norm(bk) == pytest.approx(1e-3, rel=1e-12)
