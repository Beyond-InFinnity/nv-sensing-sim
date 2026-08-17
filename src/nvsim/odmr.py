"""CW-ODMR at the rate-equation level: unit-normalized fluorescence with one
Lorentzian dip per exact eigen-transition (see docs/PHYSICS.md)."""
import numpy as np

from .constants import A_PAR_N14_HZ, D_GS_HZ
from .hamiltonian import h_gs, transition_frequencies


def odmr_spectrum(f_hz, d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0.0, 0.0, 0.0),
                  contrast=0.2, fwhm_hz=8e6):
    """Normalized fluorescence vs MW frequency f_hz (array-like, Hz)."""
    f = np.asarray(f_hz, dtype=float)
    hwhm = fwhm_hz / 2
    s = np.ones_like(f)
    for f0 in transition_frequencies(h_gs(d_hz, e_hz, b_nv_t)):
        s -= contrast * hwhm**2 / ((f - f0) ** 2 + hwhm**2)
    return s


def odmr_spectrum_n14(f_hz, d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0.0, 0.0, 0.0),
                      contrast=0.2, fwhm_hz=1e6, a_par_hz=None, saturation=None):
    """CW-ODMR with the 14N hyperfine triplet and optional power broadening.

    Each electronic transition (toward ms_target = ±1) splits into three lines
    at f_trans + ms_target * A_par * mI, mI in {-1, 0, +1}, contrast/3 each
    (static-hyperfine approximation, docs/PHYSICS.md). saturation = s applies
    FWHM * sqrt(1+s) and contrast * s/(1+s).
    """
    if a_par_hz is None:
        a_par_hz = A_PAR_N14_HZ
    f = np.asarray(f_hz, dtype=float)
    c_eff, w_eff = contrast, fwhm_hz
    if saturation is not None:
        c_eff = contrast * saturation / (1 + saturation)
        w_eff = fwhm_hz * np.sqrt(1 + saturation)
    hwhm = w_eff / 2
    f_minus, f_plus = transition_frequencies(h_gs(d_hz, e_hz, b_nv_t))
    s = np.ones_like(f)
    for ms_target, f0 in ((-1, f_minus), (+1, f_plus)):
        for mi in (-1, 0, 1):
            fc = f0 + ms_target * a_par_hz * mi
            s -= (c_eff / 3) * hwhm**2 / ((f - fc) ** 2 + hwhm**2)
    return s
