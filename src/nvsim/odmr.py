"""CW-ODMR at the rate-equation level: unit-normalized fluorescence with one
Lorentzian dip per exact eigen-transition (see docs/PHYSICS.md)."""
import numpy as np

from .constants import D_GS_HZ
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
