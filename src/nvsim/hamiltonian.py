"""NV ground-state spin-1 Hamiltonian. H/h in Hz; B in tesla; NV frame z along the NV axis.

Conventions per docs/PHYSICS.md: H/h = D*Sz^2 + E*(Sx^2 - Sy^2) + (gamma_e/2pi)*(B . S).
Transition frequencies are eigenvalue differences from the eigenstate with maximum
overlap with |ms=0>; valid while the ms=0 character is well defined (gamma*B << D).
"""
import numpy as np
import qutip

from .constants import D_GS_HZ, GAMMA_E_HZ_PER_T, NV_AXES

SX = qutip.jmat(1, "x")
SY = qutip.jmat(1, "y")
SZ = qutip.jmat(1, "z")
_MS0 = qutip.basis(3, 1)  # jmat basis order: m = +1, 0, -1


def h_gs(d_hz=D_GS_HZ, e_hz=0.0, b_nv_t=(0.0, 0.0, 0.0)):
    """Ground-state Hamiltonian H/h (Hz), B given in the NV frame (tesla)."""
    bx, by, bz = b_nv_t
    return (
        d_hz * SZ**2
        + e_hz * (SX**2 - SY**2)
        + GAMMA_E_HZ_PER_T * (bx * SX + by * SY + bz * SZ)
    )


def transition_frequencies(h):
    """(f_minus, f_plus) in Hz: transitions from the mostly-|ms=0> eigenstate."""
    evals, evecs = h.eigenstates()
    i0 = int(np.argmax([abs(_MS0.overlap(v)) ** 2 for v in evecs]))
    others = sorted(evals[i] - evals[i0] for i in range(3) if i != i0)
    return float(others[0]), float(others[1])


def nv_frame(orientation):
    """Orthonormal (x, y, z) rows for one NV orientation; z along NV_AXES[orientation]."""
    z = NV_AXES[orientation]
    helper = np.array([0.0, 0.0, 1.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def b_lab_to_nv(b_lab_t, orientation):
    """Project a lab-frame B (tesla, 3-vector) into one NV orientation's frame."""
    return nv_frame(orientation) @ np.asarray(b_lab_t, dtype=float)
