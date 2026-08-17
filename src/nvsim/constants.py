"""Physical constants for the NV ground-state model. Units: SI, frequencies in Hz."""
import numpy as np

D_GS_HZ = 2.870e9            # zero-field splitting
GAMMA_E_HZ_PER_T = 28.02e9   # electron gyromagnetic ratio / 2pi, magnitude
DD_DT_HZ_PER_K = -74e3       # thermal shift of D (Phase 1)
A_PAR_N14_HZ = -2.16e6       # 14N axial hyperfine (Phase 1)

# Four NV orientations along <111>; rows are unit vectors in the diamond cubic frame.
NV_AXES = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
) / np.sqrt(3)
