#!/usr/bin/env python3
import numpy as np
from qr_transport import audit, jump_alpha, jump_transport, metric, path_incidence_qbase, perturb_carrier, radius_from_alpha, transport


def test_exact_transport():
    R = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    q, _, _ = path_incidence_qbase(len(R))
    s = transport(R, q)
    assert s.metric()["spectral_match_percent"] == 100.0


def test_jump_law():
    R = np.array([1.0, 1.5, 3.0, 4.0, 9.0])
    q, _, _ = path_incidence_qbase(len(R))
    state = transport(R, q)
    q2 = perturb_carrier(q, eps=1e-2)
    jump = jump_transport(state, q2)
    assert jump.metric()["spectral_match_percent"] == 100.0


def test_direct_jump_formula():
    R = np.array([2.0, 4.0, 7.0, 11.0])
    q, _, _ = path_incidence_qbase(len(R))
    state = transport(R, q)
    q2 = perturb_carrier(q, eps=1e-4)
    alpha2 = jump_alpha(state.alpha, q, q2)
    R2 = radius_from_alpha(alpha2, q2, state.s_star)
    assert metric(R2, R)["spectral_match_percent"] == 100.0


def test_audit():
    R = np.exp(np.linspace(0.0, 1.0, 12))
    q, _, _ = path_incidence_qbase(len(R))
    cert = audit(R, q)
    assert cert["canonical"]["metric"]["spectral_match_percent"] == 100.0
    assert cert["permutation_jump"]["metric"]["spectral_match_percent"] == 100.0
    assert cert["phase_scrambled_signal"]["metric"]["spectral_match_percent"] == 100.0


if __name__ == "__main__":
    test_exact_transport(); test_jump_law(); test_direct_jump_formula(); test_audit()
    print("all tests passed")
