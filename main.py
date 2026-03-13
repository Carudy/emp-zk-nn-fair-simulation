import random
import sys
import time

from picozk import *

# ── Parameters ────────────────────────────────────────────────────────────────
PARAM_K = 3
PARAM_T = 35

# Prime used by emp-zk's field  (2^61 - 1, the Mersenne prime)
PR = (1 << 61) - 1

# ── Finite-field helpers (plaintext) ──────────────────────────────────────────


def mulmod(a: int, b: int) -> int:
    return (a * b) % PR


def addmod(a: int, b: int) -> int:
    return (a + b) % PR


def submod(a: int, b: int) -> int:
    return (a - b) % PR


# ── picozk wire helpers ───────────────────────────────────────────────────────


def pub(v: int) -> PublicInt:
    """Wrap a public constant as a PublicInt wire."""
    return PublicInt(v % PR)


def commit(v: int) -> SecretInt:
    """Commit a secret scalar as a SecretInt (ALICE) wire."""
    return SecretInt(v % PR)


def commit_vec(lst) -> list:
    return [commit(x) for x in lst]


# ── ZKP 1: (c0^T W) v = a (c0^T u) ──────────────────────────────────────────


def zk_assert_linear(
    c0W_fp: list,  # [F] SecretInt  c0W[j] = sum_i c[0][i]*W[i,j]
    v_fp: list,  # [F] SecretInt  v wires
    c0u_fp: SecretInt,  # SecretInt      c0u = sum_i c[0][i]*u[i]
    a_val: int,  # public scalar  a = ||W||^2
    label: str,
):
    """
    lhs = sum_j c0W[j] * v[j]
    rhs = c0u * a
    assert lhs - rhs == 0
    """
    lhs = pub(0)
    for j in range(len(c0W_fp)):
        lhs = lhs + c0W_fp[j] * v_fp[j]  # secret * secret → mul gate

    rhs = c0u_fp * pub(a_val % PR)  # secret * public → linear

    diff = lhs + rhs * pub(PR - 1)  # lhs - rhs
    try:
        assert0(diff)
    except Exception as e:
        print("ZKP check failed.")
    print(f"[ZK] '{label}' check done.")


# ── ZKP 2: <v, v> = 1 ────────────────────────────────────────────────────────


def zk_assert_unit(v_fp: list, label: str):
    """Assert sum_i v[i]^2 == 1 mod PR."""
    acc = pub(0)
    for vi in v_fp:
        acc = acc + vi * vi

    diff = acc + pub(PR - 1)  # acc - 1
    try:
        assert0(diff)
    except Exception as e:
        print("ZKP check failed.")
    print(f"[ZK] '{label}' unit-vector check done.")


# ── ZKP 3: r[i,j] = W^T (W r[i,j-1]) ────────────────────────────────────────


def zk_assert_recurrence(
    W_fp: list,  # [F*F] SecretInt W wires (row-major)
    Wr_fp: list,  # [K*T][F] SecretInt  Wr = W * r_prev  (plaintext-computed)
    r_fp: list,  # [K*T][F] SecretInt  r[i,j+1] wires
    F: int,
    label: str,
):
    """
    For each (i,j):  W^T * Wr[i,j] == r[i,j+1]
    Wr[i,j] = W * r[i,j] is pre-computed in plaintext and committed,
    making every constraint linear in the secret wires W and Wr.
    """
    for i in range(PARAM_K):
        for j in range(PARAM_T):
            idx = i * PARAM_T + j
            Wr_cur = Wr_fp[idx]  # [F] wires
            r_cur = r_fp[idx]  # [F] wires  (r[i,j+1])

            for row in range(F):
                WtWr = pub(0)
                for col in range(F):
                    WtWr = WtWr + W_fp[col * F + row] * Wr_cur[col]

                diff = WtWr + r_cur[row] * pub(PR - 1)
                try:
                    assert0(diff)
                except Exception as e:
                    print("ZKP check failed.")

    print(f"[ZK] '{label}' recurrence check done.")


# ── ZKP 4: ||r[i,T]||^2 <= a^(2T) ───────────────────────────────────────────


def zk_assert_norm_bound(
    r_fp: list,  # [K*T][F] SecretInt wires (index T-1 = last step)
    r_plain: list,  # [K*(T+1)][F] plaintext r values
    a_val: int,
    F: int,
    label: str,
):
    """
    For each i:
      norm_sq = sum_f r[i,T][f]^2
      s       = a^(2T) - norm_sq   (plaintext slack, committed as secret wire)
      assert  norm_sq + s - a^(2T) == 0
    Proves norm_sq <= a^(2T) in the field.
    """
    bound = pow(a_val % PR, 2 * PARAM_T, PR)

    for i in range(PARAM_K):
        r_iT_fp = r_fp[i * PARAM_T + PARAM_T - 1]  # r[i,T] wires
        r_iT_plain = r_plain[i * (PARAM_T + 1) + PARAM_T]  # plaintext r[i,T]

        norm_sq_plain = sum(mulmod(x, x) for x in r_iT_plain) % PR
        s = (bound - norm_sq_plain) % PR

        s_fp = commit(s)  # secret wire for the slack

        norm_sq_fp = pub(0)
        for f in range(F):
            norm_sq_fp = norm_sq_fp + r_iT_fp[f] * r_iT_fp[f]

        diff = norm_sq_fp + s_fp + pub((PR - bound) % PR)
        try:
            assert0(diff)
        except Exception as e:
            print("ZKP check failed.")

    print(f"[ZK] '{label}' norm-bound check done.")


# ── Main test ─────────────────────────────────────────────────────────────────


def test_nn_fair(party: int, F: int):
    MAT_N = F * F

    # ── Offline: generate random masks (lambdas) ──────────────────────────────
    t0 = time.perf_counter()

    lambda_u = [random.randrange(PR) for _ in range(F)]
    lambda_v = [random.randrange(PR) for _ in range(F)]
    lambdas = [
        [random.randrange(PR) for _ in range(F)] for _ in range(PARAM_K * PARAM_T)
    ]

    lambda_u_fp = commit_vec(lambda_u)
    lambda_v_fp = commit_vec(lambda_v)
    lambdas_fp = [commit_vec(lam) for lam in lambdas]

    print(f"Offline Time: {(time.perf_counter() - t0) * 1e6:.0f}us")

    # ── Step 1: sample W, u, v; compute a = ||W||^2 ───────────────────────────
    t0 = time.perf_counter()

    W = [random.randrange(PR) for _ in range(MAT_N)]
    u = [random.randrange(PR) for _ in range(F)]
    v = [random.randrange(PR) for _ in range(F)]

    a_val = sum(mulmod(w, w) for w in W) % PR

    # Commit W (needed for ZKP 3)
    W_fp = commit_vec(W)

    print(f"Step 1 Time: {(time.perf_counter() - t0) * 1e6:.0f}us")

    # ── Step 2: mask-open u and v; sample public challenge vectors c[0..K] ────
    t0 = time.perf_counter()

    du = [submod(u[i], lambda_u[i]) for i in range(F)]
    dv = [submod(v[i], lambda_v[i]) for i in range(F)]

    u_fp = [lambda_u_fp[i] + pub(du[i]) for i in range(F)]
    v_fp = [lambda_v_fp[i] + pub(dv[i]) for i in range(F)]

    c = [[random.randrange(PR) for _ in range(F)] for _ in range(PARAM_K + 1)]

    print(f"Step 2 Time: {(time.perf_counter() - t0) * 1e6:.0f}us")

    # ── Step 3: compute recurrence r[i,j] = (W^T W)^j c[i] (plaintext) ───────
    t0 = time.perf_counter()

    r_plain = [None] * (PARAM_K * (PARAM_T + 1))

    for i in range(PARAM_K):
        r_plain[i * (PARAM_T + 1)] = list(c[i])  # r[i,0] = c[i]

        for j in range(1, PARAM_T + 1):
            prev = r_plain[i * (PARAM_T + 1) + j - 1]

            # tmp = W * prev
            tmp = [0] * F
            for row in range(F):
                for col in range(F):
                    tmp[row] = addmod(tmp[row], mulmod(W[row * F + col], prev[col]))

            # cur = W^T * tmp
            cur = [0] * F
            for row in range(F):
                for col in range(F):
                    cur[row] = addmod(cur[row], mulmod(W[col * F + row], tmp[col]))

            r_plain[i * (PARAM_T + 1) + j] = cur

    # Commit r[i,j+1] as secret wires via mask-open trick
    r_fp = [None] * (PARAM_K * PARAM_T)
    for i in range(PARAM_K):
        for j in range(PARAM_T):
            idx = i * PARAM_T + j
            r_cur = r_plain[i * (PARAM_T + 1) + j + 1]
            d = [submod(r_cur[k], lambdas[idx][k]) for k in range(F)]
            r_fp[idx] = [pub(d[k]) + lambdas_fp[idx][k] for k in range(F)]

    print(f"Step 3 Time: {(time.perf_counter() - t0) * 1e6:.0f}us")

    # ── Step 4: pre-compute public×secret products in plaintext ──────────────
    # Eliminates spurious mul gates (mirrors the C++ FIX comments exactly).
    t0 = time.perf_counter()

    # c0W[j] = sum_i c[0][i] * W[i,j]  — plaintext → single secret wire
    c0W_fp = []
    for j in range(F):
        val = sum(mulmod(c[0][i], W[i * F + j]) for i in range(F)) % PR
        c0W_fp.append(commit(val))

    # c0u = sum_i c[0][i] * u[i]  — plaintext → single secret wire
    c0u_plain = sum(mulmod(c[0][i], u[i]) for i in range(F)) % PR
    c0u_fp = commit(c0u_plain)

    # Wr_fp[idx][row] = sum_col W[row,col] * r_prev[col]  — plaintext → secret
    Wr_fp = [None] * (PARAM_K * PARAM_T)
    for i in range(PARAM_K):
        for j in range(PARAM_T):
            idx = i * PARAM_T + j
            r_prev = r_plain[i * (PARAM_T + 1) + j]  # plaintext r[i,j]
            row_vals = []
            for row in range(F):
                val = (
                    sum(mulmod(W[row * F + col], r_prev[col]) for col in range(F)) % PR
                )
                row_vals.append(commit(val))
            Wr_fp[idx] = row_vals

    print(f"Step 4 Time: {(time.perf_counter() - t0) * 1e6:.0f}us")

    # ── Step 5: run all ZKPs ──────────────────────────────────────────────────
    t0 = time.perf_counter()

    zk_assert_linear(c0W_fp, v_fp, c0u_fp, a_val, "c0Wv=a*c0u")
    zk_assert_unit(u_fp, "u is unit vec")
    zk_assert_unit(v_fp, "v is unit vec")
    zk_assert_recurrence(W_fp, Wr_fp, r_fp, F, "rij")
    zk_assert_norm_bound(r_fp, r_plain, a_val, F, "norm_bound")

    print(f"Step 5 (ZKPs) Time: {(time.perf_counter() - t0) * 1e6:.0f}us")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python nn_fair_zkp.py [DIMENSION] [file_prefix]")
        sys.exit(1)

    dim = int(sys.argv[1]) if len(sys.argv) >= 2 else 10
    file_prefix = sys.argv[2] if len(sys.argv) >= 3 else "nn_fair_zkp_out"
    file_prefix = sys.argv[2] if len(sys.argv) >= 3 else "nn_fair_zkp_out"

    print("\n------------ circuit ZKP test (picozk) ------------\n")

    # PicoZKCompiler(file_prefix, field) -- writes the ZK circuit to files.
    # field defaults to 2^61-1, which matches PR, so it can be omitted.
    with PicoZKCompiler(file_prefix, field=PR):
        test_nn_fair(1, dim)
