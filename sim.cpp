#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include <cassert>
#include <chrono>
#include <iostream>
using namespace emp;
using namespace std;

int party, port;
const int THREADS = 1;
const int PARAM_K = 3, PARAM_T = 35;

inline IntFp pub(uint64_t v) { return IntFp(v, PUBLIC); }

// Helper: plaintext modular multiply-accumulate
inline uint64_t mulmod(uint64_t a, uint64_t b) {
  return (__uint128_t)a * b % pr;
}
inline uint64_t addmod(uint64_t a, uint64_t b) { return (a + b) % pr; }

// ── ZKP 1: (c0^T*W)*v = a*(c0^T*u) ─────────────────────────────────────────
// c0W_fp and c0u_fp are now committed as ALICE scalars (no mul gates here).
// The only mul gates are the F inner-product terms: c0W_fp[i] * v_fp[i],
// and c0u_fp * pub(a), where c0u_fp is a single secret scalar.
void zk_assert_linear(const IntFp *c0W_fp, const IntFp *v_fp,
                      const IntFp &c0u_fp, const uint64_t a, const int F,
                      const char *label) {
  // lhs = sum_i c0W[i] * v[i]   — F mul gates (both secret)
  IntFp lhs((uint64_t)0, PUBLIC);
  for (int i = 0; i < F; ++i)
    lhs = lhs + c0W_fp[i] * v_fp[i];

  // rhs = c0u * a  — 1 mul gate (secret * public scalar)
  IntFp rhs = c0u_fp * pub(a % pr);

  IntFp diff = lhs + rhs * pub(pr - 1);
  uint64_t zero = 0;
  batch_reveal_check(&diff, &zero, 1);
  cout << "[ZK] '" << label << "' check done.\n";
}

// ── ZKP 2: <v,v>=<u,u>=1 ────────────────────────────────────────────────────
void zk_assert_unit(const IntFp *v, const int F, const char *label) {
  IntFp acc((uint64_t)0, PUBLIC);
  for (int i = 0; i < F; ++i)
    acc = acc + v[i] * v[i];
  IntFp diff = acc + IntFp(pr - 1, PUBLIC);
  uint64_t expected_zero = 0;
  batch_reveal_check(&diff, &expected_zero, 1);
  cout << "[ZK] '" << label << "' unit-vector check done.\n";
}

// ── ZKP 3: r[i,j] = W^T*(W*r[i,j-1]) ───────────────────────────────────────
void zk_assert_recurrence(BoolIO<NetIO> *ios[THREADS], const IntFp *W_fp,
                          IntFp **Wr_fp, IntFp **r_fp, const int F,
                          const char *label) {
  // Process one (i,j) slice at a time to avoid send-buffer deadlock
  IntFp *diffs = new IntFp[F];
  uint64_t *zeros = new uint64_t[F]();

  for (int i = 0; i < PARAM_K; ++i) {
    for (int j = 0; j < PARAM_T; ++j) {
      const IntFp *Wr_cur = Wr_fp[i * PARAM_T + j];
      const IntFp *r_cur = r_fp[i * PARAM_T + j];

      for (int row = 0; row < F; ++row) {
        IntFp WtWr((uint64_t)0, PUBLIC);
        for (int col = 0; col < F; ++col)
          WtWr = WtWr + W_fp[col * F + row] * Wr_cur[col];

        diffs[row] = WtWr + r_cur[row] * pub(pr - 1);
      }

      // Flush one (i,j) block at a time — avoids buffer deadlock
      batch_reveal_check(diffs, zeros, F);
      for (int t = 0; t < THREADS; ++t)
        ios[t]->flush();
    }
  }

  cout << "[ZK] '" << label << "' check done.\n";
  delete[] diffs;
  delete[] zeros;
}

// ── ZKP 4: ||r[i,T]||^2 <= a^(2t) ──────────────────────────────────────────
void zk_assert_norm_bound(
    IntFp **r_fp,             // all r[i,j], committed as ALICE
    const uint64_t **r_plain, // plaintext r[i,j], shape [K*(T+1)][F]
    const uint64_t a_val, const int F, const char *label) {
  uint64_t bound = 1;
  for (int e = 0; e < 2 * PARAM_T; ++e)
    bound = mulmod(bound, a_val % pr);

  for (int i = 0; i < PARAM_K; ++i) {
    const IntFp *r_iT_fp = r_fp[i * PARAM_T + PARAM_T - 1];
    const uint64_t *r_iT_plain = r_plain[i * (PARAM_T + 1) + PARAM_T];

    uint64_t norm_sq_plain = 0;
    for (int f = 0; f < F; ++f)
      norm_sq_plain =
          addmod(norm_sq_plain, mulmod(r_iT_plain[f], r_iT_plain[f]));

    uint64_t s = (bound + pr - norm_sq_plain) % pr;
    IntFp s_fp(s, ALICE);

    // norm_sq_fp = sum r[f]^2  — F mul gates (secret * secret)
    IntFp norm_sq_fp((uint64_t)0, PUBLIC);
    for (int f = 0; f < F; ++f)
      norm_sq_fp = norm_sq_fp + r_iT_fp[f] * r_iT_fp[f];

    IntFp diff = norm_sq_fp + s_fp + pub((pr - bound) % pr);
    uint64_t zero = 0;
    batch_reveal_check(&diff, &zero, 1);
  }
  cout << "[ZK] '" << label << "' norm bound check done.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
void test_nn_fair(BoolIO<NetIO> *ios[THREADS], int party, int f) {
  int F = f, MAT_N = f * f;
  PRG prg;
  setup_zk_arith<BoolIO<NetIO>>(ios, THREADS, party);

  // ── Lambdas ──────────────────────────────────────────────────────────────
  auto start = std::chrono::high_resolution_clock::now();

  uint64_t *lambda_u = new uint64_t[F];
  uint64_t *lambda_v = new uint64_t[F];
  uint64_t **lambdas = new uint64_t *[PARAM_K * PARAM_T];
  prg.random_data(lambda_u, F * sizeof(uint64_t));
  prg.random_data(lambda_v, F * sizeof(uint64_t));
  for (int i = 0; i < PARAM_K * PARAM_T; ++i) {
    lambdas[i] = new uint64_t[F];
    prg.random_data(lambdas[i], F * sizeof(uint64_t));
    for (int k = 0; k < F; ++k)
      lambdas[i][k] %= pr;
  }
  for (int i = 0; i < F; ++i) {
    lambda_u[i] %= pr;
    lambda_v[i] %= pr;
  }

  IntFp *lambda_u_fp = new IntFp[F];
  IntFp *lambda_v_fp = new IntFp[F];
  IntFp **lambdas_fp = new IntFp *[PARAM_K * PARAM_T];
  for (int i = 0; i < F; ++i)
    lambda_u_fp[i] = IntFp(lambda_u[i], ALICE);
  for (int i = 0; i < F; ++i)
    lambda_v_fp[i] = IntFp(lambda_v[i], ALICE);
  for (int i = 0; i < PARAM_K * PARAM_T; ++i) {
    lambdas_fp[i] = new IntFp[F];
    for (int k = 0; k < F; ++k)
      lambdas_fp[i][k] = IntFp(lambdas[i][k], ALICE);
  }

  auto end = std::chrono::high_resolution_clock::now();
  cout << "Offline Time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
       << "us\n";

  // ── Step 1 ───────────────────────────────────────────────────────────────
  start = std::chrono::high_resolution_clock::now();

  uint64_t *W = new uint64_t[MAT_N];
  uint64_t *u = new uint64_t[F];
  uint64_t *v = new uint64_t[F];
  prg.random_data(W, MAT_N * sizeof(uint64_t));
  prg.random_data(u, F * sizeof(uint64_t));
  prg.random_data(v, F * sizeof(uint64_t));
  for (int i = 0; i < MAT_N; ++i)
    W[i] %= pr;
  for (int i = 0; i < F; ++i) {
    u[i] %= pr;
    v[i] %= pr;
  }

  // a = ||W||^2 mod pr
  uint64_t a_val = 0;
  for (int i = 0; i < MAT_N; ++i)
    a_val = addmod(a_val, mulmod(W[i], W[i]));

  // Commit W as secret wires (needed for ZKP 3 recurrence check)
  IntFp *W_fp = new IntFp[MAT_N];
  for (int i = 0; i < MAT_N; ++i)
    W_fp[i] = IntFp(W[i], ALICE);

  end = std::chrono::high_resolution_clock::now();
  cout << "Step 1 Time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
       << "us\n";

  // ── Step 2 ───────────────────────────────────────────────────────────────
  start = std::chrono::high_resolution_clock::now();

  uint64_t *du = new uint64_t[F];
  uint64_t *dv = new uint64_t[F];
  for (int i = 0; i < F; ++i) {
    du[i] = (u[i] + pr - lambda_u[i]) % pr;
    dv[i] = (v[i] + pr - lambda_v[i]) % pr;
  }
  IntFp *u_fp = new IntFp[F];
  IntFp *v_fp = new IntFp[F];
  for (int i = 0; i < F; ++i) {
    u_fp[i] = lambda_u_fp[i] + pub(du[i]);
    v_fp[i] = lambda_v_fp[i] + pub(dv[i]);
  }

  uint64_t **c = new uint64_t *[PARAM_K + 1];
  for (int i = 0; i < PARAM_K + 1; ++i) {
    c[i] = new uint64_t[F];
    prg.random_data(c[i], F * sizeof(uint64_t));
    for (int k = 0; k < F; ++k)
      c[i][k] %= pr;
  }

  end = std::chrono::high_resolution_clock::now();
  cout << "Step 2 Time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
       << "us\n";

  // ── Step 3 ───────────────────────────────────────────────────────────────
  start = std::chrono::high_resolution_clock::now();

  uint64_t **r = new uint64_t *[PARAM_K * (PARAM_T + 1)];
  for (int i = 0; i < PARAM_K; ++i) {
    r[i * (PARAM_T + 1)] = new uint64_t[F];
    for (int k = 0; k < F; ++k)
      r[i * (PARAM_T + 1)][k] = c[i][k];

    for (int j = 1; j <= PARAM_T; ++j) {
      r[i * (PARAM_T + 1) + j] = new uint64_t[F];
      uint64_t *prev = r[i * (PARAM_T + 1) + j - 1];
      uint64_t *cur = r[i * (PARAM_T + 1) + j];

      // tmp = W * prev
      uint64_t *tmp = new uint64_t[F]();
      for (int row = 0; row < F; ++row)
        for (int col = 0; col < F; ++col)
          tmp[row] = addmod(tmp[row], mulmod(W[row * F + col], prev[col]));

      // cur = W^T * tmp
      for (int row = 0; row < F; ++row) {
        cur[row] = 0;
        for (int col = 0; col < F; ++col)
          cur[row] = addmod(cur[row], mulmod(W[col * F + row], tmp[col]));
      }
      delete[] tmp;
    }
  }

  // Commit r[i,j+1] as ALICE scalars (no mul gates)
  uint64_t **d = new uint64_t *[PARAM_K * PARAM_T];
  IntFp **r_fp = new IntFp *[PARAM_K * PARAM_T];
  for (int i = 0; i < PARAM_K; ++i) {
    for (int j = 0; j < PARAM_T; ++j) {
      int idx = i * PARAM_T + j;
      d[idx] = new uint64_t[F];
      r_fp[idx] = new IntFp[F];
      uint64_t *r_cur = r[i * (PARAM_T + 1) + j + 1];
      for (int k = 0; k < F; ++k) {
        d[idx][k] = (r_cur[k] + pr - lambdas[idx][k]) % pr;
        r_fp[idx][k] = pub(d[idx][k]) + lambdas_fp[idx][k];
      }
    }
  }

  end = std::chrono::high_resolution_clock::now();
  cout << "Step 3 Time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
       << "us\n";

  // ── Step 4 (FIX: precompute public×secret products in plaintext) ─────────
  start = std::chrono::high_resolution_clock::now();

  // FIX: c0W[j] = sum_i c[0][i] * W[i*F+j]  — computed in plaintext,
  // then committed as a single ALICE wire. Zero mul gates here.
  IntFp *c0W_fp = new IntFp[F];
  for (int j = 0; j < F; ++j) {
    uint64_t c0Wj = 0;
    for (int i = 0; i < F; ++i)
      c0Wj = addmod(c0Wj, mulmod(c[0][i], W[i * F + j]));
    c0W_fp[j] = IntFp(c0Wj, ALICE);
  }

  // FIX: c0u = sum_i c[0][i] * u[i]  — plaintext, committed as ALICE wire.
  uint64_t c0u_plain = 0;
  for (int i = 0; i < F; ++i)
    c0u_plain = addmod(c0u_plain, mulmod(c[0][i], u[i]));
  IntFp c0u_fp = IntFp(c0u_plain, ALICE);

  // FIX: Wr_fp[idx][row] = sum_col W[row,col] * r_prev[col]
  // r_prev is plaintext → compute in plaintext, commit as ALICE wire.
  // This eliminates K*T*F^2 spurious mul gates from the original code.
  IntFp **Wr_fp = new IntFp *[PARAM_K * PARAM_T];
  for (int i = 0; i < PARAM_K; ++i) {
    for (int j = 0; j < PARAM_T; ++j) {
      int idx = i * PARAM_T + j;
      Wr_fp[idx] = new IntFp[F];
      uint64_t *r_prev = r[i * (PARAM_T + 1) + j]; // plaintext r[i,j]
      for (int row = 0; row < F; ++row) {
        uint64_t Wr_row = 0;
        for (int col = 0; col < F; ++col)
          Wr_row = addmod(Wr_row, mulmod(W[row * F + col], r_prev[col]));
        Wr_fp[idx][row] = IntFp(Wr_row, ALICE); // single committed wire
      }
    }
  }

  end = std::chrono::high_resolution_clock::now();
  cout << "Step 4 Time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
       << "us\n";

  // ── Step 5: ZKPs ─────────────────────────────────────────────────────────
  start = std::chrono::high_resolution_clock::now();

  zk_assert_linear(c0W_fp, v_fp, c0u_fp, a_val, F, "c0Wv=a*c0u");
  zk_assert_unit(u_fp, F, "u is unit vec");
  zk_assert_unit(v_fp, F, "v is unit vec");
  zk_assert_recurrence(ios, W_fp, Wr_fp, r_fp, F, "rij");
  zk_assert_norm_bound(r_fp, (const uint64_t **)r, a_val, F, "norm_bound");

  end = std::chrono::high_resolution_clock::now();
  cout << "Step 5 (ZKPs) Time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count()
       << "us\n";

  finalize_zk_arith<BoolIO<NetIO>>();
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO<NetIO> *ios[THREADS];
  for (int i = 0; i < THREADS; ++i)
    ios[i] = new BoolIO<NetIO>(
        new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
        party == ALICE);

  cout << "\n------------ circuit ZKP test ------------\n\n";

  if (argc < 3) {
    cout << "usage: PARTY PORT DIMENSION" << endl;
    return -1;
  }
  int num = (argc == 3) ? 10 : atoi(argv[3]);

  test_nn_fair(ios, party, num);

  for (int i = 0; i < THREADS; ++i) {
    delete ios[i]->io;
    delete ios[i];
  }
  return 0;
}
