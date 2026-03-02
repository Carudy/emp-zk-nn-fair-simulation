#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace emp;
using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// Protocol constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int       F     = 10;
static constexpr int       THREADS = 1;
static constexpr long long SCALE = 1LL << 20;   // fixed-point scale

// ─────────────────────────────────────────────────────────────────────────────
static uint64_t fp_add_(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)(a + b) % pr);
}
static uint64_t fp_sub_(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a + pr - b) % pr);
}
static uint64_t fp_mul_(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * b % pr);
}
static uint64_t fp_neg_(uint64_t a) {
    return a == 0 ? 0ULL : (uint64_t)(pr - a);
}
static uint64_t fp_from_double(double x) {
    long long v = llround(x * SCALE);
    return (uint64_t)(((v % (long long)pr) + (long long)pr) % (long long)pr);
}

// ─────────────────────────────────────────────────────────────────────────────
struct SingularPair { double sigma; vector<double> u, v; };

static SingularPair power_iter(const vector<vector<double>>& W,
                               int max_iter = 2000, double tol = 1e-12)
{
    int n = (int)W.size();
    vector<double> v(n, 1.0/sqrt((double)n)), u(n), vp(n);
    double sigma = 0;
    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < n; ++i) {
            u[i] = 0;
            for (int j = 0; j < n; ++j) u[i] += W[i][j] * v[j];
        }
        sigma = 0; for (double x : u) sigma += x*x; sigma = sqrt(sigma);
        for (double& x : u) x /= sigma;
        vp = v;
        for (int j = 0; j < n; ++j) {
            v[j] = 0;
            for (int i = 0; i < n; ++i) v[j] += W[i][j] * u[i];
        }
        double nv = 0; for (double x : v) nv += x*x; nv = sqrt(nv);
        for (double& x : v) x /= nv;
        double d = 0;
        for (int j = 0; j < n; ++j) { double dd = v[j]-vp[j]; d += dd*dd; }
        if (sqrt(d) < tol) break;
    }
    return {sigma, u, v};
}

// ─────────────────────────────────────────────────────────────────────────────
static void zk_assert_unit(const vector<IntFp>& w, const char* label)
{
    // Accumulate sum of squares
    IntFp acc((uint64_t)0, PUBLIC);
    for (int i = 0; i < F; ++i)
        acc = acc + w[i] * w[i];

    // Subtract SCALE² (public constant)
    uint64_t scale_sq = fp_mul_((uint64_t)(SCALE % (long long)pr),
                                (uint64_t)(SCALE % (long long)pr));
    IntFp diff = acc + IntFp(fp_neg_(scale_sq), PUBLIC);

    // batch_reveal_check opens `diff` and asserts it equals 0
    uint64_t expected_zero = 0;
    batch_reveal_check(&diff, &expected_zero, 1);

    cout << "[ZK] '" << label << "' unit-vector check done.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// ────────────────────── Party Process─────────────────────────────────────────
void prover_work(NetIO* raw_io, vector<uint64_t>& lambda_u, vector<uint64_t>& lambda_v,
                 vector<uint64_t>& du, vector<uint64_t>& dv) {
    // Toy deterministic matrix
    vector<vector<double>> W(F, vector<double>(F));
    for (int i = 0; i < F; ++i)
        for (int j = 0; j < F; ++j)
            W[i][j] = ((i * F + j) % 7) - 3.0;

    SingularPair sv = power_iter(W);
    cout << "[P] calced ||W||2, u, and v "\n";

    vector<uint64_t> u_fp(F), v_fp(F);
    for (int i = 0; i < F; ++i) {
        u_fp[i] = fp_from_double(sv.u[i]);
        v_fp[i] = fp_from_double(sv.v[i]);
    }

    PRG prg;
    prg.random_data(lambda_u.data(), F * sizeof(uint64_t));
    prg.random_data(lambda_v.data(), F * sizeof(uint64_t));
    for (int i = 0; i < F; ++i) {
        lambda_u[i] %= pr;
        lambda_v[i] %= pr;
        du[i] = fp_sub_(u_fp[i], lambda_u[i]);
        dv[i] = fp_sub_(v_fp[i], lambda_v[i]);
    }
    raw_io->send_data(du.data(), F * sizeof(uint64_t));
    raw_io->send_data(dv.data(), F * sizeof(uint64_t));
    raw_io->flush();
    cout << "[P] Sent du, dv.\n";
}


void verifier_work(NetIO* raw_io, vector<uint64_t>& du, vector<uint64_t>& dv) {
    raw_io->recv_data(du.data(), F * sizeof(uint64_t));
    raw_io->recv_data(dv.data(), F * sizeof(uint64_t));
    cout << "[V] Received du, dv.\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────── main ────────────────────────────────────────────
int main(int argc, char** argv) {
    int party, port;
    parse_party_and_port(argv, &party, &port);

    // ── Network setup (BoolIO<NetIO>, matching official pattern) ─────────────
    BoolIO<NetIO>* ios[THREADS];
    for (int i = 0; i < THREADS; ++i)
        ios[i] = new BoolIO<NetIO>(
            new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
            party == ALICE);

    cout << "\n--- Matrix Singular-Vector ZKP ---\n\n";

    // ── ZK arith setup ───────────────────────────────────────────────────────
    setup_zk_arith<BoolIO<NetIO>>(ios, THREADS, party);

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 1 [Prover]: compute dominant singular pair, encode, mask
    // ─────────────────────────────────────────────────────────────────────────
    // We use ios[0]->io to send the public offsets du, dv
    NetIO* raw_io = ios[0]->io;

    vector<uint64_t> lambda_u(F, 0), lambda_v(F, 0);
    vector<uint64_t> du(F, 0),       dv(F, 0);

    if (party == ALICE) {
        prover_work(raw_io, lambda_u, lambda_v, du, dv);
    } else {
        verifier_work(raw_io, du, dv);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 2: commit λ; both build [u] = [λ_u] + du,  [v] = [λ_v] + dv
    // ─────────────────────────────────────────────────────────────────────────
    vector<IntFp> w_u(F), w_v(F);
    for (int i = 0; i < F; ++i) {
        IntFp wl_u((uint64_t)(party == ALICE ? lambda_u[i] : 0ULL), ALICE);
        IntFp wl_v((uint64_t)(party == ALICE ? lambda_v[i] : 0ULL), ALICE);
        w_u[i] = wl_u + IntFp((uint64_t)du[i], PUBLIC);
        w_v[i] = wl_v + IntFp((uint64_t)dv[i], PUBLIC);
    }
    cout << "[" << (party==ALICE?"P":"V") << "] Wires [u],[v] built.\n";

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 3: ZK proof – u and v are unit vectors
    // ─────────────────────────────────────────────────────────────────────────
    zk_assert_unit(w_u, "u");
    zk_assert_unit(w_v, "v");

    // ── Finalize ─────────────────────────────────────────────────────────────
    finalize_zk_arith<BoolIO<NetIO>>();
    cout << "[" << (party==ALICE?"P":"V") << "] Done.\n";

    // ── Cleanup ──────────────────────────────────────────────────────────────
    for (int i = 0; i < THREADS; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
