// ─────────────────────── Helpers ─────────────────────────────────────────────
inline uint64_t mod_mul(uint64_t a, uint64_t b) {
    __uint128_t res = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)(res & pr);
    uint64_t hi = (uint64_t)(res >> 61);
    uint64_t r = lo + hi;
    if (r >= pr) r -= pr;
    return r;
}

inline uint64_t mod_add(uint64_t a, uint64_t b) {
    a += b;
    if (a >= pr) a -= pr;
    return a;
}

uint64_t mod_inv(uint64_t x) {
    uint64_t result = 1, base = x, exp = pr - 2;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base);
        base = mod_mul(base, base);
        exp >>= 1;
    }
    return result;
}

// pr = 2^61 - 1 ≡ 3 mod 4, so sqrt(x) = x^((pr+1)/4) mod pr
uint64_t mod_sqrt(uint64_t x) {
    uint64_t exp = ((uint64_t)pr + 1) / 4;
    uint64_t result = 1, base = x;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base);
        base = mod_mul(base, base);
        exp >>= 1;
    }
    return result;
}

void mat_vec_mul(const uint64_t *M, const double *x, double *out, int n) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += (double)M[i * n + j] * x[j];
        out[i] = s;
    }
}

void mat_T_vec_mul(const uint64_t *M, const double *x, double *out, int n) {
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += (double)M[i * n + j] * x[i];
        out[j] = s;
    }
}

double vec_norm(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    return sqrt(s);
}

void vec_normalize(double *x, int n) {
    double norm = vec_norm(x, n);
    for (int i = 0; i < n; ++i) x[i] /= norm;
}

// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute a, u, v in Z_pr such that W*v = a*u (mod pr)
 * u and v are unit vectors: u^T u = 1, v^T v = 1 (mod pr)
 * a = ||W||_2 (spectral norm), represented as a field element
 *
 * Steps:
 *   1. Power iteration (real-valued) to find dominant singular vectors
 *   2. Scale to integers, map to field elements
 *   3. Normalize in field using modular sqrt + inverse
 */
void spectral_decomp_field(const uint64_t *W,
                           uint64_t &a_out,
                           uint64_t *u_out,   // length F
                           uint64_t *v_out,   // length F
                           int iters = 200)
{
    // ── Step 1: real-valued power iteration ──────────────────────────────────
    double v[F], u[F], tmp[F];
    for (int i = 0; i < F; ++i) v[i] = 1.0 / sqrt((double)F);

    for (int it = 0; it < iters; ++it) {
        // u = normalize(W * v)
        mat_vec_mul(W, v, u, F);
        vec_normalize(u, F);
        // v = normalize(W^T * u)
        mat_T_vec_mul(W, u, tmp, F);
        vec_normalize(tmp, F);
        for (int i = 0; i < F; ++i) v[i] = tmp[i];
    }

    // ── Step 2: scale doubles -> field elements ───────────────────────────────
    // We scale by SCALE so that values become large integers, reducing
    // quantization error. Negatives map to pr - |x|.
    const double SCALE = 1e9;  // modest scale to stay well below 2^61

    uint64_t u_raw[F], v_raw[F];
    uint64_t norm2_u = 0, norm2_v = 0;

    for (int i = 0; i < F; ++i) {
        int64_t ui_int = (int64_t)round(u[i] * SCALE);
        int64_t vi_int = (int64_t)round(v[i] * SCALE);

        u_raw[i] = (ui_int >= 0) ? (uint64_t) ui_int % pr
                                 : pr - (uint64_t)(-ui_int) % pr;
        v_raw[i] = (vi_int >= 0) ? (uint64_t) vi_int % pr
                                 : pr - (uint64_t)(-vi_int) % pr;

        norm2_u = mod_add(norm2_u, mod_mul(u_raw[i], u_raw[i]));
        norm2_v = mod_add(norm2_v, mod_mul(v_raw[i], v_raw[i]));
    }

    // ── Step 3: compute a_raw = (u_raw^T * W * v_raw) / (u_raw^T * u_raw) ───
    uint64_t Wv[F] = {};
    for (int i = 0; i < F; ++i)
        for (int j = 0; j < F; ++j)
            Wv[i] = mod_add(Wv[i], mod_mul(W[i * F + j], v_raw[j]));

    uint64_t uT_Wv = 0;
    for (int i = 0; i < F; ++i)
        uT_Wv = mod_add(uT_Wv, mod_mul(u_raw[i], Wv[i]));

    // a_raw satisfies: W * v_raw = a_raw * u_raw  (approximately, mod pr)
    uint64_t a_raw = mod_mul(uT_Wv, mod_inv(norm2_u));

    // ── Step 4: normalize u, v to unit vectors in the field ──────────────────
    // |u_raw| = sqrt(norm2_u), so u_unit = u_raw * inv(sqrt(norm2_u))
    // After normalization: W * v_unit = a_out * u_unit
    // => a_out = a_raw * sqrt(norm2_u) * inv(sqrt(norm2_v))
    //          = a_raw * sqrt(norm2_u) / sqrt(norm2_v)
    uint64_t sqrt_norm2_u = mod_sqrt(norm2_u);
    uint64_t sqrt_norm2_v = mod_sqrt(norm2_v);

    uint64_t inv_sqrt_u = mod_inv(sqrt_norm2_u);
    uint64_t inv_sqrt_v = mod_inv(sqrt_norm2_v);

    // a_out: W*(v_raw/|v|) = (a_raw * |u|/|v|) * (u_raw/|u|)
    //                      = a_raw * sqrt_norm2_u * inv_sqrt_v * u_unit
    a_out = mod_mul(mod_mul(a_raw, sqrt_norm2_u), inv_sqrt_v);

    for (int i = 0; i < F; ++i) u_out[i] = mod_mul(u_raw[i], inv_sqrt_u);
    for (int i = 0; i < F; ++i) v_out[i] = mod_mul(v_raw[i], inv_sqrt_v);
}
