#pragma once
// src/artifact/dequant.cpp - P1.S3: scalar reference dequantizers, transcribed from ggml.
//
// P1.S3 asks for every encoding present in the Q2_0 file. This starts with the ones on the critical
// path (the routed experts are all Q2_0) plus the trivial element types, and is structured so the
// rest drop in. Source of truth per encoding is quoted in each function.
//
// Q2_0 is the one that matters, and its contract is settled from source in docs/q2_0-contract.md:
//   block = { fp16 d; uint8 qs[16] }  = 64 elements, 18 bytes
//   code {0,1,2,3} -> symbol {-1,0,+1,+2};  v = (code - 1) * d
//   4 codes per byte, LSB-first: byte j/4, bits (j%4)*2
//   QK2_0 = 64 is PROVEN from the artifact's own offset brackets, not assumed (the llama.cpp PR that
//   introduced the type used 128; this artifact is 64).
//
// Validation: `--check <file.gguf>` dequantizes real blocks and asserts the STRUCTURAL invariant that
// every output value is one of {-d, 0, +d, +2d} for that block's own scale. That catches bit-order,
// sign and stride bugs, which are the failure modes a dequantizer actually has. It is deliberately
// not a correctness-vs-ggml test - that needs ggml linked, which is P1.S3's next step.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"

namespace strata {

// ---- fp16/bf16 -> fp32. Both handle subnormals and infinities; naive shifts do not.
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0)
            f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) {
                man <<= 1;
                --exp;
            }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31)
        f = (sign << 31) | 0x7F800000u | (man << 13);
    else
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline float bf16_to_fp32(uint16_t h) {
    const uint32_t f = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

// ---- Q2_0: 64 elements from an 18-byte block. Mirrors dequantize_row_q2_0 (ggml master 3cf03257).
inline void dequantize_q2_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 64; ++j) {
        const int byte_index = j / 4;
        const int bit_offset = (j % 4) * 2;
        const int code = (qs[byte_index] >> bit_offset) & 0x03;
        out[j] = (float)(code - 1) * d; // code {0,1,2,3} -> symbol {-1,0,+1,+2}
    }
}

// ---- Q8_0: 32 elements from a 34-byte block.
inline void dequantize_q8_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const int8_t* qs = (const int8_t*)(block + 2);
    for (int j = 0; j < 32; ++j) out[j] = (float)qs[j] * d;
}

// ---- Q4_0: 32 elements from an 18-byte block. Low nibble is element j, high nibble is j+16.
inline void dequantize_q4_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = (float)((int)(qs[j] & 0x0F) - 8) * d;
        out[j + 16] = (float)((int)(qs[j] >> 4) - 8) * d;
    }
}

// ---- Q5_0: 32 elements from a 22-byte block. Same nibble layout as Q4_0 plus a 5th bit per element
// packed in qh[4]; the 32 bits of qh are the high bits of elements 0..31 in order.
inline void dequantize_q5_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qh = block + 2;
    const uint8_t* qs = block + 6;
    const uint32_t h =
        (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) | ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
    for (int j = 0; j < 16; ++j) {
        const int x0 = (int)(qs[j] & 0x0F) | (int)(((h >> j) & 1u) << 4);
        const int x1 = (int)(qs[j] >> 4) | (int)(((h >> (j + 16)) & 1u) << 4);
        out[j] = (float)(x0 - 16) * d;
        out[j + 16] = (float)(x1 - 16) * d;
    }
}
// ---- Q5_1: 32 elements from a 24-byte block: fp16 d, fp16 m, then Q5_0's qh[4] and qs[16]. The 5-bit codes are
// unsigned and the block's minimum m is added instead of Q5_0's fixed -16 offset (dequantize_row_q5_1).
inline void dequantize_q5_1(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const float m = fp16_to_fp32(read_u16(block + 2));
    const uint8_t* qh = block + 4;
    const uint8_t* qs = block + 8;
    const uint32_t h =
        (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) | ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
    for (int j = 0; j < 16; ++j) {
        const int x0 = (int)(qs[j] & 0x0F) | (int)(((h >> j) & 1u) << 4);
        const int x1 = (int)(qs[j] >> 4) | (int)(((h >> (j + 16)) & 1u) << 4);
        out[j] = (float)x0 * d + m;
        out[j + 16] = (float)x1 * d + m;
    }
}
// ---- IQ4_NL: 32 elements from an 18-byte block. Non-linear 4-bit codebook, NOT a linear grid, so
// the table must be exact. Copied verbatim from ggml-common.h `kvalues_iq4nl` at 3cf03257:
//     -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
// Layout matches dequantize_row_iq4_nl (ggml-quants.c): low nibble is element j, high nibble j+16.
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                         1,    13,   25,  38,  53,  69,  89,  113};

inline void dequantize_iq4_nl(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = d * (float)kvalues_iq4nl[qs[j] & 0x0F];
        out[j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
    }
}
// ---- Q6_K: 256 elements from a 210-byte super-block, transcribed from dequantize_row_q6_K
// (ggml-quants.c at 3cf03257). Layout: ql[128] low nibbles, qh[64] high 2 bits, scales[16] int8,
// fp16 d LAST. Processed in two 128-element halves, each advancing ql by 64, qh by 32, sc by 8.
// The `sc` indices are is+{0,2,4,6} with is = l/16 - so each 32-element group draws from 8 of the 16
// scales, and the four output positions within a 32-group use different scales. Easy to get subtly
// wrong, which is why it is transcribed rather than recalled.
inline void dequantize_q6_K(const uint8_t* block, float* out) {
    const uint8_t* ql = block;                       // 128 bytes
    const uint8_t* qh = block + 128;                 // 64
    const int8_t* sc = (const int8_t*)(block + 192); // 16
    const float d = fp16_to_fp32(read_u16(block + 208));
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}
// ---- Q4_K: 256 elements from a 144-byte super-block. Layout is `fp16 d`, `fp16 dmin` (so the FIRST
// four bytes are the scale pair, unlike Q6_K where d is last), `scales[12]`, `qs[128]`.
// Both functions transcribed from ggml-quants.c at 3cf03257. The 6-bit scale/min unpacking in
// get_scale_min_k4 is the part that is impossible to recall reliably: the low 4 bits of the second
// half come from q[j+4], and the high 2 bits from the low halves' top bits (q[j-4] and q[j]).
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

inline void dequantize_q4_K(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));      // d
    const float mn = fp16_to_fp32(read_u16(block + 2)); // dmin
    const uint8_t* scales = block + 4;                  // 12 bytes
    const uint8_t* q = block + 16;                      // 128 bytes
    float* y = out;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * (float)sc, m1 = mn * (float)m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * (float)sc, m2 = mn * (float)m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}
// ---- Q5_K: 256 elements from a 176-byte super-block: `fp16 d`, `fp16 dmin`, `scales[12]`,
// `qh[32]`, `qs[128]`. Same shape as Q4_K plus the 5th bit plane. Transcribed from
// dequantize_row_q5_K (ggml-quants.c, 3cf03257). The 5th bits are selected by a mask that SHIFTS LEFT
// BY 2 each 64-element group (u1 = 1,4,16,64; u2 = 2,8,32,128) - a recalled version would very likely
// have used a fixed mask or a shift by 1.
inline void dequantize_q5_K(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const float mn = fp16_to_fp32(read_u16(block + 2));
    const uint8_t* scales = block + 4; // 12
    const uint8_t* qh = block + 16;    // 32
    const uint8_t* ql = block + 48;    // 128
    float* y = out;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * (float)sc, m1 = mn * (float)m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * (float)sc, m2 = mn * (float)m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
}
// ---- Q3_K: 256 elements from a 110-byte super-block: `hmask[32]`, `qs[64]`, `scales[12]`, `fp16 d`
// LAST. Transcribed from dequantize_row_q3_K (ggml-quants.c, 3cf03257). This type has NO `dmin` - its
// scales are SIGNED 6-bit with a `- 32` bias - so it does not follow the Q4_K/Q5_K pattern at all.
// Two details that cannot be recalled:
//   * the 12 scale bytes are bit-repacked through three 32-bit masks before use (kmask1/2 below),
//     because 16 six-bit scales do not fit in 12 bytes linearly;
//   * the low 2 bits are offset by -4 when the high bit is CLEAR, i.e. the correction is NEGATIVE.
inline void dequantize_q3_K(const uint8_t* block, float* out) {
    const uint8_t* hm = block;     // 32
    const uint8_t* q = block + 32; // 64
    const float d_all = fp16_to_fp32(read_u16(block + 108));

    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, block + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = (const int8_t*)aux;

    float* y = out;
    int is = 0;
    uint8_t m = 1; // hoisted: ggml declares this BEFORE the n loop, so it does not reset between halves
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (float)((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m = (uint8_t)(m << 1);
        }
        q += 32;
    }
}
// ---- IQ4_XS: 256 elements from a 136-byte super-block: `fp16 d`, `uint16 scales_h`, `scales_l[4]`,
// `qs[128]`. Transcribed from dequantize_row_iq4_xs (ggml-quants.c, 3cf03257). Reuses the IQ4_NL
// codebook. Eight 32-element groups; each group's scale is 6 bits assembled from the LOW nibble of
// `scales_l[ib/2]` (selected by ib%2) and TWO bits of `scales_h` at shift 2*ib - so the high bits come
// from a 16-bit field indexed by group, not from a byte array. Then `- 32`, as in Q3_K.
inline void dequantize_iq4_xs(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint16_t scales_h = read_u16(block + 2);
    const uint8_t* scales_l = block + 4; // 4
    const uint8_t* qs = block + 8;       // 128
    float* y = out;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = dl * (float)kvalues_iq4nl[qs[j] & 0x0F];
            y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}
// ---- IQ3_XXS: 256 elements from a 98-byte block.  GSQ-RCO ships the hyper-connection projections, the router
// and the SSM gates as BF16; OrcaRouter's Uncensored files keep them in their quantized form instead (Q6_K and
// IQ3_XXS), so STRATA_FP16=load has to decode them to reach the same FP16 rows.  Transcribed from
// dequantize_row_iq3_xxs (ggml-quants.c at 3cf03257); the three tables are ggml-common.h's at that commit:
//   block = { fp16 d; uint8 qs[64] grid indices; uint32 scales_and_signs[8] } = 256 elements, 98 bytes
//   each of the 8 words carries four 7-bit indices into ksigns_iq2xs (one per 8 values) plus a 4-bit scale, and
//   one grid entry is FOUR magnitudes: v = d * (0.5 + s) * 0.5 * grid[j] * (sign bit set ? -1 : +1).
static const uint8_t kmask_iq2xs[8] = {
      1,   2,   4,   8,  16,  32,  64, 128
};
static const uint8_t ksigns_iq2xs[128] = {
        0,   129,   130,     3,   132,     5,     6,   135,   136,     9,    10,   139,    12,   141,   142,    15,
      144,    17,    18,   147,    20,   149,   150,    23,    24,   153,   154,    27,   156,    29,    30,   159,
      160,    33,    34,   163,    36,   165,   166,    39,    40,   169,   170,    43,   172,    45,    46,   175,
       48,   177,   178,    51,   180,    53,    54,   183,   184,    57,    58,   187,    60,   189,   190,    63,
      192,    65,    66,   195,    68,   197,   198,    71,    72,   201,   202,    75,   204,    77,    78,   207,
       80,   209,   210,    83,   212,    85,    86,   215,   216,    89,    90,   219,    92,   221,   222,    95,
       96,   225,   226,    99,   228,   101,   102,   231,   232,   105,   106,   235,   108,   237,   238,   111,
      240,   113,   114,   243,   116,   245,   246,   119,   120,   249,   250,   123,   252,   125,   126,   255
};
static const uint32_t iq3xxs_grid[256] = {
      0x04040404,   0x04040414,   0x04040424,   0x04040c0c,   0x04040c1c,   0x04040c3e,   0x04041404,   0x04041414,
      0x04041c0c,   0x04042414,   0x04043e1c,   0x04043e2c,   0x040c040c,   0x040c041c,   0x040c0c04,   0x040c0c14,
      0x040c140c,   0x040c142c,   0x040c1c04,   0x040c1c14,   0x040c240c,   0x040c2c24,   0x040c3e04,   0x04140404,
      0x04140414,   0x04140424,   0x04140c0c,   0x04141404,   0x04141414,   0x04141c0c,   0x04141c1c,   0x04141c3e,
      0x04142c0c,   0x04142c3e,   0x04143e2c,   0x041c040c,   0x041c043e,   0x041c0c04,   0x041c0c14,   0x041c142c,
      0x041c3e04,   0x04240c1c,   0x04241c3e,   0x04242424,   0x04242c3e,   0x04243e1c,   0x04243e2c,   0x042c040c,
      0x042c043e,   0x042c1c14,   0x042c2c14,   0x04341c2c,   0x04343424,   0x043e0c04,   0x043e0c24,   0x043e0c34,
      0x043e241c,   0x043e340c,   0x0c04040c,   0x0c04041c,   0x0c040c04,   0x0c040c14,   0x0c04140c,   0x0c04141c,
      0x0c041c04,   0x0c041c14,   0x0c041c24,   0x0c04243e,   0x0c042c04,   0x0c0c0404,   0x0c0c0414,   0x0c0c0c0c,
      0x0c0c1404,   0x0c0c1414,   0x0c14040c,   0x0c14041c,   0x0c140c04,   0x0c140c14,   0x0c14140c,   0x0c141c04,
      0x0c143e14,   0x0c1c0404,   0x0c1c0414,   0x0c1c1404,   0x0c1c1c0c,   0x0c1c2434,   0x0c1c3434,   0x0c24040c,
      0x0c24042c,   0x0c242c04,   0x0c2c1404,   0x0c2c1424,   0x0c2c2434,   0x0c2c3e0c,   0x0c34042c,   0x0c3e1414,
      0x0c3e2404,   0x14040404,   0x14040414,   0x14040c0c,   0x14040c1c,   0x14041404,   0x14041414,   0x14041434,
      0x14041c0c,   0x14042414,   0x140c040c,   0x140c041c,   0x140c042c,   0x140c0c04,   0x140c0c14,   0x140c140c,
      0x140c1c04,   0x140c341c,   0x140c343e,   0x140c3e04,   0x14140404,   0x14140414,   0x14140c0c,   0x14140c3e,
      0x14141404,   0x14141414,   0x14141c3e,   0x14142404,   0x14142c2c,   0x141c040c,   0x141c0c04,   0x141c0c24,
      0x141c3e04,   0x141c3e24,   0x14241c2c,   0x14242c1c,   0x142c041c,   0x142c143e,   0x142c240c,   0x142c3e24,
      0x143e040c,   0x143e041c,   0x143e0c34,   0x143e242c,   0x1c04040c,   0x1c040c04,   0x1c040c14,   0x1c04140c,
      0x1c04141c,   0x1c042c04,   0x1c04342c,   0x1c043e14,   0x1c0c0404,   0x1c0c0414,   0x1c0c1404,   0x1c0c1c0c,
      0x1c0c2424,   0x1c0c2434,   0x1c14040c,   0x1c14041c,   0x1c140c04,   0x1c14142c,   0x1c142c14,   0x1c143e14,
      0x1c1c0c0c,   0x1c1c1c1c,   0x1c241c04,   0x1c24243e,   0x1c243e14,   0x1c2c0404,   0x1c2c0434,   0x1c2c1414,
      0x1c2c2c2c,   0x1c340c24,   0x1c341c34,   0x1c34341c,   0x1c3e1c1c,   0x1c3e3404,   0x24040424,   0x24040c3e,
      0x24041c2c,   0x24041c3e,   0x24042c1c,   0x24042c3e,   0x240c3e24,   0x24141404,   0x24141c3e,   0x24142404,
      0x24143404,   0x24143434,   0x241c043e,   0x241c242c,   0x24240424,   0x24242c0c,   0x24243424,   0x242c142c,
      0x242c241c,   0x242c3e04,   0x243e042c,   0x243e0c04,   0x243e0c14,   0x243e1c04,   0x2c040c14,   0x2c04240c,
      0x2c043e04,   0x2c0c0404,   0x2c0c0434,   0x2c0c1434,   0x2c0c2c2c,   0x2c140c24,   0x2c141c14,   0x2c143e14,
      0x2c1c0414,   0x2c1c2c1c,   0x2c240c04,   0x2c24141c,   0x2c24143e,   0x2c243e14,   0x2c2c0414,   0x2c2c1c0c,
      0x2c342c04,   0x2c3e1424,   0x2c3e2414,   0x34041424,   0x34042424,   0x34042434,   0x34043424,   0x340c140c,
      0x340c340c,   0x34140c3e,   0x34143424,   0x341c1c04,   0x341c1c34,   0x34242424,   0x342c042c,   0x342c2c14,
      0x34341c1c,   0x343e041c,   0x343e140c,   0x3e04041c,   0x3e04042c,   0x3e04043e,   0x3e040c04,   0x3e041c14,
      0x3e042c14,   0x3e0c1434,   0x3e0c2404,   0x3e140c14,   0x3e14242c,   0x3e142c14,   0x3e1c0404,   0x3e1c0c2c,
      0x3e1c1c1c,   0x3e1c3404,   0x3e24140c,   0x3e24240c,   0x3e2c0404,   0x3e2c0414,   0x3e2c1424,   0x3e341c04
};

inline void dequantize_iq3_xxs(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;             // 64 grid indices
    const uint8_t* ss = block + 66;            // 32: per 32 values, four 7-bit sign indices + one 4-bit scale
    float* y = out;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux32;
        std::memcpy(&aux32, ss + 4 * ib32, 4);
        const float db = d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t* g1 = (const uint8_t*) (iq3xxs_grid + qs[2 * l + 0]);
            const uint8_t* g2 = (const uint8_t*) (iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * (float) g1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                y[j + 4] = db * (float) g2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}

// ---- element types
inline void dequantize_f32(const uint8_t* p, float* out, int n) {
    std::memcpy(out, p, (size_t)n * 4);
}
inline void dequantize_f16(const uint8_t* p, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = fp16_to_fp32(read_u16(p + 2 * i));
}
inline void dequantize_bf16(const uint8_t* p, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = bf16_to_fp32(read_u16(p + 2 * i));
}

} // namespace strata
