// dequantize_iq3_xxs (include/strata/artifact/dequant.hpp) on synthetic blocks: random scales, grid indices and
// sign/scale words, plus the edge words 0x00000000 and 0xFFFFFFFF.  The oracle below is the SAME encoding written
// the other way round - it walks the 256 output positions and derives, for each one, which grid entry, which byte
// of it and which sign bit it draws from - so it shares no indexing with the block-oriented implementation it
// checks.  Checked on the fp16 bits the engine actually stores, not on fp32 intermediates.  No model, no GPU.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// The same table, read as "which entry, which byte, which sign bit feeds output position e of the 32-value half".
uint16_t oracle(const uint8_t* blk, int ib32, int e) {
    static const uint8_t kSignBits[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    uint32_t aux;
    std::memcpy(&aux, blk + 66 + 4 * ib32, 4);
    const int l = e / 8, j = e % 8;
    const uint8_t entry = blk[2 + 8 * ib32 + 2 * l + (j >= 4 ? 1 : 0)];
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = strata::fp16_to_fp32(dh);
    const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
    const uint32_t grid = strata::iq3xxs_grid[entry];
    const int mag = (int) ((grid >> (8 * (j % 4))) & 0xFFu);
    const bool neg = (strata::ksigns_iq2xs[(aux >> (7 * l)) & 127u] & kSignBits[j]) != 0;
    return strata::kernels::f16_from_f32(neg ? -db * (float) mag : db * (float) mag);
}

}  // namespace

int main() {
    constexpr int kBlocks = 4096;
    std::mt19937 rng(20261010);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<uint32_t> word(0, 0xFFFFFFFFu);
    std::uniform_real_distribution<float> scale(-0.05f, 0.05f);

    std::vector<uint8_t> blocks((size_t) kBlocks * 98);
    for (int b = 0; b < kBlocks; ++b) {
        uint8_t* blk = blocks.data() + (size_t) b * 98;
        const uint16_t d = strata::kernels::f16_from_f32(scale(rng));
        std::memcpy(blk, &d, 2);
        for (int i = 0; i < 64; ++i) blk[2 + i] = (uint8_t) byte(rng);
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = (b == 0) ? 0u : (b == 1 ? 0xFFFFFFFFu : word(rng));
            std::memcpy(blk + 66 + 4 * w, &v, 4);
        }
        if (b == 0) std::memset(blk + 2, 0x00, 64);  // every grid index 0: the entry is 0x04040404
        if (b == 1) std::memset(blk + 2, 0xFF, 64);  // every grid index 255
    }

    int bad = 0, shown = 0;
    std::vector<uint16_t> got(256);
    for (int b = 0; b < kBlocks; ++b) {
        const uint8_t* blk = blocks.data() + (size_t) b * 98;
        float f[256];
        strata::dequantize_iq3_xxs(blk, f);
        for (int i = 0; i < 256; ++i) got[i] = strata::kernels::f16_from_f32(f[i]);
        for (int i = 0; i < 256; ++i) {
            const uint16_t want = oracle(blk, i / 32, i % 32);
            if (got[i] == want) continue;
            ++bad;
            if (shown++ < 6) {
                std::printf("  VIOLATION block %d elem %d: got %04x want %04x\n", b, i, got[i], want);
            }
        }
    }
    std::printf("IQ3_XXS dequant: %d blocks, %d bit mismatches -> %s\n", kBlocks, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
