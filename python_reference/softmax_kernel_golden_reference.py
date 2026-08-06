"""
Golden reference implementation
softmax_kernel.py
Vectorized BF16/FP32 Online Softmax Kernel :  Base-2 Shift-LUT Exponential.

Mirrors RTL pipeline:
    bf16_compare.sv  -> running max (m, BF16) + new_max_flag
    bf16_delta.sv    -> |m_old - m_new| magnitude
    shift_lut_exp.sv -> 32-entry Q1.11 ROM + barrel shift, base-2 exp approx
    fp32_mac.sv / accumulator_update.sv -> l_new = l_old*r + n  (FP32)

r/n routing: only ONE side of the recurrence is rescaled per step, so a
single exp unit suffices (see softmax_engine_top.sv S2 stage).

"""

import torch
import warnings


# CONSTANTS (frozen to match shift_lut_exp.sv)


LOG2E = 1.4426950408889634
ROM_BITS = 5                     # 32-entry ROM (matches RTL)
ROM_ENTRIES = 1 << ROM_BITS      # 32
Q11_SCALE = 1 << 11              # Q1.11 ROM output format
Q45_SCALE = 1 << 5               # Q4.5 format for z = delta*log2(e) (5 frac bits)
SAT_THRESH = 480                 # z_q45 >= 480 (== 15.0) -> saturate to 0
BARREL_WIDTH = 16                # shifts >= this collapse to 0 (matches RTL shifter width)



# FIXED-POINT / SHIFT-LUT PIPELINE STAGES (Vectorized & Unit-Testable)


def build_shift_lut_rom():
    """
    32-entry ROM, Q1.11 fixed point.
    ROM[j] = round(2^(-j/32) * 2048), j = 0..31
    """
    rom = torch.zeros(ROM_ENTRIES, dtype=torch.int64)
    for j in range(ROM_ENTRIES):
        val = 2.0 ** (-j / ROM_ENTRIES)
        rom[j] = int(round(val * Q11_SCALE))
    return rom


def shift_lut_exp_vectorized(delta_f32, rom_tensor):
    """
    Vectorized model of shift_lut_exp.sv: approximates 2^(-delta) for
    delta >= 0 using a log2(e) multiply, 32-entry ROM, and barrel shift.

    delta_f32: non-negative FP32 tensor
    Returns:
        exp_approx: FP32 tensor, approx of e^(-delta), in [0, 1]
        sat_mask:   bool tensor, True where delta saturated (output forced 0)
    """
    device = delta_f32.device
    rom_tensor = rom_tensor.to(device)

    delta_f32 = torch.nan_to_num(delta_f32, nan=1e6, posinf=1e6, neginf=1e6)
    delta_f32 = torch.clamp(delta_f32, max=1e6)

    z = delta_f32 * LOG2E
    z_q45 = torch.round(z * Q45_SCALE).to(torch.int64)

    sat_mask = z_q45 >= SAT_THRESH
    z_q45_safe = torch.clamp(z_q45, min=0, max=SAT_THRESH - 1)

    z_int = z_q45_safe >> ROM_BITS               # barrel shift amount
    z_frac = z_q45_safe & (ROM_ENTRIES - 1)       # ROM index

    rom_val = rom_tensor[z_frac]

    valid_shift = z_int < BARREL_WIDTH
    safe_shift = torch.clamp(z_int, max=BARREL_WIDTH)
    shifted = rom_val >> safe_shift
    exp_q11 = torch.where(valid_shift, shifted, torch.zeros_like(shifted))
    exp_q11 = torch.where(sat_mask, torch.zeros_like(exp_q11), exp_q11)

    exp_approx = exp_q11.to(torch.float32) / Q11_SCALE
    return exp_approx, sat_mask


def characterize_exp_error(rom_table, x_max=15.0, num_points=2000):
    """
    Compare shift_lut_exp against true e^(-x) for x in [0, x_max).
    Returns x_vals, approx, true_vals, rel_err (%), sat_mask -- all tensors.
    """
    x_vals = torch.linspace(0.0, x_max, num_points)
    approx, sat = shift_lut_exp_vectorized(x_vals, rom_table)
    true_vals = torch.exp(-x_vals)
    rel_err = torch.where(
        true_vals > 1e-12,
        torch.abs(approx - true_vals) / true_vals * 100.0,
        torch.zeros_like(true_vals),
    )
    return x_vals, approx, true_vals, rel_err, sat



# TOP-LEVEL LAYER


class BF16SoftmaxLayer:
    """
    Streaming online softmax matching softmax_engine_top.sv:
      Pass 1: m (BF16 running max), l (FP32 running denominator), r/n routing
              -> single shift_lut_exp unit per step, no pipeline stall.
      Pass 2: p_i = shift_lut_exp(|x_i - m_final|) / l   (vectorized over N)

    """

    def __init__(self, rom_bits=ROM_BITS):
        if rom_bits != ROM_BITS:
            raise ValueError(
                f"RTL ROM is frozen at {ROM_BITS} address bits ({ROM_ENTRIES} entries); "
                f"got rom_bits={rom_bits}"
            )
        self.rom = build_shift_lut_rom()

    @staticmethod
    def _to_bf16(x_fp32):
        return x_fp32.to(torch.bfloat16)

    def fp_online_softmax(self, x_float_seq):
    
        device = x_float_seq.device
        rom = self.rom.to(device)

        x_fp32 = x_float_seq.to(torch.float32)
        seq_len = x_fp32.shape[-1]

        # Pass 1: streaming m (BF16) / l (FP32) recurrence 
        m_curr = self._to_bf16(x_fp32[..., 0])
        l_curr = torch.ones_like(x_fp32[..., 0])  # l_0 = 1.0 (FP32)

        total_sat = 0

        for n in range(1, seq_len):
            x_n_bf16 = self._to_bf16(x_fp32[..., n])

            # bf16_compare: new_max_flag = (x_n > m_curr)
            new_max_flag = x_n_bf16.to(torch.float32) > m_curr.to(torch.float32)
            m_new = torch.where(new_max_flag, x_n_bf16, m_curr)

            # bf16_delta: magnitude between old max and incoming score
            # (== |m_old - m_new| exactly one of the two terms below is 0)
            delta = torch.abs(m_curr.to(torch.float32) - x_n_bf16.to(torch.float32))
            exp_delta, sat = shift_lut_exp_vectorized(delta, rom)
            total_sat += int(sat.sum().item())

            # r/n routing (single exp unit, no stall):
            #   x_n is new max -> old l needs rescaling by exp_delta, new term weight = 1.0
            #   m_curr stays max -> old l keeps r=1.0, new term rescaled by exp_delta
            r = torch.where(new_max_flag, exp_delta, torch.ones_like(exp_delta))
            n_term = torch.where(new_max_flag, torch.ones_like(exp_delta), exp_delta)

            # accumulator_update.sv: l_new = l_old * r + n
            l_curr = l_curr * r + n_term
            m_curr = m_new

        # Pass 2: vectorized numerator over the full sequence 
        m_final_fp32 = m_curr.to(torch.float32)
        delta_final = torch.abs(x_fp32 - m_final_fp32.unsqueeze(-1))
        num, sat_final = shift_lut_exp_vectorized(delta_final, rom)
        total_sat += int(sat_final.sum().item())

        l_safe = torch.clamp(l_curr, min=1e-12)
        probs = num / l_safe.unsqueeze(-1)

        return probs, total_sat



# UNIT TESTS


def run_tests():
    print("--- Running fp_softmax_kernel Unit Tests [BF16/FP32, 32-entry ROM] ---\n")

    rom = build_shift_lut_rom()
    assert rom[0] == Q11_SCALE, f"ROM[0] should be 1.0 in Q1.11, got {rom[0]}"
    # 2^(-16/32) = 2^-0.5 ~= 0.70710678 * 2048 ~= 1448
    assert rom[16] in (1447, 1448), f"ROM[16] expected ~1448, got {rom[16]}"
    print("[PASS] build_shift_lut_rom")

    # shift_lut_exp_vectorized: delta=0 -> exp=1.0 exactly
    d0 = torch.tensor([0.0])
    e0, s0 = shift_lut_exp_vectorized(d0, rom)
    assert abs(e0.item() - 1.0) < 1e-6, f"exp(0) should be 1.0, got {e0.item()}"
    assert not s0.item()
    print("[PASS] shift_lut_exp_vectorized(0) == 1.0")

    # Saturation: large delta -> 0
    d_big = torch.tensor([20.0])
    e_big, s_big = shift_lut_exp_vectorized(d_big, rom)
    assert e_big.item() == 0.0 and s_big.item(), "large delta should saturate to 0"
    print("[PASS] shift_lut_exp_vectorized saturation")

    # Approximation quality: delta=1.0 -> true e^-1 ~= 0.3679
    d1 = torch.tensor([1.0])
    e1, _ = shift_lut_exp_vectorized(d1, rom)
    true1 = torch.exp(torch.tensor(-1.0))
    rel_err = abs(e1.item() - true1.item()) / true1.item() * 100
    assert rel_err < 5.0, f"relative error too high at delta=1.0: {rel_err:.2f}%"
    print(f"[PASS] shift_lut_exp_vectorized(1.0) rel_err={rel_err:.2f}%")

    # Full softmax recurrence vs PyTorch reference
    x_seq = torch.tensor([[1.0, 2.0, 3.0]])
    layer = BF16SoftmaxLayer()
    p_bf16, sat_events = layer.fp_online_softmax(x_seq)
    p_torch = torch.softmax(x_seq, dim=-1)
    max_diff = torch.max(torch.abs(p_bf16 - p_torch)).item()
    assert max_diff < 5e-2, f"Softmax divergence too high: {max_diff}"
    print(f"[PASS] fp_online_softmax (Max Diff vs Torch: {max_diff:.4f}, sat_events={sat_events})")

    x_masked = torch.tensor([[-3.0, -5.0, -2.0, -1e4, -1e4]])
    p_masked, sat_masked = layer.fp_online_softmax(x_masked)
    assert p_masked[0, 3].item() == 0.0 and p_masked[0, 4].item() == 0.0, \
        "masked positions should be exactly 0 without MASK_FILL"
    assert p_masked[0].argmax().item() == 2, "true max (index 2, -2.0) should dominate"
    print(f"[PASS] natural masking without MASK_FILL (sat_events={sat_masked})")

    finfo_min = torch.finfo(torch.float32).min
    x_hf_masked = torch.tensor([[-3.0, -5.0, -2.0, finfo_min, finfo_min]])
    p_hf, sat_hf = layer.fp_online_softmax(x_hf_masked)
    assert torch.isfinite(p_hf).all(), "probs must stay finite with HF-style mask fill"
    assert p_hf[0, 3].item() == 0.0 and p_hf[0, 4].item() == 0.0, \
        "HF-style masked positions should be exactly 0"
    assert abs(p_hf.sum().item() - 1.0) < 1e-3, f"probs should sum to 1, got {p_hf.sum().item()}"
    assert p_hf[0].argmax().item() == 2, "true max (index 2, -2.0) should still dominate"
    print(f"[PASS] HF-style finfo.min mask fill stays finite (sat_events={sat_hf})")

    print("\nALL UNIT TESTS PASSED.")


if __name__ == "__main__":
    run_tests()
