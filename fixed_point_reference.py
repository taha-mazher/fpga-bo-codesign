"""Fixed-point helpers shared with the RTL (trimmed from the research repo's integer reference model).
Number formats (all signed): x and hidden activations Q4.12 int16; weights scaled by 2^F; biases by 2^(12+F)."""
import numpy as np

XF = 12                      # input / hidden fractional bits
RELU6_MAX = 6 << XF


def frac_bits(w):
    """Largest F such that round(w * 2^F) fits int16."""
    return int(15 - np.ceil(np.log2(np.abs(w).max() + 1e-12)))


def hexs(v, bits):
    return format(int(v) & ((1 << bits) - 1), f'0{bits // 4}x')
