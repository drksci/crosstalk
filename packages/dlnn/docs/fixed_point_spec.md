# Tier E fixed-point specification (M2a)

Single source: the docstring of `dlnn/tiers/emulated.py` (numpy golden model). `hw/rtl/tdl_layer.v` implements exactly the same
arithmetic and is checked against it bit for bit (`tests/test_emulated_rtl.py`, `emulated.rtl_crosscheck`).

## Per-layer arithmetic

Input to layer 1 are ADC codes, input to layer l>1 are the int8 outputs of layer l-1.

| step | formula | width |
|---|---|---|
| ADC (numpy, not RTL) | `xq = clip(rnd(x*input_scale/s0), -L, L)`, `L=2^(b-1)-1`, `s0=fullscale/L` | b bit signed (b=8 default) |
| delay line | per input line a shift register of depth `(K-1)*d+1`, tap k read at `k*d`; reset = zeros before n=0 | IN_W |
| ternary MAC | `acc = bq[o] + sum_{i,k} W[o,i,k]*xq[i,n-k*d]`, W in {-1,0,+1} coded 2 bit (00, 01, 11) | int32 |
| bias | `bq = rnd(bias / s_in)`, `s_in` = real value of one input LSB | int32 |
| scale | `z = clip((acc*M + 2^14) >>> 15, -128, 127)`, `M = rnd(2^15 * s_in/vsat * 128/4)` | acc*M in int64 |
| tanh LUT | `v = LUT[z+128]`, `LUT[j] = rnd(127*tanh((j-128)*4/128))` (range +-4, step 1/32) | int8 |
| per-line gain | `y = (v*R[o] + 2^14) >>> 15`, `R[o] = rnd(2^15*gain[o]/max|gain|)` signed Q.15 (2^15 = identity) | int8 |
| pool | `P[o] = sum_n y[o,n]` (integer), real feature = `P*s_out/T` | int32 |

`rnd` = round half away from zero. `>>>` = arithmetic shift (floor). `s_out = max|gain|/127`. Pooled-sum -> float, the float readout and
softmax are numpy post-processing in both paths ("ADC + software" in the plan).

## RTL interface (`tdl_layer`)

Parameters `I, O, K, D, IN_W, OUT_W`. One sample per clock on `x_flat` (line i at `[i*IN_W +: IN_W]`), `in_valid`; static buses `w_flat`,
`bq_flat`, `r_flat`, `m_scale`; 256-entry LUT loaded through `lut_we/lut_addr/lut_wdata`; `clr` between examples. Latency 3 cycles
(delay-line register, MAC+bias register, scale/LUT/gain register); `out_valid`, `y_flat`, `pool_flat` out. The testbench
`tb_tdl_layer.v` takes geometry via `iverilog -P` and all data via `$readmemh` files named by plusargs, writes results with `$fwrite`.

## Deviation from the M0 text of the spec

M0 applied the per-line requantisation with `|gain|/max|gain|` (sign dropped) and skipped it when all gains were `allclose`. M2a uses the
signed ratio and always applies it (identity when `R = 2^15`), which is arithmetically identical for positive equal gains and keeps the
sign for negative gains. Trained TDL-Net gains (gamma = mean|W|) are positive, so results do not change for them.
