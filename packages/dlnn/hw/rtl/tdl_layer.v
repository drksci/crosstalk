// tdl_layer.v : one tapped-delay-line layer, digital (Tier E) implementation.
//
//   u[o,n]  = sum_{i,k} W[o,i,k] * x[i, n-k*D]  + bq[o]        (W in {-1,0,+1}, zeros before n=0)
//   z[o,n]  = clip((u[o,n]*M + 2^14) >>> 15, -128, 127)       (64-bit signed intermediate)
//   v[o,n]  = LUT[z[o,n] + 128]                                (256-entry tanh LUT, int8, loaded via lut_* port)
//   y[o,n]  = (v[o,n]*R[o] + 2^14) >>> 15                      (per-line Q.15 gain; R = 2^15 is identity)
//   pool[o] = sum_n y[o,n]                                     (integer time-sum of the streamed example)
//
// Bit-exact twin of dlnn/tiers/emulated.py (spec in its docstring and docs/fixed_point_spec.md).
// One sample per clock per input line.  A shift register of depth (K-1)*D+1 per input line is the
// digital delay line; taps are read at k*D.  Pipeline: input register -> MAC+bias register ->
// activation/requant register, so y for the sample presented in cycle t is valid at cycle t+3
// (out_valid marks it).  `clr` (synchronous) zeroes the delay lines, pipeline and pool; assert it
// for one cycle between examples.
`timescale 1ns/1ps
module tdl_layer #(
    parameter integer I     = 2,     // input lines
    parameter integer O     = 8,     // output lines
    parameter integer K     = 8,     // taps
    parameter integer D     = 1,     // dilation (tap spacing in samples)
    parameter integer IN_W  = 8,     // input code width (signed)
    parameter integer OUT_W = 8      // output code width (signed); LUT entries are OUT_W bits
) (
    input  wire                      clk,
    input  wire                      clr,
    input  wire                      in_valid,
    input  wire [I*IN_W-1:0]         x_flat,    // line i at [i*IN_W +: IN_W], two's complement
    // static configuration
    input  wire [O*I*K*2-1:0]        w_flat,    // tap (o,i,k) at [((o*I+i)*K+k)*2 +: 2]: 00=0, 01=+1, 11=-1
    input  wire [O*32-1:0]           bq_flat,   // signed int32 bias per output line (input LSB units)
    input  wire signed [31:0]        m_scale,   // LUT-index multiplier, 15 fractional bits
    input  wire [O*32-1:0]           r_flat,    // signed Q.15 per-line output scale
    input  wire                      lut_we,
    input  wire [7:0]                lut_addr,
    input  wire [OUT_W-1:0]          lut_wdata,
    // outputs
    output reg                       out_valid,
    output reg  [O*OUT_W-1:0]        y_flat,
    output reg  [O*32-1:0]           pool_flat  // signed running sum of y for this example
);
    localparam integer DEPTH = (K-1)*D + 1;

    reg signed [IN_W-1:0]  sr  [0:I*DEPTH-1];  // delay lines
    reg [OUT_W-1:0]        lut [0:255];
    reg                    v0, v1;
    reg [O*32-1:0]         acc_r;

    integer i, j, k, o;
    reg signed [31:0] acc;
    reg signed [IN_W-1:0] xv;
    reg [1:0] wv;
    reg signed [63:0] p, z;
    reg signed [OUT_W-1:0] lv;
    reg signed [31:0] rr, a32;
    reg signed [63:0] q;
    reg signed [OUT_W-1:0] yq;
    reg signed [31:0] ps;

    always @(posedge clk) begin
        if (lut_we) lut[lut_addr] <= lut_wdata;

        if (clr) begin
            for (i = 0; i < I*DEPTH; i = i + 1) sr[i] <= {IN_W{1'b0}};
            v0 <= 1'b0; v1 <= 1'b0; out_valid <= 1'b0;
            acc_r <= {O*32{1'b0}};
            y_flat <= {O*OUT_W{1'b0}};
            pool_flat <= {O*32{1'b0}};
        end else begin
            // ---- stage 0: delay line shift ------------------------------------------------
            v0 <= in_valid;
            if (in_valid) begin
                for (i = 0; i < I; i = i + 1) begin
                    sr[i*DEPTH] <= x_flat[i*IN_W +: IN_W];
                    for (j = 1; j < DEPTH; j = j + 1) sr[i*DEPTH + j] <= sr[i*DEPTH + j - 1];
                end
            end

            // ---- stage 1: ternary MAC + bias (reads the delay lines as of this cycle) -------
            v1 <= v0;
            for (o = 0; o < O; o = o + 1) begin
                acc = $signed(bq_flat[o*32 +: 32]);
                for (i = 0; i < I; i = i + 1)
                    for (k = 0; k < K; k = k + 1) begin
                        wv = w_flat[((o*I + i)*K + k)*2 +: 2];
                        xv = sr[i*DEPTH + k*D];
                        if (wv == 2'b01)      acc = acc + xv;
                        else if (wv == 2'b11) acc = acc - xv;
                    end
                acc_r[o*32 +: 32] <= acc;
            end

            // ---- stage 2: scale -> clip -> LUT -> per-line requant -> pool ------------------
            out_valid <= v1;
            for (o = 0; o < O; o = o + 1) begin
                a32 = $signed(acc_r[o*32 +: 32]);
                p = a32 * m_scale;                       // 64-bit signed product
                z = (p + 64'sd16384) >>> 15;
                if (z > 64'sd127) z = 64'sd127;
                else if (z < -64'sd128) z = -64'sd128;
                lv = lut[z[7:0] + 8'd128];               // index = z + 128 (mod 256)
                rr = $signed(r_flat[o*32 +: 32]);
                q = lv * rr;                             // 64-bit signed
                yq = (q + 64'sd16384) >>> 15;
                if (v1) begin
                    y_flat[o*OUT_W +: OUT_W] <= yq;
                    ps = $signed(pool_flat[o*32 +: 32]);
                    pool_flat[o*32 +: 32] <= ps + yq;
                end
            end
        end
    end
endmodule
