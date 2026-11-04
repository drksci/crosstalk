// tb_tdl_layer.v : file-driven testbench for tdl_layer.
//
// Geometry / sizes are compile-time parameters (iverilog -P tb_tdl_layer.<NAME>=<value>); everything
// else is read from files given as plusargs (hex files for $readmemh, one word per line):
//   +x=<file>     NEX*T words of I*IN_W bits (line i at bits [i*IN_W +: IN_W], two's complement)
//   +w=<file>     O*I*K words (2 bits): 0=0, 1=+1, 3=-1,  index ((o*I+i)*K+k)
//   +bq=<file>    O words (32 bit two's complement)
//   +r=<file>     O words (32 bit two's complement, Q.15 per-line output scale)
//   +lut=<file>   256 words (OUT_W bit two's complement), index z+128
//   +M=<int>      LUT-index multiplier (signed decimal, 15 fractional bits)
//   +out=<file>   written: NEX*T lines of O signed decimals (layer output y, int8 range)
//   +pool=<file>  written: NEX lines of O signed decimals (integer sum over time of y)
// Prints "TB_DONE samples=<n> examples=<n>" on success.
`timescale 1ns/1ps
module tb_tdl_layer;
    parameter integer I = 2, O = 8, K = 8, D = 1, IN_W = 8, OUT_W = 8;
    parameter integer NEX = 4, T = 128;

    reg clk = 0, clr = 0, in_valid = 0;
    reg  [I*IN_W-1:0] x_flat = 0;
    reg  [O*I*K*2-1:0] w_flat = 0;
    reg  [O*32-1:0] bq_flat = 0, r_flat = 0;
    reg  signed [31:0] m_scale = 0;
    reg  lut_we = 0;
    reg  [7:0] lut_addr = 0;
    reg  [OUT_W-1:0] lut_wdata = 0;
    wire out_valid;
    wire [O*OUT_W-1:0] y_flat;
    wire [O*32-1:0] pool_flat;

    tdl_layer #(.I(I), .O(O), .K(K), .D(D), .IN_W(IN_W), .OUT_W(OUT_W)) dut (
        .clk(clk), .clr(clr), .in_valid(in_valid), .x_flat(x_flat),
        .w_flat(w_flat), .bq_flat(bq_flat), .m_scale(m_scale), .r_flat(r_flat),
        .lut_we(lut_we), .lut_addr(lut_addr), .lut_wdata(lut_wdata),
        .out_valid(out_valid), .y_flat(y_flat), .pool_flat(pool_flat));

    always #5 clk = ~clk;

    reg [I*IN_W-1:0]  xmem  [0:NEX*T-1];
    reg [1:0]         wmem  [0:O*I*K-1];
    reg [31:0]        bqmem [0:O-1];
    reg [31:0]        rmem  [0:O-1];
    reg [OUT_W-1:0]   lmem  [0:255];

    reg [8*512-1:0] f_x, f_w, f_bq, f_r, f_lut, f_out, f_pool;
    integer fo, fp, e, t, o, n, nsamp, mval;

    // capture outputs on the falling edge (DUT updates on the rising edge)
    always @(negedge clk) begin
        if (out_valid) begin
            for (o = 0; o < O; o = o + 1)
                $fwrite(fo, "%0d ", $signed(y_flat[o*OUT_W +: OUT_W]));
            $fwrite(fo, "\n");
            nsamp = nsamp + 1;
        end
    end

    initial begin
        nsamp = 0;
        if (!$value$plusargs("x=%s", f_x))     begin $display("TB_ERROR missing +x");    $finish; end
        if (!$value$plusargs("w=%s", f_w))     begin $display("TB_ERROR missing +w");    $finish; end
        if (!$value$plusargs("bq=%s", f_bq))   begin $display("TB_ERROR missing +bq");   $finish; end
        if (!$value$plusargs("r=%s", f_r))     begin $display("TB_ERROR missing +r");    $finish; end
        if (!$value$plusargs("lut=%s", f_lut)) begin $display("TB_ERROR missing +lut");  $finish; end
        if (!$value$plusargs("M=%d", mval))    begin $display("TB_ERROR missing +M");    $finish; end
        if (!$value$plusargs("out=%s", f_out)) begin $display("TB_ERROR missing +out");  $finish; end
        if (!$value$plusargs("pool=%s", f_pool)) begin $display("TB_ERROR missing +pool"); $finish; end
        $readmemh(f_x, xmem);
        $readmemh(f_w, wmem);
        $readmemh(f_bq, bqmem);
        $readmemh(f_r, rmem);
        $readmemh(f_lut, lmem);
        fo = $fopen(f_out, "w");
        fp = $fopen(f_pool, "w");
        for (n = 0; n < O*I*K; n = n + 1) w_flat[n*2 +: 2] = wmem[n];
        for (n = 0; n < O; n = n + 1) begin
            bq_flat[n*32 +: 32] = bqmem[n];
            r_flat[n*32 +: 32] = rmem[n];
        end
        m_scale = mval;

        // load the tanh LUT through the configuration port
        @(negedge clk);
        for (n = 0; n < 256; n = n + 1) begin
            lut_we = 1; lut_addr = n; lut_wdata = lmem[n];
            @(negedge clk);
        end
        lut_we = 0;

        for (e = 0; e < NEX; e = e + 1) begin
            clr = 1; in_valid = 0; @(negedge clk); clr = 0;
            for (t = 0; t < T; t = t + 1) begin
                x_flat = xmem[e*T + t]; in_valid = 1;
                @(negedge clk);
            end
            in_valid = 0; x_flat = 0;
            repeat (6) @(negedge clk);           // flush the 3-stage pipeline
            for (o = 0; o < O; o = o + 1)
                $fwrite(fp, "%0d ", $signed(pool_flat[o*32 +: 32]));
            $fwrite(fp, "\n");
        end
        $fclose(fo);
        $fclose(fp);
        $display("TB_DONE samples=%0d examples=%0d", nsamp, NEX);
        $finish;
    end
endmodule
