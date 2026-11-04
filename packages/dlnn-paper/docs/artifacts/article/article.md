# Taps are weights: a delay-line neural network you can solder

## Sound in a tube of mercury

Status words used throughout: **simulated** (a model ran), **emulated** (the Verilog RTL ran), **designed, not fabricated** (the board files exist and pass DRC, nothing was built). Nothing in this article is measured.

In 1945 von Neumann's EDVAC report named delay lines as the machine's memory [vonneumann1945]. A pulse goes into one end of a tube, travels to the other end, and is caught and re-injected, so a bit is a sound wave in transit. The engineering was done at the Moore School; the 1949 IRE paper is about mercury lines at "a pulse rate of several megacycles" [eckert1949]. EDSAC ran on 32 of these tanks [cam_edsac] and the designers wrote it up themselves [wilkes1950]. Sound travels in mercury at roughly 1,450 m/s (secondary source, [wp_delayline]), and each of EDSAC's lines held 560 bits in flight (same source).

Now tap the line along its length. At each tap you can read the signal as it was some fixed time ago. Multiply the taps by weights, add them up, and you have a transversal filter, a 1940 idea [kallmann1940]. Make the weights learnable, push the sum through a squashing function, and you have one layer of a convolutional neural network, which is also what a time-delay neural network is [waibel1989]. A delay line with taps is a 1-D convolution with the multiplications done by wiring.

So here is the question this project asks: what if the delay line *was* the neural net? Keep the weights to a polarity. A tap wired in is +1, a tap wired in through an inverter is -1, a tap left unconnected is 0. That is a ternary network in the BitNet b1.58 style [ma2024], and its parameters are not stored anywhere. They are *which resistors got soldered to the board*. The pick-and-place file that tells the assembly house which 0402 resistor goes where **is the model**. On our board design the CPL for the three tiles lists 85, 315 and 316 populated crossbar resistors, and a script checks that these are exactly the non-zero weights of the trained network.

This is a research article and a prototype report, not a product. Everything in it is simulation of a design that has not been built. We took a small ternary network all the way from training, through four simulation fidelities, to routed copper (for part of it), and we tell you where each piece stands and where it broke. Each number you see in a grey chip links to a field in a JSON file; the claim list is `docs/artifacts/data/claims.json` and the script `scripts/check_article_numbers.py` re-reads the sources and fails if the page and the data disagree.

One framing note up front. Our task is 8-class modulation recognition (OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK) on a synthetic generator with 16,000 training and 4,000 test windows of 128 complex samples at SNRs from 0 to 20 dB. Accuracy on this set is **not comparable** with RadioML numbers in the literature. We use it to compare our own models with each other.

## Race it: what the live arena shows

The widget above runs the small delay-line net in your browser next to conventional competitors on the same stream. Read the table before you play, because the headline is not flattering to the delay-line net.

All models are trained on the same 16,000 windows and scored on the same 4,000. "Mixed" is accuracy over SNR 0-20 dB; "10 dB+" restricts to SNR of 10 dB or more. All are **simulated** (numpy or torch on a CPU).

| model | params | mixed | 10 dB+ |
|---|---|---|---|
| DLNN w8, ternary taps and readout | {{arena.dlnn.params | 1,275}} | {{arena.dlnn.acc | 0.806}} | {{arena.dlnn.ge10 | 0.838}} |
| DLNN w8, ternary taps, float digital readout | 1,275 | {{arena.dlnn_fro.acc | 0.851}} | {{arena.dlnn_fro.ge10 | 0.888}} |
| 1-D CNN, same topology, float | {{arena.cnn.params | 1,248}} | {{arena.cnn.acc | 0.909}} | {{arena.cnn.ge10 | 0.955}} |
| cumulant and statistics features + logistic regression | 208 | {{arena.cumulant.acc | 0.875}} | {{arena.cumulant.ge10 | 0.958}} |
| random ternary delay reservoir + ridge readout | {{arena.reservoir.params | 1,544}} trained | {{arena.reservoir.acc | 0.718}} | n/a |
| MLP on raw I/Q, about 1k params | {{arena.mlp_s.params | 1,068}} | {{arena.mlp_s.acc | 0.261}} | n/a |
| MLP on raw I/Q, about 10k params | {{arena.mlp_l.params | 10,078}} | {{arena.mlp_l.acc | 0.542}} | n/a |
| MLP on phase-invariant features incl. z^2 and z^4 line spectra, about 1k params | {{arena.mlp_pow_s.params | 992}} | {{arena.mlp_pow_s.acc | 0.783}} | n/a |
| same, about 9k params | 9,192 | {{arena.mlp_pow_l.acc | 0.845}} | {{arena.mlp_pow_l.ge10 | 0.933}} |

Three things to take from it.

**The float CNN and the cumulant classifier beat the delay-line net on accuracy.** The CNN has the same topology as the DLNN (three causal convolutions, 8 taps, width 8) with real-valued weights and a ReLU, and it scores 0.909 against 0.806. The classical cumulant classifier, 208 parameters and no gradient descent, scores 0.875. The case for the delay-line net is **cost and physicality**, not accuracy at this size. Giving the readout floats (the readout runs on the microcontroller anyway) recovers part of the gap, 0.851.

**A fair MLP is a feature MLP.** Our first pair of MLPs was a strawman and we say so in the "what we got wrong" section. A raw-I/Q MLP does not learn this task at either size: 0.261 at 1k parameters and 0.542 at 10k, and no amount of normalisation, dropout, weight decay or schedule search moved them (the 1k net went from 0.285 to 0.26, which is noise). The input representation is the problem: the data has a uniformly random carrier phase and a dense MLP has no way to be invariant to it. Give the MLP phase-invariant features and it recovers: about 0.78 at matched size (992 parameters), still below the DLNN's 0.806, and 0.845 at about nine times the parameters, above it. That feature extraction is not free: 9,274 multiplies for the small one against 3,144 for the whole DLNN.

**Energy, as an estimate.** Per classification, in the arena's operation-count convention, the DLNN needs 93,376 adds, 3,144 multiplies and 3,072 tanh evaluations; the CNN needs 151,616 adds, 147,528 multiplies and the same number of nonlinearities. Using Horowitz's 45 nm, 0.9 V figures (a 32-bit float add costs 0.9 pJ and a multiply 3.7 pJ [Horowitz, ISSCC 2014, Fig. 1.1.9, read from the primary PDF]) and charging each tanh as ten float multiplies, the digital-equivalent *arithmetic* energy is about 209 nJ for the DLNN and 796 nJ for the CNN. These are estimates for a hypothetical 45 nm datapath. They ignore memory traffic, instruction overhead and the fact that nobody builds this on a 45 nm ASIC.

The honest comparison for the board is the electrical one. Our design currents add up to 216 mA at 5 V for the three tiles plus carrier, about 1.08 W, and a classification occupies roughly 150 microseconds (a 128-sample window at 1 MS/s plus pipeline fill). That is about 162 uJ per classification, roughly 774 times the arithmetic estimate above. For scale, an STM32G474 running the CNN, charged at the datasheet's 173.5 uA/MHz at 170 MHz (DS12288 Rev 2, table 21, typical, 25 C), with an optimistic one cycle per multiply-accumulate and an assumed 3.3 V supply, comes to at least 84 uJ. So at this scale and with this op-amp-heavy tile **the analog board does not win on energy**. All of these are **estimates**, built from a design current, one datasheet table and assumptions we state; none is measured.

## Where this comes from, and who got there first

We are not the first to put a neural network on a delay line, and we will not claim to be. The honest framing is: this is a reproducible, four-fidelity, open-hardware build with an explicit prior-art comparison.

The delay-line strand runs from mercury memory [eckert1949] through transversal filters [kallmann1940], bucket-brigade and charge-coupled delay lines [sangster1969][boyle1970], and surface-acoustic-wave devices [tancrell1971]. ADALINE put adaptive weights on a tapped delay line [widrow1960]. Waibel's time-delay neural networks made the delayed-input convolution a standard network layer [waibel1989]. WaveNet-style dilated causal convolutions are the same structure with spacing between taps [wavenet2016][bai2018]. Delay-based reservoir computing uses one nonlinear node and a delay loop [appeltant2011], and we use a random ternary delay reservoir as one of the arena competitors. Ternary weights come from the quantisation literature [li2016] and, recently, BitNet b1.58 [ma2024].

The **closest prior work** is on the RF side, and a reader should look at it first:

- **Pour and Ha (ISCAS 2025)** [pour2025] build an RF-domain recurrent network for anomaly detection in wireless systems, and the Virginia Tech thesis by **Merton (2026)** [merton2026] describes the original design as using "a passive tapped transmission line delay for sequential input samples" and proposes an active gm-C all-pass delay line to replace it. We read the thesis abstract page and found the ISCAS paper by DOI; we did not read either full text, so we cannot say how close their circuit is to ours (for example whether their taps are signed or ternary, or what their nonlinearity is). Their network is recurrent and their task is anomaly detection. Ours is feed-forward and the task is modulation recognition.
- **Woo, Jung and Mukhopadhyay (2024)** [woo2024] already published ternary-weight modulation recognition in digital hardware (an FPGA design).
- **Tank and Hopfield (1987)** [tank1987] built an analog network whose delays "concentrate information in time", and Unnikrishnan, Hopfield and Tank applied time-delayed connections to digit recognition [unnikrishnan1991]. We did not read these in full either.

What this project adds is narrower: a feed-forward ternary convolutional stack where the taps of an LC delay ladder carry the weights and the pick-and-place file is the model; the same trained weights evaluated at four fidelities with the gaps reported; an open-hardware board design with its ERC and DRC results; and a candid account of what failed. It is an integration and a stress test, not a new principle.

## Taps are weights, and polarity is the ternary

Picture one layer. Each input channel runs down a delay line. At eight equally spaced points along each line there is a tap, buffered so it does not load the line. Each tap can connect to each output's summing node with one of three states:

- **+1**: a 10 kohm 0402 resistor to the output's "+" summing node;
- **-1**: a 10 kohm resistor to the "-" summing node;
- **0**: neither footprint is populated (DNP, "do not populate").

Each output has two inverting summers, one per polarity, and a difference amplifier subtracts them. That difference drives a bipolar differential pair. A differential pair with a current-source tail has a transfer function that is a hyperbolic tangent, and its width is set by physics: the diff-pair swing is tanh of the input over 2 V_T, which is 51.7 mV at room temperature. We get the neuron's soft limiter from a matched pair of NPN transistors instead of a lookup table. A track-and-hold re-times the result at the symbol instant and drives the next tile.

Between the stages the signal is a sampled stream, not a smooth waveform. A zero-order hold in the previous tile holds each symbol, the LC ladder delays it by about 0.962 microseconds per tap interval (calibrated in firmware to the real line, so the sample rate is about 1 MS/s), and the next layer's taps read it out at those intervals. At 8 samples per symbol this is a 125 kBd modulation. One PCB design, a "tile", implements every layer; layer 1 has only 2 of its 8 lines populated because the input is I and Q.

The pooling and the final readout are digital. The last tile's eight outputs go to the microcontroller's ADCs, which average over the window and apply an 8-class readout and softmax. So this is a minimal hybrid: the convolutions, the nonlinearity and the delays are analog; pooling, the readout and the classifier decision are not.

Why not make everything passive? A resistor network cannot give gain, and it cannot give a nonlinearity, so a stack of purely passive layers collapses to a single linear filter. The summers and the diff pair provide the gain and the tanh. That is the entire role of active parts in the design, and it is why the op-amp count dominates the power budget later.

The widget above shows the real trained weights of the width-8 net: pick a layer, hover a tap, read its polarity.

## The maths, lightly

A layer with input channels i, output channels o, K taps and tap spacing d computes

```
u[o,n] = sum over i,k of  W[o,i,k] * x[i, n - k*d]          W in {-1, 0, +1}
y[o,n] = gain * tanh( (u[o,n] + b[o]) / Vsat )
```

The receptive field of a stack is 1 plus the sum over layers of d(K-1). With K = 8 and d = 1 in all three layers, that is 1 + 3 x 7 = 22 samples. Phase 1 (below) used dilations 1, 4 and 16, which gives a much longer reach with the same weights; the board cannot do that, because a tile has one fixed tap spacing, so the board network has d = 1 everywhere, "TDL-Net-B".

A coaxial cable delays by length over phase velocity. A lumped LC ladder delays by section count times the square root of LC:

```
tau = N * sqrt(L*C)        Z0 = sqrt(L/C)        f_cutoff = 1 / (pi * sqrt(L*C))
```

Our ladder uses 56 sections of 120 uH and 120 pF, so Z0 is 1000 ohm, each section delays by 120 ns, and the ideal cut-off is 2.65 MHz. Taps sit every 8 sections, giving 8 taps per line. A constant-k ladder is a low-pass filter whose group delay climbs near cut-off, which is why we run the symbol stream far below it and why a real inductor's parasitics matter (later).

Gain matching is where the circuit meets the model. The summer's feedback resistor must satisfy

```
R_f = R_in * 2*V_T / (A_in * Vsat_model)
```

so that the physical tanh argument equals the model's. R_in is the 10 kohm crossbar resistor, A_in the activation amplitude in volts per model unit, and the layer's Vsat is a trained parameter. This is one of the places where the design later went wrong (see "where it breaks").

Training uses the standard ternary recipe. The forward pass uses the weights rounded to {-1, 0, +1} times a learned scale, and the backward pass uses a straight-through estimator that pretends the rounding was the identity [bengio2013]. We follow BitNet b1.58 [ma2024] in using absmean ternarisation, with a soft ramp early in training. The model has 1,152 ternary weights at width 8, 2 -> 8 -> 8 -> 8 channels, plus a handful of float scales and biases.

## Phase 1: the coax study, and why it is not buildable

We started with the obvious physical delay line: coaxial cable. Phase 1 trained a ternary network with dilations 1, 4 and 16 (one symbol per 10 ns, tap delays set by cable length), then evaluated the same trained weights at four fidelities: V (ideal numpy), E (bit-true fixed point checked against Verilog RTL), C (a SPICE-calibrated circuit model) and R (an RF channel model of the cable). Everything here is **simulated** or **emulated (RTL)**.

At width 32 the ideal tier scores 0.863, the 8-bit fixed-point tier 0.863, the circuit tier 0.663, and the RF tier 0.701. The float reference with the same topology and free taps scores 0.919. The RTL is bit-identical to the numpy golden model: 202,752 chained layer values compared, 0 mismatches. That E-tier check is the part of phase 1 that we are happiest with.

The C and R tiers apply a fitted RG-58 loss of 32 dB per microsecond with no repeaters, and the longest tap is 1.12 microseconds of cable. The network degrades. The plan itself said "not buildable" for that point, and we agree, for three reasons:

- *Loss.* The no-repeater point is the headline for C and R and it is not buildable. Part of C's gap is a modelling choice, not just line loss: with no loss compensation at all the same C tier scores 0.804 instead of 0.663.
- *Extrapolation.* The SPICE run covered a layer-1 unit cell; layers 2 and 3 used a closed-form line model fitted to layer 1 and never simulated in SPICE, and ngspice's lossy line model has no skin effect.
- *Length.* Metres of cable at 10 ns per symbol, up to over a microsecond of cable per tap, with eight lines per layer, is a coil of RG-58 and not a board.

The coax study stays in the repository as an appendix because its E-tier check and its tier-by-tier machinery carry over. The tiers were re-pointed at a board.

### The LC-ladder tile

The replacement is a lumped LC ladder at about 1 MBd, built from standard SMD parts, with the ternary crossbar as the weights. The question is whether an inductor-based ladder can hold enough delay at that rate without eating the signal.

Here is the loss insight that shaped the part choice. In a cascade of lossy sections, each radian of electrical phase costs about 1/(2Q) nepers, whatever the sections are made of. To delay a signal by the 7 tap intervals between tap 0 and tap 7, at the Nyquist frequency of the symbol stream (pi radians per sample), the line must carry 7 pi radians of phase. So the Q-limited loss is about

```
loss(tap 7, Nyquist)  ~  7*pi / (2*Q)  nepers
```

For Q = 10 that is 1.10 Np, or 9.6 dB; for Q = 30 it is 3.2 dB. The ngspice ladder with the real inductor model (DCR, self-resonance, a parallel loss resistance set from Q) gives 6.7 dB and 2.8 dB at 500 kHz. The back-of-envelope number is an upper-ish estimate at Q = 10, because the inductor's loss is split between series DCR and a parallel resistance and its effective Q varies with frequency, and it is close at Q = 30. It is our own scaling argument, not a figure from a source, and we use it for what it is good for: the **delay-bandwidth product costs Q**. To stay under 3 dB at Nyquist you need Q of about 32. A shorter delay per tap, or a lower symbol rate, buys the loss back linearly.

That fixed the operating point. The ladder is a low-pass: its tap-7 corner is 280 kHz at the datasheet minimum Q and 373 kHz at an assumed typical Q (the datasheet gives only a minimum). We run the symbol clock at 125 kBd, 8 samples per symbol. At 250 kHz, twice the symbol rate, the simulated loss from tap 0 to tap 7 is 2.26 dB at Q-min and 1.29 dB at Q = 30, inside our 3 dB budget; at 1 MHz it is 28.9 dB, which is why the 1 MS/s samples are heavily smoothed and the network has to be trained on the real response.

**The inductor-Q trade.** No JLCPCB part meets the plan's shielded, 120 uH, SRF-over-5-MHz, DCR-under-8-ohm wish list. The part we chose is a 1812 wirewound chip (FHW1812IF121JST, C394134) with DCR 3.3 ohm, SRF 3.3 MHz and a listed minimum Q of 10 at 796 kHz. It is **unshielded**. The smaller 1210 variant would fit a tighter row pitch but has DCR 7.9 ohm and a simulated loss of 3.5 dB at 250 kHz at Q-min, over budget. So we paid area for Q: a 1812 is 4.5 x 3.2 mm, and that sets the board size. The datasheet gives only a minimum Q, so the typical Q of 30 is an assumption. Buying 20 of the chips and measuring Q(f) is the first action on anyone's list.

## Training a net for the board

**Why d = 1.** With dilation gone, the width-8 net has the same 1,152 ternary weights but a dense 22-sample receptive field. We expected it to be worse than phase 1's dilated net. It is better: 0.806 mixed accuracy (0.838 at 10 dB or more) against 0.735 for the phase-1 dilated width-8 ideal tier. On this synthetic set the dense fine-grained window suits the signals better than the 1/4/16 spacing. We do not claim this transfers to any other data. Width 16 reaches 0.899; the virtual width-32, 4-layer net reaches 0.931 against 0.929 for its float baseline, so there the ternary constraint costs nothing measurable on this set. Width 8 is the network the board implements.

**Board-aware training.** The tile's delay line is not ideal: it has intersymbol interference, dispersion, a frequency-dependent tap spacing and loss. We simulated a unit pulse through the real ladder in ngspice (the M1a design values, a 482 kHz input RC), sampled each tap on the symbol clock, and turned that into a small FIR per tap. During training every layer's forward pass uses a random mix of the Q = 10 and Q = 30 FIRs and an independent random delay error of up to 14 percent of a symbol on each tap.

The plain d = 1 net, run through that real line, drops from 0.806 to 0.772. The board-aware net with the float digital readout scores **0.878 on the board line**, and it is flat across the two Q corners (0.878 at Q = 30) and across the delay error (0.877 against 0.772 for the plain net). The price: on an *ideal* line the same net scores only 0.655. It is a board-only network, trained for the cable it will actually sit on, and it makes no promise elsewhere. Part of that jump is the float readout, not the line awareness: the float readout alone lifts the plain net on an ideal line from 0.806 to 0.851. The like-for-like comparison, ternary readout on the board line, is the focal-loss net below. All of this is **simulated**, the FIR covers the LC line and the input RC only, and the other circuit effects are in the P tier (next section).

Its taps are ternary in every layer; only the digital readout on the microcontroller is float. If you insist on a ternary readout too, the focal-loss variant (below) scores 0.812 on the board line and 0.675 on the ideal one.

**The QPSK collapse, and the fix.** Our first board-aware network scored 0.797 on the board line and looked fine. It classified **no QPSK at all**: per-class QPSK accuracy was 0.00. QPSK and 8PSK had merged into one output, 8PSK absorbing the true QPSK, and the total stayed flat because 8PSK went from weak to decent. A mean accuracy hides a dead class. Our selection rule (mean validation accuracy over the two Q corners) could not see it.

The fix has two parts. First, select differently: a candidate is eligible only if its minimum per-class validation accuracy, on both corners, is at least 0.35, using 2,000 held-out training windows; the test set is never used to choose. Second, train differently: for the ternary readout, **focal loss** (gamma = 2) stopped the merge in 2 of 3 seeds, where plain cross-entropy collapsed in two seeds of three. Fine-tuning from the ideal-line net helped but stayed under the floor, and mixing ideal-line batches into board-line training hurt both. The float digital readout never collapsed, and focal loss on top of it lowered accuracy.

After the fix the weakest classes are still QPSK and 8PSK. The float-readout board net gets 0.690 on QPSK and 0.664 on 8PSK; the ternary-readout focal net gets 0.492 and 0.548. Per-class seed spread under focal loss is large (QPSK from zero to about 0.6 across seeds), so the floor rule matters more than the seed mean. Three seeds, one dataset, no independent replication.

## Four fidelities, one model

The same trained weights run through four simulations:

| tier | what it runs | what it omits |
|---|---|---|
| **V** (virtual) | ideal numpy math; with the M1a line FIR as the "board-line" reference | everything physical except the line response |
| **E** (emulated) | 8-bit fixed point, checked bit-for-bit against Verilog RTL (Icarus) | all analog effects |
| **C** (circuit) | ngspice on the KiCad-exported tile netlist, with datasheet-derived part models | layout, crosstalk, noise (stated in the run's notice) |
| **P** (parasitics) | scikit-rf ABCD cascade of the real LC sections with parasitics, mutual inductance, ribbon, plus a 1,000-board Monte Carlo | EM and layout; **it is lumped-element at about 1 MHz, not RF** |

The coax study's R (RF) tier is retired to the appendix. At 1 MHz there is no transmission-line physics on a 16 cm board, so we stopped pretending.

**V.** On the board line (the ngspice FIR) the board-aware net scores 0.878; on an ideal line 0.655.

**E.** The bit-true tier and the RTL are phase-1 evidence: 198,144 values compared at width 8 with 0 mismatches, and the width-32 numbers above. [[TBD: E-tier RTL run on the d = 1 board model; the tile maths is the same ternary layer, but the numbers cited here are for the phase-1 nets]]

**C.** [[C-tier pending: ngspice on the exported tile netlist, at least 64 examples per layer end to end; fill with C accuracy against V (board line) and against P, per-layer NMSE, and the accuracy of the calibrated behavioural model on the 4000-example set]]

**P.** Nominal parasitics, the full 4,000-example test set, and datasheet-minimum Q give 0.8705 (Q = 10) and 0.872 (Q = 30). Against the right reference, V on the board-line FIR (0.878), that is -0.75 points, with 0.948 top-1 agreement between P and V. (Against ideal-line V the P tier looks 21 points better, which is a property of the board-aware training and not a hardware gain. Ignore that comparison.)

The degradation-lab widget interpolates a precomputed five-axis grid (`sweeps_grid.json`, five points per axis, one golden board); accuracy across the whole grid stays within a couple of points of nominal, and interpolated points are marked as such.

**Monte Carlo.** We drew 1,000 boards on a fixed stratified 800-example subset, with L and C at +/-5 percent, R at +/-1 percent, a 0.5 percent clock error, 2 mV of transistor V_BE mismatch and 6 ns of S/H phase error. Mean accuracy 0.856, sigma **0.6 points**, 5th percentile 0.848, worst board 0.841. Yield is 1.00, but read the rule before you celebrate: "yield" is accuracy at or above the ideal-line V accuracy on the same subset (0.655) minus 5 points, a threshold of 0.605. That threshold is low because the board-aware net scores poorly on the ideal line. The more informative number is the spread: tolerances at the datasheet widths cost essentially nothing, and the single largest contributors (gain tolerance, V_BE mismatch) each have a sigma of about 0.4 points. A one-at-a-time sweep over the tolerance axes moves accuracy by under about a point, mostly inside the subset's own noise (about 1.1 points standard error for 1,000 examples), so we found no tolerance knee in the swept ranges. The exceptions are the gain trim and a 10 dB stage SNR, both below.

What P does not model is long: no EM or layout, no inductor temperature drift or saturation, no op-amp slew or supply noise, no clock jitter beyond a static error, no charge-injection dependence on signal. It is a model of a board that has not been built, and **no hardware measurement exists**.

## Where it breaks

**The gain-trim finding (the biggest one).** Run the P tier on the board *as designed* and accuracy is **0.17**. With the right gain it is 0.87. The cause is a bug in our own design arithmetic: the summer's feedback resistor was sized as if each tap carried the full driven amplitude, but a matched ladder (1 kohm source and load against a 1 kohm line) delivers about half of it. The taps come out around 6 dB low relative to a fixed bias, and the three-layer cascade collapses to noise. The board-aware model was trained at the level where the mean on-lag tap gain is one, which needs 8.98 dB more than the as-built chain. A fixed 6 dB trim recovers only 0.45, and a sweep of the trim shows the network wants its gain level to within about 2 dB. We call this a design bug report for the hardware owner, not a tuned metric. The fix is larger R_f and R_b (equivalently, a driver gain), by about a factor 2.9 in the C-tier netlist. [[C-tier pending: confirm that the SPICE tile netlist and the final schematic both carry the R_f/R_b change, and report C accuracy with and without it]]

**Crosstalk turned out harmless, with a caveat.** Two neighbouring unshielded wirewound inductors couple magnetically, and the plan wanted -40 dB. A point-dipole estimate (we assumed an effective volume fraction, and that orthogonal axes keep 10 percent of the parallel coupling, so treat it as a guide to where to look and not as a field solve) gives a nearest-neighbour coupling of k = 0.027 for parallel axes at 3 mm. On the 1,000-example subset (nominal 0.850) we put that into the P tier four ways: parallel at 3 mm 0.859, alternating at 3 mm 0.850, parallel at 5 mm 0.851, alternating at 5 mm 0.849. All within the subset noise. The network does not care. The caveat: the -40 dB *target* is not met by the estimate at 3 mm even with alternating axes (and a 1812 body is 3.2 mm wide, so a 3 mm pitch is geometrically impossible); the layout uses a 4.2 mm row pitch inside a line and orthogonal axes between top and bottom lines, and the real coupling has never been measured.

**Ringing, end capacitors and tap spacing.** A constant-k ladder rings. A step at tap 7 overshoots by 23.2 percent in the lossless limit with the plan's 1.5 MHz input filter. Dropping the filter to 482 kHz cuts the lossless worst case to 8.8 percent, and with an assumed typical Q the overshoot is 2.8 percent (against 13.7 percent with the 1.5 MHz filter). The ringing is the sharp cut-off, not the inductor loss, so the filter has to hold even for a perfect line. The largest unplanned finding of the electrical design: full-value capacitors at the two end nodes of the ladder are a 60 pF shunt discontinuity. The tap spacing then ripples by 92.9 percent peak to peak over 20-250 kHz; half-value end capacitors cut that to 28.7 percent. What remains (31.9 percent peak to peak at Q = 10) we did **not** fix; it is deterministic, so the network is trained on it, and it is a real source of intersymbol interference. Random component tolerance adds a delay sigma of 18.3 ns at tap 7.

**The rest of the analog chain** passes in ngspice with generic and datasheet-derived models: crossbar "0" isolation of 56.1 dB at 500 kHz (budget 45 dB), a differential pair that fits tanh with R^2 = 1.0000 using a current-mirror tail, and a summer-to-track-and-hold slice with 0.99 percent rms error against an ideal tanh. A resistor tail, our first try, gave a visibly distorted tanh, which is why the mirror. Temperature is a real weakness: the tanh width drifts by 0.33 percent per kelvin and all layers drift together. These are **simulated** results with parts models we fitted; the BJT model is generic, not vendor-accurate.

## The board

Everything in this section is **designed, not fabricated**; nothing was measured. The boards were generated by scripts (schematic as code, then PCB placement, then freerouting) and checked with KiCad 9's ERC and DRC. You can see them, layer by layer and in 3D, in the viewer.

| board | state | DRC |
|---|---|---|
| Carrier for the NUCLEO-G474RE, with gold artwork | **routed** | {{board.carrier.drc | 0}} violations |
| TILE-8 layer 1, *reduced* (2 of 8 lines) | **routed** | {{board.l1.drc | 0}} violations, 0 unrouted |
| TILE-8 layer 2, full 8 lines | **placed, not routed** | not run to completion ({{board.l2.unconn | 499}} unconnected items) |
| TILE-8 layer 3, full 8 lines | **placed, not routed** | same |

**The carrier** is 134.1 x 90.1 mm, four layers, with 50 surface-mount parts in its placement file. It holds the two Morpho sockets for the Nucleo, the DAC and ADC conditioning and the 5 V rail. Its bottom side carries a piece of artwork in exposed ENIG gold: a copper island with a solder-mask opening, no net, kept at least half a millimetre from anything else, with no copper feature or gap under 0.2 mm. The artwork was supplied by the project owner. Gold needs ENIG; with HASL it would come out silver. DRC on the carrier with the art reports 0 violations. (The mask-bridge check is switched off in the art file only, because every seam between pieces of one aperture trips it, so we run it on the board without the art.)

**The reduced layer-1 tile** is 180.1 x 142.1 mm, with 2 delay lines of 8 (so 112 inductors, 16 taps), the full output chain, and a crossbar footprint grid cut to the populated lines. Its CPL has 508 rows, and 85 of the crossbar resistors are populated, equal to the model's 85 non-zero layer-1 weights. DRC: 0 violations. The inner layers are a GND plane directly under the inductor rows and a 5 V plane with GND islands.

**The full tiles are placed, not routed.** Layers 2 and 3 have all eight lines: 448 inductors, about 2,200 placements and 4,800 SMD pins each, and their populated crossbar resistors (315 and 316) match the model as well. Geometry checks are clean, the connections are not made. We cut the scope and routed the layer-1 slice, because autorouting the full tile did not finish (next section).

**Bill of materials, as an estimate.** Part prices are single readings of JLCPCB part pages and need a live quote. A full tile costs about $75.89 in parts, of which the 448 inductors are $44.49; the layer-1 tile about $32.21. The modelled cost of the whole width-8 demonstrator (three tiles, carrier, a Nucleo at an assumed $18, cables and a supply) is $406 with economic assembly or $439 standard, or $507 to $549 with a 25 percent contingency for shipping, duties and a respin. The plan's original guess was higher. The weak points of the BOM are stock and the extended-part fee for one inductor part used 448 times per tile, none of which we verified. Nobody has built this.

**Things a human must check before ordering:** the Nucleo Morpho header geometry (taken from the Nucleo-64 pitch, not from a real board); JLC's rotation offsets for the SOT-363, TSSOP and MSOP parts in the CPL; the inductor orientation and crosstalk; a live quote and DFM check; and the 3.3 V bias network, which was checked by ERC only.

## Scaling up (simulation only)

Width scales badly on this architecture, in a way that is easy to count. A layer of width W takes W input channels, each needing its own delay line, and sends each to W outputs. A tile has 8 lines and 8 outputs, so a layer needs (W/8)^2 tiles, with the partial sums stitched through a separate header. Inductors per network are 448 (D-1) (W/8)^2 + 112 (W/8) for depth D, and tiles (D-1)(W/8)^2 + W/8. From our cost model:

| width x depth | tiles | inductors |
|---|---|---|
| 8 x 3 (the demonstrator) | {{scale.W8_D3.tiles | 3}} | {{scale.W8_D3.ind | 1,008}} |
| 16 x 4 | {{scale.W16_D4.tiles | 14}} | {{scale.W16_D4.ind | 5,600}} |
| 32 x 4 (the "scaled virtual model") | {{scale.W32_D4.tiles | 52}} | {{scale.W32_D4.ind | 21,952}} |
| 64 x 6 | {{scale.W64_D6.tiles | 328}} | {{scale.W64_D6.ind | 144,256}} |

Width 32 with four layers needs 48 full tiles plus 4 layer-1 tiles, 52 in all, and 21,952 inductors. (The plan's earlier "48 tiles and about 21,500 inductors" counted only the full tiles.) That is 8,320 square centimetres of PCB and, at volume pricing without fixed fees, about $4,135 for the tiles, of which the inductors alone are $1,574. The virtual net at that size scores 0.931. We will not build 52 tiles: the width-32 network is **simulation only**, a point on a scaling curve that says the arithmetic of the architecture, not that anything works at that size. We make no claim past the (W/8)^2 counting, and none about accuracy of a real width-32 board, where gain and tolerance would have to hold across four layers.

The energy numbers earlier also scale against it: tile power is dominated by the roughly 106 op-amps per full tile (72 mA at 5 V by our estimate), not by the ternary taps, which cost nothing to hold. The digital-equivalent comparison gets worse for the analog side, not better, as you add tiles.

## What didn't work

**Routing the full tile.** We spent 17 minutes on freerouting's fanout stage for the full layer-2 tile and stopped: 3,547 of its SMD pins needed fanout and no autoroute pass had completed, with 660 connections left even after our scripted routing of the ladders. The 2-line slice took 18 minutes with the optimiser off (the optimiser ran out of Java heap and lost a whole run). The full tile is expected to need hours. Obvious next steps are a longer budget or two half-tiles routed separately. We did not try.

**The 8PSK/QPSK confusion.** It never went away. In the arena the ternary-readout DLNN scores 0.334 on 8PSK. The board-aware nets merge QPSK into 8PSK unless something stops them, and the focal-loss fix gives QPSK back at the cost of 8PSK. The two constellations differ by a 45-degree rotation and a carrier-phase offset the network does not know; a short window at low SNR does not separate them. We note that the cumulant and feature-MLP baselines have the same weak pair, so this is the data as much as the network.

**The MLP strawman.** Our first arena MLPs scored 0.285 and 0.547 (raw I/Q, minimal tuning) and we presented them as the conventional baseline. They were weak by construction. We retuned over a grid, with a validation split, up to three seeds, and learned that raw-I/Q MLPs cannot be rescued by tuning (now 0.261 and 0.542), while a feature MLP is competitive. The search optimum sat on the grid edge in several places, so a wider search might add a point or two. The live arena widget may still ship the first, weaker pair. [[TBD: refresh data/arena_models.json with the retuned MLPs and the feature-MLP rows]]

**The design arithmetic.** The gain-trim bug above came from our own spreadsheet; plan values for the feedback resistor, the header pin count, the delay sigma and the tile cost did not reproduce from the plan's own parts, and the numbers here supersede them. The crosstalk target is not demonstrated; the inductor is unshielded and its SRF is below the plan's requirement; the temperature drift of the tanh needs thermal control, a gain calibration, or a retrain that tolerates it.

**The LLM commentary experiment (appendix-level).** In phase 1 we asked a local language model to read a text card of the tier comparison and say which tier diverged. With the card as designed, which contains a divergence list computed in Python, it attributed 5 of 5 injected cases under a loose rule. Without that list it attributed 0 of 5, and it raised false alarms on the controls. It restated the card and did not analyse anything. We have removed the LLM from the default pipeline; the write-up lives in `docs/llm_commentary.md` and `docs/llm_eval.md`.

## Reproduce it

Everything runs from the repository. The path goes: a notebook or script writes JSON to `docs/artifacts/data/`; the page and the verify notebook both read those files.

- Phase-1 notebooks are in `notebooks/` (`00_overview` through `06_compare_and_llm`), one per tier.
- Training the board net: `python -m dlnn.model.train --name tdlnet_b_w8 --width 8 --dilations 1,1,1 --ternary-only`. A full re-run into a second directory gave a byte-identical weights file.
- Arena models: `python -m dlnn.model.arena`; the MLP search is `models/arena/mlp_search.py`.
- Line response and acceptance checks: `hw/board/spice/run_checks.py` (about 15 seconds, ngspice 42).
- Check the article against the data: `python scripts/check_article_numbers.py` re-reads every source field and fails on a mismatch.
- [[repro: Colab and Binder links for the notebooks, and the path and PASS/FAIL table of the `00_verify` notebook, pending M6]]

Licences planned for the public repository: MIT for code, CERN-OHL-P-2.0 for hardware, CC BY 4.0 for the text. The RadioML datasets are not redistributed [deepsig]. The energy references are Horowitz (ISSCC 2014, Fig. 1.1.9) and the STM32G474 datasheet DS12288 Rev 2, table 21; both are in `models/arena/energy_refs.json`.

## References

- [appeltant2011] Appeltant, L.; Soriano, M. C.; Van der Sande, G.; et al. (2011). Information processing using a single dynamical node as complex system. Nature Communications 2:468. https://doi.org/10.1038/ncomms1476
- [bai2018] Bai, S.; Kolter, J. Z.; Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271. https://arxiv.org/abs/1803.01271
- [bengio2013] Bengio, Y.; Leonard, N.; Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432. https://arxiv.org/abs/1308.3432
- [boyle1970] Boyle, W. S.; Smith, G. E. (1970). Charge Coupled Semiconductor Devices. Bell System Technical Journal 49(4):587-593. https://doi.org/10.1002/j.1538-7305.1970.tb01790.x
- [cam_edsac] University of Cambridge Computer Laboratory (1999). Some EDSAC statistics (EDSAC 99 pages). cl.cam.ac.uk. https://www.cl.cam.ac.uk/events/EDSAC99/statistics.html
- [deepsig] DeepSig Inc. (2026). RadioML datasets page (licence statement). deepsig.ai. https://www.deepsig.ai/datasets/
- [eckert1949] Auerbach, J. A.; Eckert, J. P.; Shaw, R. F.; Sheppard, C. B. (1949). Mercury Delay Line Memory Using a Pulse Rate of Several Megacycles. Proceedings of the IRE 37(8):855-861. https://doi.org/10.1109/JRPROC.1949.229683
- [kallmann1940] Kallmann, H. E. (1940). Transversal Filters. Proceedings of the IRE 28(7):302-310. https://doi.org/10.1109/jrproc.1940.229346
- [li2016] Li, F.; Zhang, B.; Liu, B. (2016). Ternary Weight Networks. arXiv:1605.04711. https://arxiv.org/abs/1605.04711
- [ma2024] Ma, S.; Wang, H.; Ma, L.; et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764. https://arxiv.org/abs/2402.17764
- [merton2026] Merton, N. A. (2026). Scalable and Reconfigurable True-Time Delay Line for Integrated Radio-Frequency Recurrent Neural Processors. Master's thesis, Virginia Tech. https://vtechworks.lib.vt.edu/items/21439a7f-042e-417e-b6ca-0e2f0dd8f45a
- [pour2025] Pour, F. L.; Ha, D. S. (2025). Toward Analog Implementation of Recurrent Neural Networks in the RF Domain for Proactive Anomaly Detection in B5G Wireless Networks. IEEE International Symposium on Circuits and Systems (ISCAS) 2025. https://doi.org/10.1109/iscas56072.2025.11043369
- [sangster1969] Sangster, F. L. J.; Teer, K. (1969). Bucket-brigade electronics: new possibilities for delay, time-axis conversion, and scanning. IEEE Journal of Solid-State Circuits 4(3):131-136. https://doi.org/10.1109/jssc.1969.1049975
- [tancrell1971] Tancrell, R. H.; Holland, M. G. (1971). Acoustic surface wave filters. Proceedings of the IEEE 59(3):393-409. https://doi.org/10.1109/proc.1971.8180
- [tank1987] Tank, D. W.; Hopfield, J. J. (1987). Neural computation by concentrating information in time. Proceedings of the National Academy of Sciences 84(7):1896-1900. https://doi.org/10.1073/pnas.84.7.1896
- [unnikrishnan1991] Unnikrishnan, K. P.; Hopfield, J. J.; Tank, D. W. (1991). Connected-digit speaker-dependent speech recognition using a neural network with time-delayed connections. IEEE Transactions on Signal Processing 39(3):698-713. https://doi.org/10.1109/78.80888
- [vonneumann1945] von Neumann, J. (1993). First draft of a report on the EDVAC (written 1945; reprint). IEEE Annals of the History of Computing 15(4):27-75. https://doi.org/10.1109/85.238389
- [waibel1989] Waibel, A.; Hanazawa, T.; Hinton, G.; Shikano, K.; Lang, K. J. (1989). Phoneme recognition using time-delay neural networks. IEEE Trans. Acoustics, Speech, and Signal Processing 37(3):328-339. https://doi.org/10.1109/29.21701
- [wavenet2016] van den Oord, A.; Dieleman, S.; Zen, H.; et al. (2016). WaveNet: A Generative Model for Raw Audio. arXiv:1609.03499. https://arxiv.org/abs/1609.03499
- [widrow1960] Widrow, B.; Hoff, M. E. (1960). Adaptive switching circuits. IRE WESCON Convention Record, Part 4, pp. 96-104. https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf
- [wilkes1950] Wilkes, M. V.; Renwick, W. (1950). The EDSAC (Electronic Delay Storage Automatic Calculator). Mathematical Tables and Other Aids to Computation 4(30):61-65. https://doi.org/10.2307/2002769
- [woo2024] Woo, J.; Jung, K.; Mukhopadhyay, S. (2024). Efficient Hardware Design of DNN for RF Signal Modulation Recognition Employing Ternary Weights. IEEE Access 12:80165-80175. https://doi.org/10.1109/access.2024.3409180
- [wp_delayline] Wikipedia contributors (2026). Delay-line memory. Wikipedia. https://en.wikipedia.org/wiki/Delay-line_memory

Energy references (not in the literature list; read from primary sources, see models/arena/energy_refs.json): Horowitz, ISSCC 2014, Fig. 1.1.9; STMicroelectronics DS12288 Rev 2, table 21.

---
Evidence chips are rendered here as bare values; every value resolves in `docs/artifacts/data/claims.json` and is checked by `scripts/check_article_numbers.py`.
