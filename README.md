Pulse - Pre/post-processing utility for generating quantized log-normally-distributed spike-intervals from raw audio, and back again.

How it works:
- This version's implementation uses a sliding linear DFT and a 128 unit log gabor filter bank.
- The sliding DFT is computed at every offset in the input signal and the log gabor filter bank is sampled.
- The ABS power of the LG filterbank responses is compared against the ABS power of the resampled filters after advancing the sliding DFT by 1 sample in time _without_ the knowledge of the input impulse for that step. If the difference in power exceeds a preset threshold, that filter is involuted back into the sliding DFT at a power level equal to the threshold multiplied by the original sampled response and this is considered a "spike" for that unit.
- For each LG filter this gives you a continuous stream of log-normally distributed intervals that I interpret as spike intervals. If it helps visualize picture a stream of spikes for each of the 128 filters over time, where each spike interval is quantized (slightly fuzzy and imprecise).
- This process can be reversed and the utility has both encode and decode modes.


Caveats:
- There's at least a few more viable versions of this idea in my old code files that are all somewhat different. There is another version in particular that uses a log-linear sliding DFT rather than a linear DFT; that version has a large number of advantages over this one but is also more complex.
- The LG "hyper-parameters" are probably still poorly optimized. Experiment with these values:
    - SURFACE_DFT_LENGTH
    - SURFACE_LG_NUM_UNITS
    - SURFACE_LG_NUM_OCTAVES
    - SURFACE_LG_BANDWIDTH
    - SURFACE_LG_STD
    - SURFACE_INPUT_DFT_NOISE
    - SURFACE_OUTPUT_DFT_NOISE
    - SURFACE_LG_SPIKE_MIN_POW
    - SURFACE_LG_SPIKE_THRESHOLD
    - SURFACE_LG_SPIKE_POW_RATIO
    - SURFACE_LG_SPIKE_DECAY
- The utility does not compress the quantized interval files (which can get a bit large due to the number of lg filters and lack of unused bit reduction in serialization). The less overall "spike density" the more compressible these files would theoretically be.
- Because this version uses a linear sliding DFT (as opposed to log-linear), low frequencies are difficult to sample without sacrificing performance or losing fidelity. As a result, frequencies < 50hz are fairly weak with the default LG filter bank settings in this repo (although you could just use a regular filter in post to strengthen them). The issue is we _do_ want very very low frequencies for a high quality generative model, not because they are audible or perceptible, but because including them in the intermediate representation allows the generator to do what it needs to do to create high quality music.
