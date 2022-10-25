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

#define SURFACE_DFT_LENGTH						2048

#define SURFACE_LG_NUM_UNITS					128

#define SURFACE_LG_NUM_OCTAVES					8.0f

#define SURFACE_LG_BANDWIDTH					0.70f

#define SURFACE_LG_STD							8.0f

#define SURFACE_INPUT_DFT_NOISE					0.00001f

#define SURFACE_OUTPUT_DFT_NOISE				0.00001f

#define SURFACE_LG_SPIKE_MIN_POW				0.00001f

#define SURFACE_LG_SPIKE_THRESHOLD				1.0f 

#define SURFACE_LG_SPIKE_POW_RATIO				1.618f

#define SURFACE_LG_SPIKE_DECAY					0.88f
