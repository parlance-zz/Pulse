#pragma once

#include <math.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#include <crtdbg.h>
#include <chrono>
#include <stdint.h>
#include <assert.h>

#ifdef _DEBUG
#define _DEBUG_FILTERS
#endif

#define SURFACE_DFT_LENGTH						6144 
#define SURFACE_DFT_FILTER_TRUNCATE_THRESHOLD	0.00001f

#define SURFACE_LG_NUM_CHANNELS			1
#define SURFACE_LG_NUM_UNITS			256
#define SURFACE_LG_NUM_OCTAVES			10.85f 
#define SURFACE_LG_BANDWIDTH			1.0f
#define SURFACE_LG_STD					8.0f

#define SURFACE_INPUT_DFT_NOISE			0.00001f
#define SURFACE_OUTPUT_DFT_NOISE		0.00001f

#define SURFACE_LG_SPIKE_MIN_POW		0.00001f
#define SURFACE_LG_SPIKE_THRESHOLD		1.0f 
#define SURFACE_LG_SPIKE_POW_RATIO		1.55f
#define SURFACE_LG_SPIKE_DECAY			0.88f

#define MEM_ALIGNMENT		64
#define _ALIGNED		   

#ifndef M_PI
#define M_PI	3.141592653589793238462643383f
#endif

using namespace std;

// avx2 "horizontal" reduction helper functions
inline float hsum_ps_sse3(const __m128 v)
{
	__m128 shuf = _mm_movehdup_ps(v);
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}

inline float hsum256_ps_avx(const __m256 v)
{
	__m128 vlow = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1);	
	return hsum_ps_sse3(_mm_add_ps(vlow, vhigh));	
}

inline __m256 hmax_ps_avx(const __m256 v1)
{
    __m256 v2 = _mm256_permute_ps(v1, 0b10'11'00'01);
    __m256 v3 = _mm256_max_ps(v1, v2); 
    __m256 v4 = _mm256_permute_ps(v3, 0b00'00'10'10);
    __m256 v5 = _mm256_max_ps(v3, v4);
    __m128 v6 = _mm256_extractf128_ps(v5, 1);
    __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6);
	return _mm256_insertf128_ps(_mm256_castps128_ps256(v7), v7, 1);
}

struct LG_PARAMS { float q, std, gain, outWeight, decay; }; // log normal filter parameters

// construct, sample, and scatter a log normal filter set
struct FILTER_SET
{
	int numFilters;
	int dftLength;
	float filterTruncateThreshold;
	float filterSampleScale;

	LG_PARAMS *lgParams;

	int filterBufferLength;
	_ALIGNED float *filterBuffer, *filterGain, *filterScale;
	_ALIGNED int *filterLength, *filterBufferOffset;
	_ALIGNED int *filterDFTOffset;

	_ALIGNED float *totalResponse, *normalizedResponse;

	void Init(int _numFilters, int _dftLength, float _filterTruncateThreshold, const LG_PARAMS *_lgParams)
	{
		numFilters = _numFilters;
		dftLength = _dftLength;
		filterTruncateThreshold = _filterTruncateThreshold;
		filterSampleScale = 1.0f / float(dftLength);

		lgParams = (LG_PARAMS*)malloc(numFilters * sizeof(LG_PARAMS));
		memcpy(lgParams, _lgParams, numFilters * sizeof(LG_PARAMS));

		filterLength = (int*)_aligned_malloc(numFilters * sizeof(int), MEM_ALIGNMENT);
		filterBufferOffset = (int*)_aligned_malloc(numFilters * sizeof(int), MEM_ALIGNMENT);
		filterDFTOffset = (int*)_aligned_malloc(numFilters * sizeof(int), MEM_ALIGNMENT);
		filterGain = (float*)_aligned_malloc(numFilters * sizeof(float), MEM_ALIGNMENT);
		filterScale = (float*)_aligned_malloc(numFilters * sizeof(float), MEM_ALIGNMENT);

		float *response = (float*)_aligned_malloc(dftLength * sizeof(float), MEM_ALIGNMENT);
		filterBuffer = (float*)_aligned_malloc(dftLength * numFilters * sizeof(float), MEM_ALIGNMENT); filterBufferLength = 0;

		totalResponse = (float*)_aligned_malloc(dftLength * sizeof(float), MEM_ALIGNMENT); memset(totalResponse, 0, dftLength * sizeof(float));
		normalizedResponse = (float*)_aligned_malloc(dftLength * sizeof(float), MEM_ALIGNMENT); memset(normalizedResponse, 0, dftLength * sizeof(float));

		for (int lg = 0; lg < numFilters; lg++)
		{
			filterGain[lg] = lgParams[lg].gain;
			filterScale[lg] = 1.0f / filterGain[lg];

			// calculate filter kernel
			for (int q = 0; q < dftLength; q++)
			{
				float dftQ = float(q + 0.5f) / float(dftLength);
				float a = log(dftQ / lgParams[lg].q) * lgParams[lg].std;
				response[q] = expf(-a * a);
				totalResponse[q] += response[q];
				normalizedResponse[q] += response[q] * filterGain[lg];
			}

			// crop filter kernel to truncate threshold
			int filterStart, filterEnd; float pow = 0.0f;
			for (filterStart = 0; filterStart < dftLength; filterStart++)
			{
				pow += response[filterStart] * response[filterStart];
				if (pow >= SURFACE_DFT_FILTER_TRUNCATE_THRESHOLD) break;
			}
			filterStart = (filterStart / 8) * 8; pow = 0.0f;
			for (filterEnd = dftLength - 1; filterEnd >= (filterStart + 8); filterEnd--)
			{
				pow += response[filterEnd] * response[filterEnd];
				if (pow >= filterTruncateThreshold) break;
			}
			filterEnd = ((filterEnd + 7) / 8) * 8;
			if (filterEnd < (filterStart + 8)) filterEnd = filterStart + 8;
			
			filterLength[lg] = (filterEnd - filterStart); _ASSERT(filterLength[lg] >= 8);
			filterBufferOffset[lg] = filterBufferLength;
			filterDFTOffset[lg] = filterStart;

			for (int s = filterStart; s < filterEnd; s++) response[s] *= filterGain[lg];
			memcpy(&filterBuffer[filterBufferOffset[lg]], &response[filterStart], (filterEnd - filterStart) * sizeof(float));
			filterBufferLength += filterLength[lg];
		}

		filterBuffer = (float*)_aligned_realloc(filterBuffer, filterBufferLength * sizeof(float), MEM_ALIGNMENT);
		for (int s = 0; s < filterBufferLength; s++) if (s & 1) filterBuffer[s] *= -1.0f; // rotate phase of filters by 180 degrees
																						  // to sample from the middle of the DFT window
		_aligned_free(response);
	}

	// sample an lg filter's response from 2 reference DFTs
	void Sample2(const int lg, const float *dftX, const float *dftY, float *outX, float *outY, const float *dftX2, const float *dftY2, float *outX2, float *outY2)
	{
		int q = filterDFTOffset[lg];
		__m256 _lgx = _mm256_setzero_ps(), _lgy = _mm256_setzero_ps();
		__m256 _lgx2 = _mm256_setzero_ps(), _lgy2 = _mm256_setzero_ps();
		for (int b = filterBufferOffset[lg]; b < (filterBufferOffset[lg] + filterLength[lg]); b += 8)
		{
			__m256 filterX = _mm256_load_ps(&filterBuffer[b]);

			_lgx = _mm256_fmadd_ps(_mm256_load_ps(&dftX[q]), filterX, _lgx);
			_lgy = _mm256_fmadd_ps(_mm256_load_ps(&dftY[q]), filterX, _lgy);

			_lgx2 = _mm256_fmadd_ps(_mm256_load_ps(&dftX2[q]), filterX, _lgx2);
			_lgy2 = _mm256_fmadd_ps(_mm256_load_ps(&dftY2[q]), filterX, _lgy2);

			q += 8;
		}

		*outX = hsum256_ps_avx(_lgx) * filterSampleScale;
		*outY = hsum256_ps_avx(_lgy) * filterSampleScale;

		*outX2 = hsum256_ps_avx(_lgx2) * filterSampleScale;
		*outY2 = hsum256_ps_avx(_lgy2) * filterSampleScale;
	}

	// scatter / mix lg filter into a reference DFT
	void Scatter(const int lg, const float x, const float y, float *dftX, float *dftY)
	{
		__m256 magX = _mm256_set1_ps(x * filterScale[lg]), magY = _mm256_set1_ps(y * filterScale[lg]);
		__m256 m_magY = _mm256_set1_ps(-y * filterScale[lg]);

		int q = filterDFTOffset[lg];
		for (int b = filterBufferOffset[lg]; b < (filterBufferOffset[lg] + filterLength[lg]); b += 8)
		{
			__m256 filterX = _mm256_load_ps(&filterBuffer[b]);

			_mm256_store_ps(&dftX[q], _mm256_fmadd_ps(filterX, magX, _mm256_load_ps(&dftX[q])));
			_mm256_store_ps(&dftY[q], _mm256_fmadd_ps(filterX, magY, _mm256_load_ps(&dftY[q])));

			q += 8;
		}
	}

	// rescale buffer to 1.
	float Normalize(_ALIGNED float *buff, int bufferLen)
	{
		_ASSERT(bufferLen % 8 == 0);
		__m256 maxVal = _mm256_setzero_ps();
		for (int i = 0; i < bufferLen; i += 8)
			maxVal = _mm256_max_ps(maxVal, _mm256_load_ps(&buff[i]));
		maxVal = hmax_ps_avx(maxVal);

		__m256 invMaxVal = _mm256_div_ps(_mm256_set1_ps(1.0f), maxVal);
		for (int i = 0; i < bufferLen; i += 8)
			_mm256_store_ps(&buff[i], _mm256_mul_ps(_mm256_load_ps(&buff[i]), invMaxVal));

		return _mm256_cvtss_f32(maxVal);
	}

	// create an aligned memory copy of a given vector<float>. buffer will be zero-padded to 256 bits
	_ALIGNED float *GetAlignedCopy(vector<float> &v, int *bufferLen)
	{
		int newSize = (v.size() % 8) ? ((v.size() + 7) & ~7) : v.size();
		_ALIGNED float *buff = (_ALIGNED float*)_aligned_malloc(newSize * sizeof(float), MEM_ALIGNMENT);
		if (newSize != v.size()) memset(&buff[v.size()], 0, (newSize - v.size()) * sizeof(float));
		
		memcpy(buff, &v[0], v.size() * sizeof(float));
		*bufferLen = newSize; return buff;
	}

	// dump filters and calculated total filter response to float32 2-channel raw files
	void Dump(string _responseFile, string _bufferFile)
	{
		Normalize(totalResponse, dftLength);
		Normalize(normalizedResponse, dftLength);

		FILE *responseFile = fopen(_responseFile.c_str(), "wb");
		for (int q = 0; q < dftLength; q++) 
		{
			fwrite(&totalResponse[q], 1, sizeof(float), responseFile);
			fwrite(&normalizedResponse[q], 1, sizeof(float), responseFile);
		}
		fclose(responseFile);

		FILE *filterBufferFile = fopen(_bufferFile.c_str(), "wb");
		for (int s = 0; s < filterBufferLength; s++)
		{
			fwrite(&filterBuffer[s], 1, sizeof(float), filterBufferFile);
			float p = abs(filterBuffer[s]); fwrite(&p, 1, sizeof(float), filterBufferFile);
		}
		fclose(filterBufferFile);
	}

	void Shutdown()
	{
		_aligned_free(filterBuffer);
		_aligned_free(filterGain);
		_aligned_free(filterScale);
		_aligned_free(filterLength);
		_aligned_free(filterBufferOffset);
		_aligned_free(filterDFTOffset);
		_aligned_free(totalResponse);
		_aligned_free(normalizedResponse);

		free(lgParams);
	}
};

struct SURFACE
{
	float lastDFTInX, lastDFTOutX, lastLGOut;

	_ALIGNED float *dftX, *dftY, *dftTX, *dftTY; 
	_ALIGNED float *out_dftX, *out_dftY;

	LG_PARAMS *lgParams;
	FILTER_SET lgFilters;

	_ALIGNED float *lgX, *lgY, *lgPow;
	_ALIGNED float *out_lgX, *out_lgY, *out_lgPow, *out_lgP;

	_ALIGNED int64_t *lgLastSpikeTick, *lgSpikeCount;
	vector<uint8_t> lgSpikeIntervals[SURFACE_LG_NUM_UNITS];
	int totalActivity;

	int64_t ticks;

	mt19937 rng;
	float noise;

	float inline Randf() { return (rng() & 0xFFFFFF) / float(0xFFFFFF); }

	// very simple (non-optimal) quantization
	inline uint8_t QuantizeInterval(int64_t interval)
	{
		int qInterval = int(16.0 * log(interval));
		if (qInterval > 255) qInterval = 255; if (qInterval < 0) qInterval = 0;
		return uint8_t(qInterval);
	}
	inline int64_t DequantizeInterval(uint8_t qInterval) { return exp(qInterval / 16.0); }

	// initialize filters as well as both the input and output sliding DFTs
	void Init()
	{
		// init sliding DFTs
		_ASSERT((SURFACE_DFT_LENGTH % 8) == 0);
		_ASSERT((SURFACE_LG_NUM_UNITS % 8) == 0);

		lastDFTInX = 0.0f; lastLGOut = 0.0f;
		dftX = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT); memset(dftX, 0, SURFACE_DFT_LENGTH * sizeof(float));
		dftY = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT); memset(dftY, 0, SURFACE_DFT_LENGTH * sizeof(float));
		
		dftTX = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT);
		dftTY = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT);
		for (int q = 0; q < SURFACE_DFT_LENGTH; q++) // calculate sliding dft "complex IIRs"
		{
			float dftQ = float(q + 0.5f) / float(SURFACE_DFT_LENGTH);
			dftTX[q] = cosf(dftQ * float(M_PI)); 
			dftTY[q] = sinf(dftQ * float(M_PI));
		}

		lastDFTOutX = 0.0f;
		out_dftX = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT); memset(out_dftX, 0, SURFACE_DFT_LENGTH * sizeof(float));
		out_dftY = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT); memset(out_dftY, 0, SURFACE_DFT_LENGTH * sizeof(float));

		lgX = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(lgX, 0, SURFACE_LG_NUM_UNITS * sizeof(float));
		lgY = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(lgY, 0, SURFACE_LG_NUM_UNITS * sizeof(float));
		lgPow = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(lgPow, 0, SURFACE_LG_NUM_UNITS * sizeof(float));

		out_lgX = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(out_lgX, 0, SURFACE_LG_NUM_UNITS * sizeof(float));
		out_lgY = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(out_lgY, 0, SURFACE_LG_NUM_UNITS * sizeof(float));
		out_lgPow = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(out_lgPow, 0, SURFACE_LG_NUM_UNITS * sizeof(float));
		out_lgP = (float*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(float), MEM_ALIGNMENT); memset(out_lgP, 0, SURFACE_LG_NUM_UNITS * sizeof(float));

		// init lg filter parameters
		lgParams = (LG_PARAMS*)malloc(sizeof(LG_PARAMS) * SURFACE_LG_NUM_UNITS);
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			float x = 1.0f - float(lg) / float(SURFACE_LG_NUM_UNITS - 1);
			lgParams[lg].q = exp2f(-x * SURFACE_LG_NUM_OCTAVES) * SURFACE_LG_BANDWIDTH;
			lgParams[lg].std = SURFACE_LG_STD;
			lgParams[lg].gain = 1.0f;
			lgParams[lg].decay = SURFACE_LG_SPIKE_DECAY;
			lgParams[lg].outWeight = 1.0f;
		}
		lgFilters.Init(SURFACE_LG_NUM_UNITS, SURFACE_DFT_LENGTH, SURFACE_DFT_FILTER_TRUNCATE_THRESHOLD, lgParams);
#ifdef _DEBUG_FILTERS
		lgFilters.Dump("totalResponse.raw", "filterBuffer.raw");
#endif

		// spike intervals for each filter are stored in a per filter vector/list
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++) lgSpikeIntervals[lg].clear();
		lgLastSpikeTick = (int64_t*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(int64_t), MEM_ALIGNMENT); memset(lgLastSpikeTick, 0, SURFACE_LG_NUM_UNITS * sizeof(int64_t));
		lgSpikeCount = (int64_t*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(int64_t), MEM_ALIGNMENT); memset(lgSpikeCount, 0, SURFACE_LG_NUM_UNITS * sizeof(int64_t));

		totalActivity = 0;
		ticks = 0;
	}

	void Shutdown()
	{
		_aligned_free(dftX); _aligned_free(dftY);
		_aligned_free(dftTX); _aligned_free(dftTY);
		_aligned_free(out_dftX); _aligned_free(out_dftY);

		free(lgParams);
		lgFilters.Shutdown();

		_aligned_free(lgX); _aligned_free(lgY); _aligned_free(lgPow);
		_aligned_free(out_lgX); _aligned_free(out_lgY); _aligned_free(out_lgPow);

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++) lgSpikeIntervals[lg].clear();
		_aligned_free(lgLastSpikeTick);
	}

	// sliding dft with avx2
	void SlidingDFT_AVX_InOut()
	{
		__m256 inX = _mm256_set1_ps(-lastDFTInX), _lastInX = _mm256_setzero_ps();
		__m256 outX = _mm256_set1_ps(-lastDFTOutX), _lastOutX = _mm256_setzero_ps();
		__m256 _minusone = _mm256_set1_ps(-1.0f);

		for (int q = 0; q < SURFACE_DFT_LENGTH; q += 8)
		{
			__m256 dfttx = _mm256_load_ps(&dftTX[q]), dftty = _mm256_load_ps(&dftTY[q]);
			__m256 m_dftty = _mm256_mul_ps(dftty, _minusone);

			__m256 dftx = _mm256_load_ps(&dftX[q]), dfty = _mm256_load_ps(&dftY[q]);
			dftx = _mm256_add_ps(dftx, inX);

			__m256 ny = _mm256_fmadd_ps(dfty, dfttx, _mm256_mul_ps(dftx, dftty));
			__m256 nx = _mm256_fmadd_ps(dfty, m_dftty, _mm256_mul_ps(dftx, dfttx));

			_mm256_store_ps(&dftX[q], nx);
			_mm256_store_ps(&dftY[q], ny);

			_lastInX = _mm256_add_ps(nx, _lastInX);

			dftx = _mm256_load_ps(&out_dftX[q]); dfty = _mm256_load_ps(&out_dftY[q]);
			dftx = _mm256_add_ps(dftx, outX);

			ny = _mm256_fmadd_ps(dfty, dfttx, _mm256_mul_ps(dftx, dftty));
			nx = _mm256_fmadd_ps(dfty, m_dftty, _mm256_mul_ps(dftx, dfttx));

			_mm256_store_ps(&out_dftX[q], nx);
			_mm256_store_ps(&out_dftY[q], ny);

			_lastOutX = _mm256_add_ps(nx, _lastOutX);
		}

		lastDFTInX = hsum256_ps_avx(_lastInX) / float(SURFACE_DFT_LENGTH); 
		lastDFTOutX = hsum256_ps_avx(_lastOutX) / float(SURFACE_DFT_LENGTH);
	}

	// process a single sample from the input signal
	void Input(const float sample)
	{
		noise = (Randf() - 0.5f); // uniform (white) noise is required to noise all dft bins equally
		lastDFTOutX += noise * SURFACE_OUTPUT_DFT_NOISE;
		lastDFTInX -= sample - noise * SURFACE_INPUT_DFT_NOISE;
		SlidingDFT_AVX_InOut(); // advance sliding DFTs by one step with the new input sample

		// sample lg filter responses from both the input and output DFTs
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
			lgFilters.Sample2(lg, dftX, dftY, &lgX[lg], &lgY[lg], out_dftX, out_dftY, &out_lgX[lg], &out_lgY[lg]);

		// compute the abs power of each filter response
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg += 8)
		{
			__m256 _lgx = _mm256_load_ps(&lgX[lg]), _lgy = _mm256_load_ps(&lgY[lg]);
			__m256 _pow = _mm256_fmadd_ps(_lgy, _lgy, _mm256_mul_ps(_lgx, _lgx));
			_mm256_store_ps(&lgPow[lg], _mm256_sqrt_ps(_pow));

			_lgx = _mm256_load_ps(&out_lgX[lg]); _lgy = _mm256_load_ps(&out_lgY[lg]);
			_pow = _mm256_fmadd_ps(_lgy, _lgy, _mm256_mul_ps(_lgx, _lgx));
			_mm256_store_ps(&out_lgPow[lg], _mm256_sqrt_ps(_pow));
		}

		totalActivity = 0; lastLGOut = 0.0f;
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			if (out_lgPow[lg] > 0.0f)
			{
				out_lgX[lg] /= out_lgPow[lg];
				out_lgY[lg] /= out_lgPow[lg];
			}
			else
			{
				out_lgX[lg] = 0.0f;
				out_lgY[lg] = 1.0f;
				out_lgPow[lg] = SURFACE_LG_SPIKE_MIN_POW;
			}

			int64_t interval = ticks - lgLastSpikeTick[lg];
			if (interval >= 1) // spike cooldown
			{
				_ASSERT(out_lgP[lg] > 0.0f);

				float phaseDot = (lgX[lg] * out_lgX[lg] + lgY[lg] * out_lgY[lg]) / lgPow[lg];
				float powRatio = lgPow[lg] / out_lgP[lg];
				
				if ((phaseDot * powRatio) > SURFACE_LG_SPIKE_THRESHOLD) // if the abs power difference ratio exceeds our preset
				{														// preset threshold, generate a new spike
					uint8_t qInterval = QuantizeInterval(interval);
					int64_t dq = DequantizeInterval(qInterval);
					lgLastSpikeTick[lg] += dq;
					lgSpikeIntervals[lg].push_back(qInterval);

					lgSpikeCount[lg]++;
					totalActivity++;

					out_lgP[lg] *= SURFACE_LG_SPIKE_POW_RATIO; // the power of the spike is proportional to the current abs power
				}
			}

			// mix/scatter the filter response into the output DFT
			out_lgP[lg] *= lgParams[lg].decay;
			if (out_lgP[lg] < SURFACE_LG_SPIKE_MIN_POW) out_lgP[lg] = SURFACE_LG_SPIKE_MIN_POW;
			else lgFilters.Scatter(lg, out_lgX[lg] * out_lgP[lg], out_lgY[lg] * out_lgP[lg], out_dftX, out_dftY);

			lastLGOut += out_lgY[lg] * out_lgPow[lg] * lgParams[lg].outWeight;  // the output value for each timestep is the sum
		  																		// of the weighted filter responses (from the output DFT)
		}

		ticks++;
	}

	// similar to above but uses pre-loaded quantized spike intervals rather than an input signal
	float Output()
	{
		noise = (Randf() - 0.5f);
		lastDFTOutX += noise * SURFACE_OUTPUT_DFT_NOISE;
		SlidingDFT_AVX_InOut(); 

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
			lgFilters.Sample2(lg, dftX, dftY, &lgX[lg], &lgY[lg], out_dftX, out_dftY, &out_lgX[lg], &out_lgY[lg]);

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg += 8)
		{
			__m256 _lgx = _mm256_load_ps(&lgX[lg]), _lgy = _mm256_load_ps(&lgY[lg]);
			__m256 _pow = _mm256_fmadd_ps(_lgy, _lgy, _mm256_mul_ps(_lgx, _lgx));
			_mm256_store_ps(&lgPow[lg], _mm256_sqrt_ps(_pow));

			_lgx = _mm256_load_ps(&out_lgX[lg]); _lgy = _mm256_load_ps(&out_lgY[lg]);
			_pow = _mm256_fmadd_ps(_lgy, _lgy, _mm256_mul_ps(_lgx, _lgx));
			_mm256_store_ps(&out_lgPow[lg], _mm256_sqrt_ps(_pow));
		}

		totalActivity = 0; lastLGOut = 0.0f;
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			if (out_lgPow[lg] > 0.0f)
			{
				out_lgX[lg] /= out_lgPow[lg];
				out_lgY[lg] /= out_lgPow[lg];
			}
			else
			{
				out_lgX[lg] = 0.0f;
				out_lgY[lg] = 1.0f;
				out_lgPow[lg] = SURFACE_LG_SPIKE_MIN_POW;
			}

			int64_t interval = ticks - lgLastSpikeTick[lg];
			if (interval == 0)
			{
				const int numIntervals = lgSpikeIntervals[lg].size();
				if (lgSpikeCount[lg] < numIntervals)
				{
					out_lgP[lg] *= SURFACE_LG_SPIKE_POW_RATIO;
					lgLastSpikeTick[lg] += DequantizeInterval(lgSpikeIntervals[lg][lgSpikeCount[lg]]);
					lgSpikeCount[lg]++;
					totalActivity++;
				}
			}

			out_lgP[lg] *= lgParams[lg].decay;
			if (out_lgP[lg] < SURFACE_LG_SPIKE_MIN_POW) out_lgP[lg] = SURFACE_LG_SPIKE_MIN_POW;
			else lgFilters.Scatter(lg, out_lgX[lg] * out_lgP[lg], out_lgY[lg] * out_lgP[lg], out_dftX, out_dftY);

			lastLGOut += out_lgY[lg] * out_lgPow[lg] * lgParams[lg].outWeight;
		}

		ticks++;
		return lastLGOut;
	}

	// save and load quantized spike intervals
	bool SaveQuants(string outputPath)
	{
		FILE *outFile = fopen(outputPath.c_str(), "wb");

		uint64_t numLGs = SURFACE_LG_NUM_UNITS;
		fwrite(&numLGs, 1, sizeof(uint64_t), outFile);
		fwrite(&ticks, 1, sizeof(uint64_t), outFile);

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			uint64_t numIntervals = lgSpikeIntervals[lg].size();
			fwrite(&numIntervals, 1, sizeof(uint64_t), outFile);
			fwrite(&lgSpikeIntervals[lg][0], 1, numIntervals, outFile);
			printf("lg: %i spike density: %.2f%%\n", lg, 100.0f*float(numIntervals) / float(ticks));
		}

		fclose(outFile);
		return true;
	}

	size_t LoadQuants(string inputPath)
	{
		FILE *inFile = fopen(inputPath.c_str(), "rb");

		memset(lgSpikeCount, 0, SURFACE_LG_NUM_UNITS * sizeof(uint64_t));
		uint64_t numLGs; fread(&numLGs, 1, sizeof(uint64_t), inFile); _ASSERT(numLGs == SURFACE_LG_NUM_UNITS);
		size_t sampleLength; fread(&sampleLength, 1, sizeof(uint64_t), inFile);

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			uint64_t numIntervals; fread(&numIntervals, 1, sizeof(uint64_t), inFile);
			lgSpikeIntervals[lg].resize(numIntervals); fread(&lgSpikeIntervals[lg][0], 1, numIntervals, inFile);
		}

		fclose(inFile);
		return sampleLength;
	}
};