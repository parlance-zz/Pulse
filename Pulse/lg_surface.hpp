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

#define _DEBUG_FILTERS

#define SURFACE_DFT_LENGTH						4096 //2048
#define SURFACE_DFT_FILTER_TRUNCATE_THRESHOLD	0.00001f //0.0001f

#define SURFACE_LG_NUM_CHANNELS					1
#define SURFACE_LG_NUM_UNITS					256 //128 
#define SURFACE_LG_NUM_OCTAVES					10.0f //9.5f 
#define SURFACE_LG_BANDWIDTH					0.9f
#define SURFACE_LG_MIN_DEVIATION				2.0f 
#define SURFACE_LG_STD_SCALE					0.35f

#define SURFACE_INPUT_DFT_NOISE					0.00001f
#define SURFACE_OUTPUT_DFT_NOISE				0.00001f

#define SURFACE_LG_SPIKE_MIN_POW				0.00001f
#define SURFACE_LG_SPIKE_THRESHOLD				1.0f 
#define SURFACE_LG_SPIKE_POW_RATIO				1.618f
#define SURFACE_LG_SPIKE_DECAY					0.95f

#define MEM_ALIGNMENT		64
#define _ALIGNED		   

#ifndef M_PI
#define M_PI	3.1415926535897932384626433832795f
#endif

using namespace std;

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

struct LG_PARAMS { float q, std, gain, outWeight, decay; };

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

			for (int q = 0; q < dftLength; q++)
			{
				float dftQ = float(q + 0.5f) / float(dftLength);
				float a = log(dftQ / lgParams[lg].q) * lgParams[lg].std;
				response[q] = expf(-a * a);
				totalResponse[q] += response[q];
				normalizedResponse[q] += response[q] * filterGain[lg];
			}

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

			filterLength[lg] = (filterEnd - filterStart);
			filterBufferOffset[lg] = filterBufferLength;
			filterDFTOffset[lg] = filterStart;

			for (int s = filterStart; s < filterEnd; s++) response[s] *= filterGain[lg];
			memcpy(&filterBuffer[filterBufferOffset[lg]], &response[filterStart], (filterEnd - filterStart) * sizeof(float));
			filterBufferLength += filterLength[lg];
		}

		filterBuffer = (float*)_aligned_realloc(filterBuffer, filterBufferLength * sizeof(float), MEM_ALIGNMENT);
		for (int s = 0; s < filterBufferLength; s++) if (s & 1) filterBuffer[s] *= -1.0f;

		_aligned_free(response);
	}

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

	void Dump(string _responseFile, string _bufferFile)
	{
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

	inline uint8_t QuantizeInterval(int64_t interval)
	{
		//int qInterval = int(2.078087 * log(interval) + 1.672276 + 0.5) - 2;
		int qInterval = int(16.0 * log(interval));
		if (qInterval > 255) qInterval = 255; if (qInterval < 0) qInterval = 0;

		return uint8_t(qInterval);
	}

	inline int64_t DequantizeInterval(uint8_t qInterval)
	{
		//return int64_t((pow(M_PHI, qInterval + 2) - pow(M_PHI - 1.0, qInterval + 2)) / M_RT5 + 0.5);
		return exp(qInterval / 16.0);
	}

	void Init()
	{
		_ASSERT((SURFACE_DFT_LENGTH % 8) == 0);
		_ASSERT((SURFACE_LG_NUM_UNITS % 8) == 0);

		lastDFTInX = 0.0f; lastLGOut = 0.0f;
		dftX = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT); memset(dftX, 0, SURFACE_DFT_LENGTH * sizeof(float));
		dftY = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT); memset(dftY, 0, SURFACE_DFT_LENGTH * sizeof(float));
		
		dftTX = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT);
		dftTY = (float*)_aligned_malloc(SURFACE_DFT_LENGTH * sizeof(float), MEM_ALIGNMENT);
		for (int q = 0; q < SURFACE_DFT_LENGTH; q++)
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

		lgParams = (LG_PARAMS*)malloc(sizeof(LG_PARAMS) * SURFACE_LG_NUM_UNITS);
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			float x = 1.0f - float(lg) / float(SURFACE_LG_NUM_UNITS - 1);
			lgParams[lg].q = exp2f(-x * SURFACE_LG_NUM_OCTAVES) * SURFACE_LG_BANDWIDTH;
			lgParams[lg].std = SURFACE_LG_MIN_DEVIATION * powf(lgParams[lg].q, SURFACE_LG_STD_SCALE) / powf(lgParams[0].q, SURFACE_LG_STD_SCALE);
			lgParams[lg].gain = powf(lgParams[lg].q, SURFACE_LG_STD_SCALE);
			//lgParams[lg].decay = powf(SURFACE_LG_SPIKE_DECAY, powf(lgParams[lg].q, SURFACE_LG_STD_SCALE));
			lgParams[lg].decay = SURFACE_LG_SPIKE_DECAY;
			//lgParams[lg].outWeight = powf(lgParams[lg].gain, 0.25f);
			lgParams[lg].outWeight = powf(lgParams[lg].gain, 0.5f);
		}

		lgFilters.Init(SURFACE_LG_NUM_UNITS, SURFACE_DFT_LENGTH, SURFACE_DFT_FILTER_TRUNCATE_THRESHOLD, lgParams);
		
#ifdef _DEBUG_FILTERS
		lgFilters.Dump("totalResponse.raw", "filterBuffer.raw");
#endif

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++) lgSpikeIntervals[lg].clear();
		lgLastSpikeTick = (int64_t*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(int64_t), MEM_ALIGNMENT); memset(lgLastSpikeTick, 0, SURFACE_LG_NUM_UNITS * sizeof(int64_t));
		lgSpikeCount = (int64_t*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(int64_t), MEM_ALIGNMENT); memset(lgSpikeCount, 0, SURFACE_LG_NUM_UNITS * sizeof(int64_t));

		//rng.seed(uint32_t(time(NULL)));
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

	void Input(const float sample)
	{
		noise = (Randf() - 0.5f);

		lastDFTOutX += noise * SURFACE_OUTPUT_DFT_NOISE;
		lastDFTInX -= sample - noise * SURFACE_INPUT_DFT_NOISE;
		
		SlidingDFT_AVX_InOut();

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++) lgFilters.Sample2(lg, dftX, dftY, &lgX[lg], &lgY[lg], out_dftX, out_dftY, &out_lgX[lg], &out_lgY[lg]);

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
			if (interval >= 1)
			{
				float phaseDot = (lgX[lg] * out_lgX[lg] + lgY[lg] * out_lgY[lg]) / lgPow[lg];
				float powRatio = lgPow[lg] / out_lgP[lg];
				
				const float threshold = SURFACE_LG_SPIKE_THRESHOLD;
				const float pow_ratio = SURFACE_LG_SPIKE_POW_RATIO;
				//threshold *= sqrtf(float(SURFACE_LG_NUM_UNITS) / float(lg+SURFACE_LG_NUM_UNITS));
				//pow_ratio *= sqrtf(float(SURFACE_LG_NUM_UNITS) / float(lg+SURFACE_LG_NUM_UNITS));

				if ((phaseDot * powRatio) > threshold)
				{
					uint8_t qInterval = QuantizeInterval(interval);
					int64_t dq = DequantizeInterval(qInterval);
					lgLastSpikeTick[lg] += dq;
					lgSpikeIntervals[lg].push_back(qInterval);

					lgSpikeCount[lg]++;
					totalActivity++;

					out_lgP[lg] *= pow_ratio;
				}
			}

			out_lgP[lg] *= lgParams[lg].decay;
			if (out_lgP[lg] < SURFACE_LG_SPIKE_MIN_POW) out_lgP[lg] = SURFACE_LG_SPIKE_MIN_POW;
			else lgFilters.Scatter(lg, out_lgX[lg] * out_lgP[lg], out_lgY[lg] * out_lgP[lg], out_dftX, out_dftY);

			lastLGOut += out_lgY[lg] * out_lgPow[lg] * lgParams[lg].outWeight;
		}

		ticks++;
	}

	float Output()
	{
		noise = (Randf() - 0.5f);
		lastDFTOutX += noise * SURFACE_OUTPUT_DFT_NOISE;

		SlidingDFT_AVX_InOut(); // ...

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++) lgFilters.Sample2(lg, dftX, dftY, &lgX[lg], &lgY[lg], out_dftX, out_dftY, &out_lgX[lg], &out_lgY[lg]);

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
				out_lgP[lg] *= SURFACE_LG_SPIKE_POW_RATIO;
				lgLastSpikeTick[lg] += DequantizeInterval(lgSpikeIntervals[lg][lgSpikeCount[lg]]);
				lgSpikeCount[lg]++;
				totalActivity++;
			}

			out_lgP[lg] *= lgParams[lg].decay;
			if (out_lgP[lg] < SURFACE_LG_SPIKE_MIN_POW) out_lgP[lg] = SURFACE_LG_SPIKE_MIN_POW;
			else lgFilters.Scatter(lg, out_lgX[lg] * out_lgP[lg], out_lgY[lg] * out_lgP[lg], out_dftX, out_dftY);

			lastLGOut += out_lgY[lg] * out_lgPow[lg] * lgParams[lg].outWeight;
		}

		ticks++;

		return lastLGOut;
	}

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

			printf("lg: %i spikes: %i \n", lg, int(numIntervals));
		}

		fclose(outFile);

		return true;
	}

	size_t LoadQuants(string inputPath)
	{
		FILE *inFile = fopen(inputPath.c_str(), "rb");

		memset(lgSpikeCount, 0, SURFACE_LG_NUM_UNITS * sizeof(uint64_t));
		uint64_t numLGs; fread(&numLGs, 1, sizeof(uint64_t), inFile);
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

/*
#pragma once

#include <math.h>
#include <stdio.h>
#include <vector>
#include <immintrin.h>
#include <intrin.h>
#include <crtdbg.h>
#include <chrono>
#include <stdint.h>
#include <string>
#include <algorithm>

//#define _DEBUG_FILTER
//#define _DEBUG_FLG_DELAYS
//#define _DEBUG_QUANT_INTERVALS
//#define _DEBUG_SPIKES
#define _DEBUG_INTERVAL_HISTOGRAM

#define SURFACE_LG_NUM_UNITS					256
#define SURFACE_LG_NUM_OCTAVES					14
#define SURFACE_LG_FILTER_WIDTH					32 
#define SURFACE_LG_FILTER_STD					140.0
#define SURFACE_LG_DECAY						0.92

#define SURFACE_LG_SPIKE_MAX_QUANT				16
#define SURFACE_LG_SPIKE_QUANT_I_SCALE			4.0
#define SURFACE_LG_SPIKE_QUANT_P_SCALE			1.25
#define SURFACE_LG_SPIKE_QUANT_P_MIN			-10.5
#define SURFACE_LG_SPIKE_QUANT_RESERVE			8192

#define SURFACE_OUTPUT_CLIP						0.125

#define SURFACE_LG_NUM_FILTER_UNITS				(SURFACE_LG_NUM_UNITS - SURFACE_LG_FILTER_WIDTH)
#define SURFACE_LG_FILTER_MAX_DELAY				(44100 * 4)

#define MEM_ALIGNMENT		64
#define _ALIGNED			__declspec(align(MEM_ALIGNMENT))

#define M_PI	3.1415926535897932384626433832795

using namespace std;

inline double sgnf(double x) { return (x >= 0) ? 1.0 : -1.0; }

inline double hsum256_pd_avx(__m256d v)
{
	__m128d vlow = _mm256_castpd256_pd128(v);
	__m128d vhigh = _mm256_extractf128_pd(v, 1);
	vlow = _mm_add_pd(vlow, vhigh);     
	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

struct QUANT
{
	uint8_t qi;	// quantized interval length
	uint8_t qp; // quantized amplitude
};

struct SURFACE
{
	_ALIGNED double *lgFilter, *lgFilterAbs, *lgQ, *lgTX, *lgTY;
	_ALIGNED double *lgX, *lgY, *flgX, *flgY, *flgPow, *out_flgPow;
	_ALIGNED double *out_lgX, *out_lgY;
	_ALIGNED int *flgDelay, *flgQuantIntervalScale;

	_ALIGNED int64_t *lgLastSpikeTick;
	vector<QUANT> lgSpikeIntervals[SURFACE_LG_NUM_FILTER_UNITS];

	int64_t ticks;

	inline uint8_t QuantizeInterval(int flg, int64_t interval, int64_t *dequantized) // intervals are quantized using a per filter pre-computed table
	{
		for (int q = 0; q < (SURFACE_LG_SPIKE_MAX_QUANT - 1); q++)
		{
			int64_t dq = flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q];
			if (dq >= interval)
			{
				if (dequantized != NULL) *dequantized = dq;
				return uint8_t(q);
			}
		}

		if (dequantized != NULL) *dequantized = flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + SURFACE_LG_SPIKE_MAX_QUANT - 1];
		return uint8_t(SURFACE_LG_SPIKE_MAX_QUANT - 1);
	}

	inline uint8_t QuantizePow(int flg, double pow)
	{
		double lg = log(pow) - SURFACE_LG_SPIKE_QUANT_P_MIN; if (lg < 0.0) lg = 0.0;

		int q = int(lg * SURFACE_LG_SPIKE_QUANT_P_SCALE + 0.5);
		if (q > (SURFACE_LG_SPIKE_MAX_QUANT - 1)) q = SURFACE_LG_SPIKE_MAX_QUANT;

		return uint8_t(q);
	}

	inline double DequantizePow(int flg, uint8_t q)
	{
		if (q == 0) return 0.0;
		return exp(double(q) / SURFACE_LG_SPIKE_QUANT_P_SCALE + SURFACE_LG_SPIKE_QUANT_P_MIN);
	}

	inline int64_t DequantizeInterval(int flg, uint8_t qInterval)
	{
		return flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + qInterval];
	}

	void Init()
	{
		_ASSERT((SURFACE_LG_NUM_UNITS % 4) == 0);
		_ASSERT((SURFACE_LG_FILTER_WIDTH % 4) == 0);
		_ASSERT(SURFACE_LG_NUM_UNITS > (SURFACE_LG_FILTER_WIDTH + 4));	
		_ASSERT((SURFACE_LG_SPIKE_MAX_QUANT % 4) == 0);
		_ASSERT(SURFACE_LG_SPIKE_MAX_QUANT <= SURFACE_LG_FILTER_WIDTH);

		ticks = 0;

		lgQ = (double*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(double), MEM_ALIGNMENT);
		lgTX = (double*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(double), MEM_ALIGNMENT);
		lgTY = (double*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(double), MEM_ALIGNMENT);
		
		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg++)
		{
			double x = 1.0 - 1.0 / double(SURFACE_LG_NUM_UNITS) - double(lg) / double(SURFACE_LG_NUM_UNITS);
			double y = double(SURFACE_LG_FILTER_WIDTH / 2) / double(SURFACE_LG_NUM_UNITS); // add some frequencies > 1 so we can filter closer to 1
			lgQ[lg] = exp2((y - x) * SURFACE_LG_NUM_OCTAVES);

			double decay = pow(SURFACE_LG_DECAY, lgQ[lg]);
			lgTX[lg] = cos(lgQ[lg] * M_PI) * decay;
			lgTY[lg] = sin(lgQ[lg] * M_PI) * decay;
		}

		lgX = (double*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(double), MEM_ALIGNMENT); memset(lgX, 0, SURFACE_LG_NUM_UNITS * sizeof(double));
		lgY = (double*)_aligned_malloc(SURFACE_LG_NUM_UNITS * sizeof(double), MEM_ALIGNMENT); memset(lgY, 0, SURFACE_LG_NUM_UNITS * sizeof(double));
		flgX = (double*)_aligned_malloc(SURFACE_LG_NUM_FILTER_UNITS * sizeof(double), MEM_ALIGNMENT); memset(flgX, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));
		flgY = (double*)_aligned_malloc(SURFACE_LG_NUM_FILTER_UNITS * sizeof(double), MEM_ALIGNMENT); memset(flgX, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));
		flgPow = (double*)_aligned_malloc(SURFACE_LG_NUM_FILTER_UNITS * sizeof(double), MEM_ALIGNMENT); memset(flgPow, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));
		out_flgPow = (double*)_aligned_malloc(SURFACE_LG_NUM_FILTER_UNITS * sizeof(double), MEM_ALIGNMENT); memset(out_flgPow, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));

		out_lgX = (double*)_aligned_malloc(SURFACE_LG_FILTER_WIDTH * SURFACE_LG_NUM_FILTER_UNITS * sizeof(double), MEM_ALIGNMENT); memset(out_lgX, 0, SURFACE_LG_FILTER_WIDTH * SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));
		out_lgY = (double*)_aligned_malloc(SURFACE_LG_FILTER_WIDTH * SURFACE_LG_NUM_FILTER_UNITS * sizeof(double), MEM_ALIGNMENT); memset(out_lgY, 0, SURFACE_LG_FILTER_WIDTH * SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));

		lgFilter = (double*)_aligned_malloc(SURFACE_LG_FILTER_WIDTH * sizeof(double), MEM_ALIGNMENT);
		lgFilterAbs = (double*)_aligned_malloc(SURFACE_LG_FILTER_WIDTH * sizeof(double), MEM_ALIGNMENT);
		for (int i = 0; i < SURFACE_LG_FILTER_WIDTH; i++)
		{
			double x = double(i) / double(SURFACE_LG_FILTER_WIDTH) - 0.5;
			lgFilter[i] = exp(-SURFACE_LG_FILTER_STD * x * x);
			lgFilterAbs[i] = lgFilter[i];
			if (i & 1) lgFilter[i] = -lgFilter[i];
		}
#ifdef _DEBUG_FILTER
		FILE *filterFile = fopen("filterBuffer.raw", "wb");
		fwrite(lgFilter, 1, SURFACE_LG_FILTER_WIDTH * sizeof(double), filterFile);
		fclose(filterFile);
#endif
	
		flgDelay = (int*)_aligned_malloc(SURFACE_LG_NUM_FILTER_UNITS * sizeof(int), MEM_ALIGNMENT); GetFLGDelay();
		flgQuantIntervalScale = (int*)_aligned_malloc(SURFACE_LG_SPIKE_MAX_QUANT * SURFACE_LG_NUM_FILTER_UNITS * sizeof(int), MEM_ALIGNMENT); GetIntervalQuantScales();

		for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
		{
			lgSpikeIntervals[flg].clear();
			lgSpikeIntervals[flg].reserve((flg + 1) * SURFACE_LG_SPIKE_QUANT_RESERVE);
		}

		lgLastSpikeTick = (int64_t*)_aligned_malloc(SURFACE_LG_NUM_FILTER_UNITS * sizeof(int64_t), MEM_ALIGNMENT); memset(lgLastSpikeTick, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(int64_t));
	}

	void Shutdown()
	{
		_aligned_free(lgQ); _aligned_free(lgTX); _aligned_free(lgTY);

		_aligned_free(lgFilter); _aligned_free(lgFilterAbs);
		_aligned_free(lgX); _aligned_free(lgY); 
		_aligned_free(out_lgX); _aligned_free(out_lgY);
		_aligned_free(flgX); _aligned_free(flgY); _aligned_free(flgPow); _aligned_free(out_flgPow);
		_aligned_free(flgDelay); _aligned_free(flgQuantIntervalScale);

		for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++) lgSpikeIntervals[flg].clear();
		_aligned_free(lgLastSpikeTick);
	}

	// sliding dft with avx2

	void SlidingDFT_AVX(double sample)
	{
		const __m256d inX = _mm256_set1_pd(sample);
		const __m256d _minusone = _mm256_set1_pd(-1.0);

		for (int lg = 0; lg < SURFACE_LG_NUM_UNITS; lg += 4)
		{
			const __m256d lgtx = _mm256_load_pd(&lgTX[lg]), lgty = _mm256_load_pd(&lgTY[lg]);
			const __m256d m_lgty = _mm256_mul_pd(lgty, _minusone);

			const __m256d lgx = _mm256_load_pd(&lgX[lg]), lgy = _mm256_load_pd(&lgY[lg]);
			__m256d ny = _mm256_fmadd_pd(lgy, lgtx, _mm256_mul_pd(lgx, lgty));
			__m256d nx = _mm256_fmadd_pd(lgy, m_lgty, _mm256_mul_pd(lgx, lgtx));

			const __m256d lgq = _mm256_load_pd(&lgQ[lg]);
			nx = _mm256_fmadd_pd(lgq, _mm256_mul_pd(inX, lgq), nx);

			_mm256_store_pd(&lgX[lg], nx);
			_mm256_store_pd(&lgY[lg], ny);
		}

		for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
		{
			for (int s = 0; s < SURFACE_LG_FILTER_WIDTH; s += 4)
			{
				const int lg = flg + s, out_lg = flg * SURFACE_LG_FILTER_WIDTH + s;

				const __m256d lgtx = _mm256_loadu_pd(&lgTX[lg]), lgty = _mm256_loadu_pd(&lgTY[lg]);
				const __m256d m_lgty = _mm256_mul_pd(lgty, _minusone);

				const __m256d lgx = _mm256_load_pd(&out_lgX[out_lg]), lgy = _mm256_load_pd(&out_lgY[out_lg]);
				__m256d ny = _mm256_fmadd_pd(lgy, lgtx, _mm256_mul_pd(lgx, lgty));
				__m256d nx = _mm256_fmadd_pd(lgy, m_lgty, _mm256_mul_pd(lgx, lgtx));

				_mm256_store_pd(&out_lgX[out_lg], nx);
				_mm256_store_pd(&out_lgY[out_lg], ny);
			}
		}
	}

	void InputSample(vector<double> &inputSample, vector<double> &outputSample)
	{
		outputSample.resize(inputSample.size());

#ifdef _DEBUG_INTERVAL_HISTOGRAM
		int *intervalHistogram = (int*)malloc(SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int));
		memset(intervalHistogram, 0, SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int));
		int *powHistogram = (int*)malloc(SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int));
		memset(powHistogram, 0, SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int));
#endif

		for (int64_t i = 0; i < inputSample.size(); i++)
		{
			SlidingDFT_AVX(inputSample[i]);
	
			for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
			{
				__m256d x = _mm256_setzero_pd(), y = _mm256_setzero_pd();
				for (int s = 0; s < SURFACE_LG_FILTER_WIDTH; s += 4)
				{
					const __m256d f = _mm256_load_pd(&lgFilter[s]);
					x = _mm256_fmadd_pd(_mm256_loadu_pd(&lgX[flg + s]), f, x);
					y = _mm256_fmadd_pd(_mm256_loadu_pd(&lgY[flg + s]), f, y);
				}
				const double nx = hsum256_pd_avx(x) / lgQ[flg + SURFACE_LG_FILTER_WIDTH / 2];
				const double ny = hsum256_pd_avx(y) / lgQ[flg + SURFACE_LG_FILTER_WIDTH / 2];

				if ((sgnf(nx) != sgnf(flgX[flg])) && (ny > 0.0)) flgPow[flg] = ny;
				else flgPow[flg] = 0.0;

				flgX[flg] = nx;
				flgY[flg] = ny;
			}

			for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
			{
				int64_t interval = ticks - lgLastSpikeTick[flg];
				int64_t minInterval = flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT];
				int64_t maxInterval = flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + SURFACE_LG_SPIKE_MAX_QUANT - 1];

				if (((interval >= minInterval) && (flgPow[flg] > 0.0)) || (interval >= maxInterval))
				{
					int64_t dq = 0; uint8_t qInterval = QuantizeInterval(flg, interval, &dq);
					lgLastSpikeTick[flg] += dq;

					uint8_t qp = QuantizePow(flg, flgPow[flg]);
					out_flgPow[flg] = DequantizePow(flg, qp);

					QUANT q; q.qi = qInterval; q.qp = qp;
					lgSpikeIntervals[flg].push_back(q);

#ifdef _DEBUG_INTERVAL_HISTOGRAM					
					intervalHistogram[flg * SURFACE_LG_SPIKE_MAX_QUANT + qInterval]++;
					powHistogram[flg * SURFACE_LG_SPIKE_MAX_QUANT + qp]++;
#endif
				}

				interval = ticks - lgLastSpikeTick[flg]; bool active = false;
				if (interval == 0)
				{
					const __m256d impulse = _mm256_set1_pd(out_flgPow[flg]);
					for (int s = 0; s < SURFACE_LG_FILTER_WIDTH; s += 4)
					{
						const __m256d lgy = _mm256_load_pd(&out_lgY[flg * SURFACE_LG_FILTER_WIDTH + s]);
						_mm256_store_pd(&out_lgY[flg * SURFACE_LG_FILTER_WIDTH + s], _mm256_add_pd(lgy, impulse));
					}

					active = true;
				}

				__m256d _x = _mm256_setzero_pd();
				for (int s = 0; s < SURFACE_LG_FILTER_WIDTH; s += 4)
				{
					const __m256d f = _mm256_load_pd(&lgFilter[s]);
					_x = _mm256_fmadd_pd(_mm256_load_pd(&out_lgX[flg * SURFACE_LG_FILTER_WIDTH + s]), f, _x);
				}
				double outX = hsum256_pd_avx(_x);

				int64_t correctedTime = i - flgDelay[flg] * 2;
				if ((correctedTime >= 0) && (correctedTime < outputSample.size()))
					outputSample[correctedTime] -= outX;
			}

			ticks++;
		}
		
#ifdef _DEBUG_INTERVAL_HISTOGRAM	
		FILE *histogramFile = fopen("interval_histogram.raw", "wb");
		fwrite(intervalHistogram, 1, SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int), histogramFile);
		fwrite(powHistogram, 1, SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int), histogramFile);
		free(intervalHistogram); free(powHistogram);
		fclose(histogramFile);
#endif

		// normalize output amplitude
		double max = 0.0; for (int64_t i = 0; i < outputSample.size(); i++) if (abs(outputSample[i]) > max) max = abs(outputSample[i]);
		if (max > 0.0)
		{
			for (int64_t i = 0; i < outputSample.size(); i++)
			{
				outputSample[i] /= max / (1.0 + SURFACE_OUTPUT_CLIP);
				if (abs(outputSample[i]) > 1.0) outputSample[i] /= abs(outputSample[i]);
			}
		}
	}

	double Output()
	{
		return 0.0;
	}

	bool SaveQuants(string outputPath)
	{
		FILE *outFile = fopen(outputPath.c_str(), "wb");
		if (!outFile) return false;

		uint64_t numFLGs = SURFACE_LG_NUM_FILTER_UNITS;
		fwrite(&numFLGs, 1, sizeof(uint64_t), outFile);
		uint64_t maxQuant = SURFACE_LG_SPIKE_MAX_QUANT;
		fwrite(&maxQuant, 1, sizeof(uint64_t), outFile);

		fwrite(flgQuantIntervalScale, 1, SURFACE_LG_NUM_FILTER_UNITS * SURFACE_LG_SPIKE_MAX_QUANT * sizeof(int), outFile);
		fwrite(&ticks, 1, sizeof(uint64_t), outFile);

		for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
		{
			uint64_t numIntervals = lgSpikeIntervals[flg].size();
			fwrite(&numIntervals, 1, sizeof(uint64_t), outFile);
			fwrite(&lgSpikeIntervals[flg][0], 1, sizeof(QUANT) * numIntervals, outFile);

#ifdef _DEBUG_SPIKES
			printf("lg: %i spikes: %i \n", flg, int(numIntervals));
#endif
		}

		fclose(outFile);
		return true;
	}

	size_t LoadQuants(string inputPath)
	{
		FILE *inFile = fopen(inputPath.c_str(), "rb");


		fclose(inFile);

		return 0;
	}

	void GetFLGDelay() // numerically determine the exact sample offset of peak amplitude for each filter's impulse response
	{
		memset(lgX, 0, SURFACE_LG_NUM_UNITS * sizeof(double)); memset(lgY, 0, SURFACE_LG_NUM_UNITS * sizeof(double));
		memset(flgX, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double)); memset(flgY, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));
		memset(flgPow, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));

		SlidingDFT_AVX(1.0);

		for (int i = 0; i < SURFACE_LG_FILTER_MAX_DELAY; i++)
		{
			SlidingDFT_AVX(0.0);

			for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
			{
				__m256d y = _mm256_setzero_pd();
				for (int s = 0; s < SURFACE_LG_FILTER_WIDTH; s += 4)
				{
					__m256d f = _mm256_load_pd(&lgFilter[s]);
					y = _mm256_fmadd_pd(_mm256_loadu_pd(&lgY[flg + s]), f, y);
				}
				flgY[flg] = hsum256_pd_avx(y);

				if (flgY[flg] > flgPow[flg])
				{
					flgDelay[flg] = i;
					flgPow[flg] = flgY[flg];
				}
			}
		}

		memset(lgX, 0, SURFACE_LG_NUM_UNITS * sizeof(double)); memset(lgY, 0, SURFACE_LG_NUM_UNITS * sizeof(double));
		memset(flgX, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double)); memset(flgY, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));
		memset(flgPow, 0, SURFACE_LG_NUM_FILTER_UNITS * sizeof(double));

#ifdef _DEBUG_FLG_DELAYS
		for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++) printf("%i - %i\n", int(flg), int(flgDelay[flg]));
#endif
	}

	// setup interval quantization scales per filter based on the local frequency scale
	void GetIntervalQuantScales()  // todo: could maybe fix the anomolous gain in the high frequencies by decreasing per flg gain when the q scale is forcibly expanded
	{
		for (int flg = 0; flg < SURFACE_LG_NUM_FILTER_UNITS; flg++)
		{
			for (int q = 0; q < SURFACE_LG_SPIKE_MAX_QUANT; q++)
			{
				double fq = lgQ[flg + SURFACE_LG_FILTER_WIDTH / 2 + SURFACE_LG_SPIKE_MAX_QUANT / 2 - q - 1];
				int qq = int(SURFACE_LG_SPIKE_QUANT_I_SCALE / fq + 1.0);
				flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q] = qq;

				if (q > 0) // forcibly expand quantization scale
				{
					if (qq < flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q - 1])
					{
						flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q] = flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q - 1] + 1;
					}
					else if (qq == flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q - 1])
					{
						bool inc = false;
						for (int q2 = 0; q2 < q; q2++)
						{
							if (flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q2] > 1)
								flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q2]--;
							else
							{
								inc = true;
								break;
							}
						}

						flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q] += inc;
					}
				}
			}

#ifdef _DEBUG_QUANT_INTERVALS
			for (int q = 0; q < SURFACE_LG_SPIKE_MAX_QUANT; q++) printf("%i - %i\n", flg, flgQuantIntervalScale[flg * SURFACE_LG_SPIKE_MAX_QUANT + q]);
#endif
		}
	}
};
*/