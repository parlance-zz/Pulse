#include "lg_surface.hpp"

using namespace std;

size_t LoadRAW(vector<float> &sample, string path)
{
	FILE *sampleFile = fopen(path.c_str(), "rb");
	if (!sampleFile) return 0;

	fseek(sampleFile, 0, SEEK_END);
	size_t sampleLength = ftell(sampleFile) / sizeof(float) / SURFACE_LG_NUM_CHANNELS;
	fseek(sampleFile, 0, SEEK_SET);

	sample.resize(sampleLength * SURFACE_LG_NUM_CHANNELS);

	for (size_t s = 0; s < sampleLength; s++)
	{
		float _s[SURFACE_LG_NUM_CHANNELS]; fread(&_s, 1, SURFACE_LG_NUM_CHANNELS * sizeof(float), sampleFile);
		for (int c = 0; c < SURFACE_LG_NUM_CHANNELS; c++) sample[s * SURFACE_LG_NUM_CHANNELS + c] = float(_s[c]);
	}

	fclose(sampleFile);
	return sampleLength;
}

int main(int argc, char *argv[])
{
	if (argc == 4)
	{
		string cmd = argv[1], inputPath = argv[2], outputPath = argv[3];

		if (cmd == "-q")
		{
			vector<float> inputSample;
			if (!LoadRAW(inputSample, inputPath))
			{
				printf("Error loading raw audio from '%s'...\n\n", inputPath.c_str());
				return 1;
			}
			else
			{
				printf("Loaded raw audio from '%s'...\n", inputPath.c_str());
			}

			SURFACE surface; surface.Init();

			auto startTime = chrono::high_resolution_clock::now();

			for (int i = 0; i < inputSample.size(); i++)
			{
				surface.Input(inputSample[i]);
				if ((i % 32768) == 0) printf(".");
			}

			auto finishTime = chrono::high_resolution_clock::now(); auto durationTime = chrono::duration_cast<std::chrono::microseconds>(finishTime - startTime); int seconds = int(durationTime.count() / 1000000);
			int millisecs = int((durationTime.count() - int64_t(seconds) * 1000000) / 1000); printf("Elapsed time : %i.%i\n", seconds, millisecs);

			printf("\nSaving quants to '%s'...\n\n", outputPath.c_str());
			if (!surface.SaveQuants(outputPath))
			{
				printf("Error saving quants to '%s'...\n\n", outputPath.c_str());
				return 1;
			}

			surface.Shutdown();
			return 0;
		}
		else if (cmd == "-d")
		{
			SURFACE surface; surface.Init();
			int64_t sampleLength = surface.LoadQuants(inputPath);
			if (sampleLength == -1)
			{
				printf("Error reading quants from '%s'...\n\n", inputPath.c_str());
				return 1;
			}
			else
			{
				printf("Loaded quants from '%s'...\n", inputPath.c_str());
			}
			vector<float> outputSample(sampleLength);
			auto startTime = chrono::high_resolution_clock::now();
			
			for (int i = 0; i < sampleLength; i++)
			{
				outputSample[i] = surface.Output();
				if ((i % 32768) == 0) printf(".");
			}
			 
			auto finishTime = chrono::high_resolution_clock::now(); auto durationTime = chrono::duration_cast<std::chrono::microseconds>(finishTime - startTime); int seconds = int(durationTime.count() / 1000000);
			int millisecs = int((durationTime.count() - int64_t(seconds) * 1000000) / 1000); printf("Elapsed time : %i.%i\n", seconds, millisecs);

			printf("Saving raw to '%s'...\n\n", outputPath.c_str());
			FILE *outFile = fopen(outputPath.c_str(), "wb");
			if (outFile)
			{
				fwrite(&outputSample[0], 1, SURFACE_LG_NUM_CHANNELS * sizeof(float) * sampleLength, outFile);
				fclose(outFile);
			}
			else
			{
				printf("Error saving raw to '%s'...\n\n", outputPath.c_str());
				return 1;
			}

			surface.Shutdown();
			return 0;
		}
	}
	
	printf("Usage:\tpulse.exe -q input.raw output.q\n\tpulse.exe -d input.q output.raw\n\n");
	return 1;
}

