import argparse
import torch
import numpy as np

FFT_LEN = 16384
FILTER_STD = 8. #25.
NUM_FILTERS = 128
NUM_OCTAVES = 9.
BANDWIDTH = 0.38 #0.33
CHUNK_STEP = 8

print("torch cuda available? : " + str(torch.cuda.is_available()))
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def synth_log_normal_filters(filterLen, numFilters, numOctaves, bandwidth, sampleStd): 
    x = torch.linspace(0.5/filterLen, 1. - 0.5/filterLen, filterLen)

    filterQ = torch.exp2(torch.linspace(0., 1., numFilters) * numOctaves)
    filterQ = filterQ / torch.max(filterQ) * bandwidth 

    coverage = torch.zeros(filterLen)
    comb = torch.from_numpy((np.arange(filterLen) & 1).astype(np.float32) * 2. - 1.).cuda()

    filters = torch.zeros((numFilters, 2, filterLen))
    for f in range(numFilters):
        fourier_filter = torch.exp(-((torch.log(x / filterQ[f]) * sampleStd) ** 2)) * comb
        coverage += torch.absolute(fourier_filter)
        ifft = torch.fft.ifft(fourier_filter)
        filters[f, 0] = torch.real(ifft) 
        filters[f, 1] = torch.imag(ifft) 

    return filters, coverage / torch.max(coverage), filterQ

def sample_filter_set(response, sample, filters):
    response[:] = torch.sum(sample * filters, dim=2)
    return
   
def inverse_filter_set(buffer, offset, filters, coeff):
    buffer[offset:offset+filters.shape[2]] += torch.sum(coeff[:,:, None] * filters, dim=0)[0]
        
def dump_filters_to_file(filters, path):
    filters.cpu().numpy().tofile(path)    
    return

def load_quant_file(path):
    quants = np.fromfile(path, dtype=np.float32)
    quants = np.reshape(quants, (len(quants) // NUM_FILTERS // 2, NUM_FILTERS, 2))
    return torch.from_numpy(quants).cuda()

def save_quant_file(cQuants, path):
    cQuants.cpu().numpy().tofile(path)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and quantize a float64 raw PCM 44.1khz mono audio file')
    parser.add_argument('--inputraw', default='',
                        help='path of the audio file to quantize (optional)')                 
    parser.add_argument('--inputquants', default='',
                        help='path of the quant file to synthesize (optional)')
    parser.add_argument('--outputraw', default='',
                        help='path of the output audio file (optional)')
    parser.add_argument('--outputquants', default='',
                        help='path of the output quant file (optional)')
    parser.add_argument('--debugfilters', action='store_true',
                        help='enable to dump filter set and coverage')
                        
    args = parser.parse_args()
    
    if args.inputraw != '' and args.inputquants != '':
        print('Error: Only input audio OR quants can be specified at once')
        exit(1)
     
    if args.inputraw == '' and args.inputquants == '':
        print('Error: Input audio or quants must be specified')
        exit(1)
     
    if args.outputraw == '' and args.outputquants == '':
        print('Error: Output must be raw audio, quants, or both')
        exit(1)
        
    lgFilters, lgTotalResponse, lgQ = synth_log_normal_filters(FFT_LEN, NUM_FILTERS, NUM_OCTAVES, BANDWIDTH, FILTER_STD)

    if args.debugfilters:
        dump_filters_to_file(lgFilters, "lgFilterSet.raw")
        (lgTotalResponse / torch.max(lgTotalResponse)).cpu().numpy().tofile("lgFilterSetCoverage.raw")

    if args.inputraw != '':
        inputRawPath = args.inputraw
        print("Loading audio from " + inputRawPath + "...")
        inData = torch.from_numpy(np.fromfile(inputRawPath, dtype=np.float32)).cuda()
        numOutputSamples = len(inData)
    if args.inputquants != '':
        inputQuantPath = args.inputquants
        print("Loading quants from " + inputQuantPath + "...")
        inputQuants = load_quant_file(inputQuantPath)
        numOutputSamples = inputQuants.shape[0] + FFT_LEN
    if args.outputraw != '':
        outData = torch.zeros(numOutputSamples)
    if args.outputquants != '':
        outputQuants = torch.zeros(((numOutputSamples - FFT_LEN), NUM_FILTERS, 2))
        
    step = 0
    lgResponse = torch.zeros((NUM_FILTERS, 2))

    for i in range(0, numOutputSamples - FFT_LEN, CHUNK_STEP):
        if args.inputraw != '':
            sample_filter_set(lgResponse, inData[i:i+FFT_LEN], lgFilters)
            if args.outputquants != '':
                outputQuants[step,:] = lgResponse
        
        if args.inputquants != '':
            lgResponse = inputQuants[step]
        if args.outputraw != '':
            inverse_filter_set(outData, i, lgFilters, lgResponse)
        
        step += 1
        if step & 0xFF == 0:
            print(".", end='')

    print("")

    if args.outputquants:
        qPath = args.outputquants
        print('Saving quants to ' + qPath + '...')
        save_quant_file(outputQuants, qPath)
        
    if args.outputraw:
        oPath = args.outputraw
        print('Saving output to ' + oPath + '...')
        (outData / torch.max(outData)).cpu().numpy().tofile(oPath)
 