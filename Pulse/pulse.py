# the q files produced by this version are stereo log-linear spectrograms with 256 bins

import argparse
import torch
import numpy as np
from mdct import mdct, imdct, mdst, imdst

FFT_LEN = 256
FILTER_STD = 72.
NUM_FILTERS = 256
NUM_OCTAVES = 8.0
BANDWIDTH = 0.96
CHUNK_STEP = FFT_LEN//2
Q_SCALE = 100.

print("torch cuda available? : " + str(torch.cuda.is_available()))
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def synth_log_normal_filters(filterLen, numFilters, numOctaves, bandwidth, sampleStd): 
    x = torch.linspace(0.5/filterLen, 1. - 0.5/filterLen, filterLen)

    filterQ = torch.exp2(torch.linspace(0., 1., numFilters) * numOctaves)
    filterQ = filterQ / torch.max(filterQ) * bandwidth 

    coverage = torch.zeros(filterLen)
    comb = torch.from_numpy((np.arange(filterLen) & 1).astype(np.float32) * 2. - 1.).cuda()

    filters = torch.zeros((numFilters, 1, filterLen))
    for f in range(numFilters):
        filters[f, 0] = torch.exp(-((torch.log(x / filterQ[f]) * sampleStd) ** 2)) * comb
        coverage += torch.absolute(filters[f, 0])

    return filters, coverage / torch.max(coverage), filterQ

def sample_filter_set(response, sample, filters):
    response[:] = torch.sum(sample * filters, dim=2)
    return
   
def inverse_filter_set(buffer, filters, coeff):
    buffer[0, :] = torch.sum(coeff[:,0, None] * filters[:,0,:], dim=0)
    buffer[1, :] = torch.sum(coeff[:,1, None] * filters[:,0,:], dim=0)
    return
        
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
        
    lgFilters, lgTotalResponse, lgQ = synth_log_normal_filters(FFT_LEN//2, NUM_FILTERS, NUM_OCTAVES, BANDWIDTH, FILTER_STD)
    #reconFilters, lgReconResponse, lgReconQ = synth_log_normal_filters(FFT_LEN//2, NUM_FILTERS, NUM_OCTAVES, BANDWIDTH, FILTER_STD)

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
        numOutputSamples = inputQuants.shape[0] * CHUNK_STEP + FFT_LEN//2
    if args.outputraw != '':
        outData = torch.zeros(numOutputSamples)
    if args.outputquants != '':
        numSteps = (numOutputSamples - FFT_LEN) // CHUNK_STEP + 1
        outputQuants = torch.zeros((numSteps, NUM_FILTERS, 2))
    
    mdctWindow = torch.sin(torch.pi / 2. / CHUNK_STEP * (torch.arange(FFT_LEN) + 0.5)).cuda()
    #mdctWindowClipped = (torch.arange(FFT_LEN) > (FFT_LEN//2)).float().cuda()
    if args.debugfilters:
        mdctWindow.cpu().numpy().tofile("mdctWindow.raw")

    step = 0
    lgResponse = torch.zeros((NUM_FILTERS, 2)); lgResponse2 = torch.zeros((NUM_FILTERS, 2)) 
    mdctResponse = torch.zeros((2, FFT_LEN//2))

    for i in range(0, numOutputSamples - FFT_LEN, CHUNK_STEP):
        if args.inputraw != '':
            tform_data = (inData[i:i+FFT_LEN] * mdctWindow).cpu().numpy()
            transform = mdct(tform_data) + 1j * mdst(tform_data)
            mdctResponse[0,:] = torch.from_numpy(transform.real).cuda()
            mdctResponse[1,:] = torch.from_numpy(transform.imag).cuda()
            sample_filter_set(lgResponse, mdctResponse, lgFilters)

            if args.outputquants != '':
                outputQuants[step,:] = torch.log(torch.absolute(lgResponse) * Q_SCALE + 1.) * torch.sign(lgResponse)
        
        if args.inputquants != '':
            lgResponse = inputQuants[step]
        if args.outputraw != '':
            """
            tform_data = (outData[i:i+FFT_LEN] * mdctWindowClipped).cpu().numpy()
            transform = mdct(tform_data) + 1j * mdst(tform_data)
            mdctResponse[0,:] = torch.from_numpy(transform.real).cuda()
            mdctResponse[1,:] = torch.from_numpy(transform.imag).cuda()
            sample_filter_set(lgResponse2, mdctResponse, lgFilters)

            lgAbsResponse = torch.sqrt((lgResponse[:,0] * lgResponse[:,0]) + (lgResponse[:,1] * lgResponse[:,1]))# ** (1/4)
            lgAbsResponse2 = torch.sqrt((lgResponse2[:,0] * lgResponse2[:,0]) + (lgResponse2[:,1] * lgResponse2[:,1]))# ** (1/4)
            lgResponse[:, 0] = lgResponse2[:,0] / lgAbsResponse2 * lgAbsResponse
            lgResponse[:, 1] = lgResponse2[:,1] / lgAbsResponse2 * lgAbsResponse
            """ 
            lgResponse[:] = (torch.exp(torch.absolute(outputQuants[step])) - 1.) / Q_SCALE * torch.sign(outputQuants[step])
            inverse_filter_set(mdctResponse, lgFilters, lgResponse)
            _mdctResponse = mdctResponse.cpu().numpy()
            _reconstructed = (imdct(_mdctResponse[0]) + imdst(_mdctResponse[1])).astype(np.float32)
            outData[i:i+FFT_LEN] += torch.from_numpy(_reconstructed).cuda() * mdctWindow
        
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
        #outData.cpu().numpy().tofile(oPath)
 