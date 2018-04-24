import dlc_bci as bci
import numpy as np 
from scipy import signal
import scipy
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt

#uploads the data-sets have been downscaled to a 100Hz sampling rate
def import100HzData():
    train_input , train_target = bci.load(root = './data_bci_100Hz')
    print(str(type(train_input)), train_input.size()) 
    print(str(type(train_target)), train_target.size())
    test_input , test_target = bci.load(root = './data_bci_100Hz', train = False)
    print(str(type(test_input)), test_input.size()) 
    print(str(type(test_target)), test_target.size())
    
    return train_input, train_target, test_input, test_target

#uploads the data-sets have been sampled at a 1000Hz sampling rate (original BCI data)
def import1000HzData():
    train_input , train_target = bci.load(root = './data_bci_1000Hz', one_khz = True)
    print(str(type(train_input)), train_input.size()) 
    print(str(type(train_target)), train_target.size())
    test_input , test_target = bci.load(root = './data_bci_1000Hz', train = False, one_khz = True)
    print(str(type(test_input)), test_input.size()) 
    print(str(type(test_target)), test_target.size())
    
    return train_input, train_target, test_input, test_target


# We only keep the signal of maximal amplitude (= the electrode the should be localed 
# the closed to the neurone that fired and so best to measure)
def maxSignalFeatures(inputData):
    numberSamples = (np.array(inputData[:, 0, 0])).size
    numberElectrodes = (np.array(inputData[0, :, 0])).size
    numberTimePoints = (np.array(inputData[0, 0, :])).size

    extractedFeatures = np.zeros((numberSamples, numberTimePoints))
    current_max = -1 

    #needs to be computationnally optimized by using the operations shown in the exercises 
    for i in range (0, numberSamples): 
        for j in range (0, numberElectrodes): 
            signal = np.array(inputData[i, j, :])
            signal_max=np.max(signal)
            if signal_max > current_max: 
                bestElectrode = j 
                current_max = signal_max
                signal_to_use = signal

        extractedFeatures[i, :] =  signal_to_use
        
    return extractedFeatures

def meanSignalFeatures(inputData): 
    numberSamples = (np.array(inputData[:, 0, 0])).size
    numberElectrodes = (np.array(inputData[0, :, 0])).size
    numberTimePoints = (np.array(inputData[0, 0, :])).size

    extractedFeatures = np.zeros((numberSamples, numberTimePoints))

    #needs to be computationnally optimized by using the operations shown in the exercises 
    for i in range (0, numberSamples): 
        for j in range (0, numberTimePoints): 
            signal = np.array(inputData[i, :, j])
            extractedFeatures[i, j]=np.mean(signal)
    return extractedFeatures

def normalizedSignalFeatures(inputData, time): 
    numberSamples = (np.array(inputData[:, 0, 0])).size
    numberElectrodes = (np.array(inputData[0, :, 0])).size
    numberTimePoints = (np.array(inputData[0, 0, :])).size
    numberExtractedMaximaPerPatient = np.zeros(numberSamples)
    numberExtractedMinimaPerPatient = np.zeros(numberSamples)
    extractedFeatures = np.zeros((numberSamples, 21))

    #needs to be computationnally optimized by using the operations shown in the exercises 
    for i in range (0, numberSamples): 
        relmaxValue = np.zeros((0, 5))
        relmaxTime = np.zeros((0, 5))
        relminValue = np.zeros((0, 5))
        relminTime = np.zeros((0, 5))
        numberExtractedMaxima = np.zeros(numberElectrodes)
        numberExtractedMinima = np.zeros(numberElectrodes)

        for j in range (0, numberElectrodes): 
            signal = np.array(inputData[i, j, :])        
            data = np.array(signal)

            #f[i,:], welchSpectralEnergy[i, :]=signal.welch(data)

            fft=scipy.fft(data) #signal denoising 
            bp=fft[:]
            for p in range(len(bp)): 
                if p>=10:
                    bp[p]=0
            ibp=scipy.ifft(bp)

            ibp = scipy.signal.detrend(ibp) #signal detrending

            #ibp = (ibp-ibp[0])/max(max(ibp), abs(min(ibp))) #signal normalization with initial offset suprresion 
            ibp = (ibp-np.mean(ibp))/np.std(ibp) #signal normalization with initial offset suprresion 

            #Find the local maxima of the model (times of the local maxima actually)
            relmax = scipy.signal.argrelmax(ibp)
            relmin = scipy.signal.argrelmin(ibp)
            
            numberExtractedMaxima[j]=relmax[0].size
            numberExtractedMinima[j]=relmin[0].size            

            #print(relmax[0].size)
            if (relmax[0].size == 5): # !!!!! THERE ARE NOT ALWAYS 5 MAXIMA BUT HOW TO SET THE BEST VALUE ????
                relmaxTime=np.append(relmaxTime, time[relmax].reshape((1,5)), axis=0)
                relmaxValue=np.append(relmaxValue, np.real(ibp[relmax].reshape((1,5))), axis=0)
                
            if (relmin[0].size == 5): # !!!!! THERE ARE NOT ALWAYS 5 MAXIMA BUT HOW TO SET THE BEST VALUE ????
                relminTime=np.append(relminTime, time[relmin].reshape((1,5)), axis=0)
                relmaxValueMin=np.append(relmaxValue, np.real(ibp[relmin].reshape((1,5))), axis=0)

      
        #featuresTime = np.median(relmaxTime, axis=0)
        numberExtractedMaximaPerPatient[i] = np.median(numberExtractedMaxima)
        featuresTime = np.median(relmaxTime, axis=0)
        featuresAmplitude = np.median(relmaxValue, axis=0) #mean per columns 
        
        numberExtractedMinimaPerPatient[i] = np.median(numberExtractedMinima)
        featuresTimeMin = np.median(relminTime, axis=0)
        featuresAmplitudeMin = np.median(relmaxValueMin, axis=0) #mean per columns 

        if not(np.isnan(np.min(featuresTime))): 
            extractedFeatures[i, [0,1,2,3,4]] = featuresTime
        if not(np.isnan(np.min(featuresTimeMin))): 
            extractedFeatures[i, [6,7,8,9,10]] = featuresTimeMin
        if not(np.isnan(np.min(featuresAmplitude))): 
            extractedFeatures[i, [11,12,13,14,15]] = featuresAmplitude
        if not(np.isnan(np.min(featuresAmplitudeMin))): 
            extractedFeatures[i, [16,17,18,19,20]] = featuresAmplitudeMin
    
    extractedFeatures[:,5] = numberExtractedMaximaPerPatient.reshape(numberExtractedMaximaPerPatient.shape)
    print(extractedFeatures)
        
    return extractedFeatures

def normalizedSignals(inputData, time, plot, title): 
    #IMPORTANT !! needs to be computationnally optimized by using the operations shown in the exercises 
    
    plt.cla()
    plt.clf()
    plt.close()
    
    normalizedOutput = np.zeros(inputData.shape)
    numberSamples = (np.array(inputData[:, 0, 0])).size
    numberElectrodes = (np.array(inputData[0, :, 0])).size

    for i in range (0, numberSamples): 
        for j in range (0, numberElectrodes): 
            signal = np.array(inputData[i, j, :])        
            data = np.array(signal)

            fft=scipy.fft(data) #signal denoising 
            bp=fft[:]
            for p in range(len(bp)): 
                if p>=10:
                    bp[p]=0
            ibp=scipy.ifft(bp)

            ibp = scipy.signal.detrend(ibp) #signal detrending

            #ibp = (ibp-ibp[0])/max(max(ibp), abs(min(ibp))) #signal normalization with initial offset suprresion 
            ibp = (ibp-np.mean(ibp))/np.std(ibp) #signal normalization with initial offset suprresion 
            
            normalizedOutput[i,j,:] = ibp.real
            
            if plot and i % 80 == 0: 
                plt.plot(time, ibp.real)
                plt.xlabel('time (ms)')
                plt.ylabel('Voltage (µV)')
                plt.title(title) 
        
            
        if plot and i % 80 == 0: 
            plt.show()
    return normalizedOutput

def normalizedSingleSignals(inputData, time, idx,plot, title): 
    #IMPORTANT !! needs to be computationnally optimized by using the operations shown in the exercises 
    
    normalizedOutput = np.zeros(inputData.shape)
    numberSamples = (np.array(inputData[:, 0, 0])).size
    numberElectrodes = (np.array(inputData[0, :, 0])).size

    for j in range (0, numberElectrodes): 
        signal = np.array(inputData[idx, j, :])        
        data = np.array(signal)

        fft=scipy.fft(data) #signal denoising 
        bp=fft[:]
        for p in range(len(bp)): 
            if p>=10:
                bp[p]=0
        ibp=scipy.ifft(bp)

        ibp = scipy.signal.detrend(ibp) #signal detrending

        #ibp = (ibp-ibp[0])/max(max(ibp), abs(min(ibp))) #signal normalization with initial offset suprresion 
        ibp = (ibp-np.mean(ibp))/np.std(ibp) #signal normalization with initial offset suprresion 

        if plot: 
            plt.plot(time, ibp.real)
            plt.xlabel('time (ms)')
            plt.ylabel('Voltage (µV)')
            plt.title(title) 

    if plot: 
        plt.show()
    return  ibp.real

def rawDataForSingleElectrodeVisualization(train_input): 
    inputlen = np.array(train_input[0, :, 0])

    for i in range (0,inputlen.size) :
        time = np.linspace(0, 500, 50)
        data = train_input[0, i, :] #observing the samples n°1 for all time steps and all the electodes 
        data = np.array(data)
        plt.plot(time, data)
        plt.title('Simple plot of electrode n°' + str(i)) 
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.show() #enables to show all electrodes separately 

    for i in range (0,inputlen.size) :
        time = np.linspace(0, 500, 50)
        data = train_input[0, i, :]
        data = np.array(data)
        plt.plot(time, data)
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.title('Simple plot of all electrodes') 
        plt.show() #enables to show all electrodes separately 

def rawDataVisualization(train_input, idx, title):
    inputlen = np.array(train_input[0, :, 0])
    
    for i in range (0,inputlen.size) :
        time = np.linspace(0, 500, 50)
        data = train_input[idx, i, :]
        data = np.array(data)
        plt.plot(time, data)
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.title(title) 
    plt.show()  