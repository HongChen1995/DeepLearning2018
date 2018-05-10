import dlc_bci as bci
import numpy as np 
from scipy import signal
import scipy
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
import random

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

def normalizedSignals(inputData, time, plot, title, showGraphs = False): 
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
            
            if showGraphs: 
                if plot and i % 80 == 0: 
                    plt.plot(time, ibp.real)
                    plt.xlabel('time (ms)')
                    plt.ylabel('Voltage (µV)')
                    plt.title(title) 

        if showGraphs:   
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
    inputlentime = len(np.array(train_input[0, 0, :]))

    for i in range (0,inputlen.size) :
        time = np.linspace(0, 500, inputlentime)
        data = train_input[0, i, :] #observing the samples n°1 for all time steps and all the electodes 
        data = np.array(data)
        plt.plot(time, data)
        plt.title('Simple plot of electrode n°' + str(i)) 
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.show() #enables to show all electrodes separately 

    for i in range (0,inputlen.size) :
        time = np.linspace(0, 500, inputlentime)
        data = train_input[0, i, :]
        data = np.array(data)
        plt.plot(time, data)
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.title('Simple plot of all electrodes') 
        plt.show() #enables to show all electrodes separately 

def rawDataVisualization(train_input, idx, title):
    inputlen = np.array(train_input[0, :, 0])
    inputlentime = len(np.array(train_input[0, 0, :]))
    
    for i in range (0,inputlen.size) :
        time = np.linspace(0, 500, inputlentime)
        data = train_input[idx, i, :]
        data = np.array(data)
        plt.plot(time, data)
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.title(title) 
    plt.show()  
    
def cross_validation(dataset):
    
    idxToDelete = np.random.choice(dataset, 16)
    print(idxToDelete)
    
    lengthDataSet = len(dataset[:,0,0])
    
    print(len(dataset[:,0,0]))
    print(lengthDataSet/4)
    
    x1=np.delete(dataset,range(0,int(lengthDataSet/4)),axis=0)
    x2=np.delete(dataset,range(int(lengthDataSet/4),int(2*lengthDataSet/4)),axis=0)
    x3=np.delete(dataset,range(int(2*lengthDataSet/4),int(3*lengthDataSet/4)),axis=0)
    x4=np.delete(dataset,range(int(3*lengthDataSet/4),int(lengthDataSet)),axis=0)

    return x1, x2, x3, x4

def cross_validation_labels(dataset):
    
    lengthDataSet = len(dataset)
    
    y1=np.delete(dataset,range(0,int(lengthDataSet/4)),axis=0)
    y2=np.delete(dataset,range(int(lengthDataSet/4),int(2*lengthDataSet/4)),axis=0)
    y3=np.delete(dataset,range(int(2*lengthDataSet/4),int(3*lengthDataSet/4)),axis=0)
    y4=np.delete(dataset,range(int(3*lengthDataSet/4),int(lengthDataSet)),axis=0)

    return y1, y2, y3, y4


def standardize(centered_tX):
    centered_tX[centered_tX==0] = float('nan')
    stdevtrain = np.nanstd(centered_tX, axis=0)
    centered_tX[centered_tX==float('nan')] = 0
    stdevtrain[stdevtrain == 0] = 0.00001
            #CHECK WHY IT IS HAPPENING
    standardized_tX = centered_tX / stdevtrain
    return standardized_tX, stdevtrain

def standardize_original(tX):
    # Removing bothering data and centering
    tX[tX==-999] = 0
    s_mean = np.mean(tX, axis=0)
    centered_tX = tX - s_mean
    stdtX, stdevtrain = standardize(centered_tX)

    return stdtX, stdevtrain, s_mean

def standardize_basis(tX):
    # Resetting all the data
    b_mean = np.mean(tX,axis=0)
    centered_mat = tX - b_mean
    centered_mat[tX==0] = 0
    standardized_tX, stdevtrain = standardize(centered_mat)

    return standardized_tX, stdevtrain, b_mean

def standardize_test_original(tX, training_original_mean, stdevtrain):
    tX[tX==-999] = 0
    centered_testx = tX - training_original_mean
    centered_testx[tX==-999] = 0
    standardized_testx = centered_testx / stdevtrain

    return standardized_testx

def standardized_testx_basis(tX, basis_original_mean, stdev):
    centered_mat = tX - basis_original_mean
    centered_mat[tX==0] = 0
    standardized_testmat = centered_mat / stdev

    return standardized_testmat

"""
Returns a polynomial basis formed of all the degrees and combinations.
- The first part of the code computes from degree 1 to a given degree (max d=15)
- The second part of the code, computes the second degree with combinations meanning
that is combines every feature with each other in order to make paramters like the PHI angles
more meaningful.
- Finally, The last one is the third degree basis with combinations of elements
that do not all have the same degree, and only taking the 15 first most meaningful
features for our model. 
"""
def build_poly_basis(tx):
    d = len(tx[0])
    n = len(tx)

    indices_s_deg = []
    indices_t_deg = []

    print("Creating indices for subsets of degree 2")
    for i in range (d):
        for t in range (i,d):
            indices_s_deg.append([t, i])
    indices_s_deg = np.array(indices_s_deg).T

    print("Creating indices for subsets of degree 3")
    max_t_degree = min(d-1,15)
    for i in range (max_t_degree):
        for t in range (i,max_t_degree):
            for j in range(t,max_t_degree):
                if not (i == t and i == j):
                    indices_t_deg.append([j, t, i])
    indices_t_deg = np.array(indices_t_deg).T

    degrees = range(3,11)
    degrees_number = len(degrees) + 1
    stdX_Ncols = tx.shape[1]
    indices_s_Ncols = indices_s_deg.shape[1]
    indices_t_Ncols = indices_t_deg.shape[1]

    number_of_rows = indices_s_Ncols + degrees_number * stdX_Ncols + indices_t_Ncols

    mat = np.zeros((n, number_of_rows))

    print("Computing first degree")
    # First degree
    mat[:, :stdX_Ncols] = tx

    print("Computing second degree WITH combinations")
    # Second degree gotten from indices
    mat[:,stdX_Ncols:stdX_Ncols + indices_s_Ncols] = tx[:, indices_s_deg[0]] * tx[:, indices_s_deg[1]]

    print("Computing from degree 3 to 10 WITHOUT combinations...")
    # Improve 3 to 10 degree
    for i in degrees:
        start_index = indices_s_Ncols + (i - 2) * stdX_Ncols
        end_index = start_index + stdX_Ncols
        mat[:,start_index:end_index] = tx**i

    print("Computing third degree WITH combinations...")
    # Third degree gotten from indices
    mat[:, number_of_rows - indices_t_Ncols: number_of_rows] = tx[:, indices_t_deg[0]] * tx[:, indices_t_deg[1]] * tx[:, indices_t_deg[2]]

    return mat

def noise(X, intensity): 
    # Adding white noise
    wn = np.random.randn(len(X), len(X[0, :]))
    return X + intensity*wn

def shift(X, shift): 
    return np.roll(X, shift)

def denoisedSignals(inputData): 
    #IMPORTANT !! needs to be computationnally optimized by using the operations shown in the exercises 
    
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

            #ibp = (ibp-ibp[0])/max(max(ibp), abs(min(ibp))) #signal normalization with initial offset suprresion 
            ibp = (ibp-np.mean(ibp))/np.std(ibp) #signal normalization with initial offset suprresion 
            
            normalizedOutput[i,j,:] = ibp.real
    return normalizedOutput

def preprocessing_train_100(train_input, train_target, denoize=False, addGaussianNoise=False):
    
    #denoise and normalize data (without detrending and so)
    tmp = np.array(train_input)
    tmp_target = np.array(train_target)
    
    if denoize:
        tmp = denoisedSignals(tmp) #Deletes the high frequencies 

    augmented_train_input = tmp[:,:,0::10]
    idxToDelete = random.sample(range(len(augmented_train_input[:,0,0])), 16) #takes 16 lines as a validation set
    augmented_train_input_validation = tmp[idxToDelete,:,0::10]
    augmented_train_input_validation_target = tmp_target[idxToDelete]
    augmented_train_input_train = np.delete(augmented_train_input, idxToDelete, 0)
    augmented_train_input_train_target = np.delete(train_target, idxToDelete, 0)
    
    final_augmented_train_input_train = augmented_train_input_train
    final_augmented_train_input_validation = augmented_train_input_validation
    final_augmented_train_input_train_target = augmented_train_input_train_target
    final_augmented_train_input_validation_target = augmented_train_input_validation_target

    for i in range(1, 10):
        augmented_train_input = tmp[:,:,i::10]
        #idxToDelete = random.sample(range(len(augmented_train_input[:,0,0])), 16) #takes 16 lines as a validation set
        augmented_train_input_validation = tmp[idxToDelete,:,i::10]
        augmented_train_input_validation_target = tmp_target[idxToDelete]
        augmented_train_input_train = np.delete(augmented_train_input, idxToDelete, 0)
        augmented_train_input_target = np.delete(train_target, idxToDelete, 0)
        
        final_augmented_train_input_train = np.concatenate((final_augmented_train_input_train, augmented_train_input_train))
        final_augmented_train_input_validation = np.concatenate((final_augmented_train_input_validation, augmented_train_input_validation))
        final_augmented_train_input_train_target = np.concatenate((final_augmented_train_input_train_target, augmented_train_input_target))
        final_augmented_train_input_validation_target = np.concatenate((final_augmented_train_input_validation_target, augmented_train_input_validation_target))

    if(addGaussianNoise):
        noise_tensor = np.zeros(final_augmented_train_input_train.shape)
        for i in range (final_augmented_train_input_train.shape[0]):
            noiseIntensity = 0.1*np.max(final_augmented_train_input_train[i,:,:])
            noise_tensor[i, :, :] = noise(final_augmented_train_input_train[i,:,:], noiseIntensity)
        return noise_tensor, final_augmented_train_input_validation, final_augmented_train_input_train_target, final_augmented_train_input_validation_target
    
    return final_augmented_train_input_train, final_augmented_train_input_validation, final_augmented_train_input_train_target, final_augmented_train_input_validation_target

def preprocessing_test_100(test_input, denoize = False):
    #denoise and normalize data (without detrending and so)
    tmp = np.array(test_input)
    if denoize:
        tmp = denoisedSignals(tmp)
    return tmp

def preprocessing_train_20(train_input, train_target, denoize=False, addGaussianNoise=False):
    
    #denoise and normalize data (without detrending and so)
    tmp = np.array(train_input)
    tmp_target = np.array(train_target)
    
    if denoize:
        tmp = denoisedSignals(tmp) #Deletes the high frequencies 

    augmented_train_input = tmp[:,:,0::50]
    idxToDelete = random.sample(range(len(augmented_train_input[:,0,0])), 16) #takes 16 lines as a validation set
    augmented_train_input_validation = tmp[idxToDelete,:,0::50]
    augmented_train_input_validation_target = tmp_target[idxToDelete]
    augmented_train_input_train = np.delete(augmented_train_input, idxToDelete, 0)
    augmented_train_input_train_target = np.delete(train_target, idxToDelete, 0)
    
    final_augmented_train_input_train = augmented_train_input_train
    final_augmented_train_input_validation = augmented_train_input_validation
    final_augmented_train_input_train_target = augmented_train_input_train_target
    final_augmented_train_input_validation_target = augmented_train_input_validation_target

    for i in range(1, 50):
        augmented_train_input = tmp[:,:,i::50]
        #idxToDelete = random.sample(range(len(augmented_train_input[:,0,0])), 16) #takes 16 lines as a validation set
        augmented_train_input_validation = tmp[idxToDelete,:,i::50]
        augmented_train_input_validation_target = tmp_target[idxToDelete]
        augmented_train_input_train = np.delete(augmented_train_input, idxToDelete, 0)
        augmented_train_input_target = np.delete(train_target, idxToDelete, 0)
        
        final_augmented_train_input_train = np.concatenate((final_augmented_train_input_train, augmented_train_input_train))
        final_augmented_train_input_validation = np.concatenate((final_augmented_train_input_validation, augmented_train_input_validation))
        final_augmented_train_input_train_target = np.concatenate((final_augmented_train_input_train_target, augmented_train_input_target))
        final_augmented_train_input_validation_target = np.concatenate((final_augmented_train_input_validation_target, augmented_train_input_validation_target))

    if(addGaussianNoise):
        noise_tensor = np.zeros(train_input.shape)
        for i in range (augmented_train_input_train.shape[0]):
            noiseIntensity = 0.1*np.max(augmented_train_input_train[i,:,:])
            noise_tensor[i , :, :] = noise(augmented_train_input_train[i,:,:], noiseIntensity)
        return noise_tensor, final_augmented_train_input_validation
    
    return final_augmented_train_input_train, final_augmented_train_input_validation, final_augmented_train_input_train_target, final_augmented_train_input_validation_target

def preprocessing_test_20(test_input, test_target, denoize = False):
    #denoise and normalize data (without detrending and so)
    tmp = np.array(test_input)
    tmp_target = np.array(test_target)
    
    if denoize:
        tmp = denoisedSignals(tmp) #Deletes the high frequencies 

    final_augmented_test = tmp[:,:,0::50]
    final_augmented_target = tmp_target.copy()
    

    for i in range(1, 50):
        augmented_test_input = tmp[:,:,i::50]
        
        final_augmented_test = np.concatenate((final_augmented_test, augmented_test_input))
        final_augmented_target = np.concatenate((final_augmented_target, tmp_target))

    
    return final_augmented_test, final_augmented_target