import dlc_bci as bci

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