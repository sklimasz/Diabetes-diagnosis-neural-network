import pandas as pd
from funcs import *
from snb_network import Network

def main():
    ## training and testing data
    pima_tr = pd.read_csv('pima_tr.csv',delimiter=' ')
    pima_te = pd.read_csv('pima_te.csv',delimiter=' ')
    pima_merged = pd.concat([pima_tr,pima_te],ignore_index=True) 

    ## normalize data to (0,1)
    dummy = normalize(pima_merged)
    for header,value in zip(pima_merged.columns.values,dummy):
            pima_merged[header]=value

    pima_tr = pima_merged.iloc[:len(pima_tr),:]
    pima_te = pima_merged.iloc[len(pima_tr):,:]
        
    ## network with 7 input neurons, 1 hidden layer with 7 hidden neurons and 1 output neuron
    ## so we have a binary classificator
    Network_training = Network(numInputs=7, numHidden=7)

    ## random starting weights in (-0.1, 0.1) interval (uniform)
    Network_training.randomizeWeights(0.05)

    ## feed data to the network
    Network_training.feedData(pima_tr)

    ## number of iterations ( full dataset trainings)
    epochs = 1000

    results = {}
    print("---")
    for epoch in range(epochs):
        if epoch % (epochs/10) == 0: ## changing X in epochs/X sets the number of partial results
            print(f"{100*epoch/epochs}% training complete")
            print("Current results from training set : ")
            partial_results = Network_training.checkAccuracy() ## check current accuracy etc.
            results[epoch] = partial_results

        Network_training.oneEpoch()

    print("Final results from training set : ")
    final_results = Network_training.checkAccuracy() ## final results from training data
    results[epochs] = final_results
    weights = Network_training.getWeights() ## getting final weights

    ## new testing network, the neuron configuration has to be the same
    ## set its weights as the testing network final weights
    Network_testing = Network(numInputs=7, numHidden=7)
    Network_testing.feedWeights(weights[0], weights[1])
    Network_testing.feedData(pima_te)


    print("Results from testing set :")
    Network_testing.checkAccuracy()

    ## seaborn plot
    visualize_results(results)

if __name__ == "__main__":
    main()