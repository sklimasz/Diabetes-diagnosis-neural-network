import numpy as np
import pandas as pd
from funcs import *
# class representing the network
# learning using backward progatation algorithm
class Network:
    def __init__(self,numInputs,numHidden):
        self.numInputs = numInputs         
        self.numHidden = numHidden
        self.inputs = np.ones(numInputs)

        self.hiddenLayerWeights = np.zeros( (numInputs,numHidden) )
        self.hiddenLayerStimuli = np.zeros(numHidden)             
        self.hiddenLayerOutputs = np.zeros(numHidden)              


        self.outputLayerWeights = np.zeros(numHidden)               
        self.outputLayerStimuli = np.array([0])                     

        self.output = 0                                             
        self.correctOutput = 0                                      
        self.learningRate = 0.1                                    

        self.dataSet = None                                         

    def randomizeWeights(self, range):  
        ## set starting weights in (-range,range) interval, uniform distribution                             
        self.hiddenLayerWeights = np.random.uniform(-range, range, size=(self.numInputs,self.numHidden) )
        self.outputLayerWeights = np.random.uniform(-range, range, size=(self.numHidden) )

    def feedData(self, data):
        self.dataSet = data             
    
    def forwardPropagate(self):     # propagation
        self.hiddenLayerStimuli = np.matmul(self.hiddenLayerWeights.T, self.inputs) 
        self.hiddenLayerOutputs = leakyRELU(self.hiddenLayerStimuli)    # change hidden layer act. function here!            
                                                                                    
        self.outputLayerStimuli = np.matmul(self.outputLayerWeights, self.hiddenLayerOutputs)   
        self.output = sigmoid(self.outputLayerStimuli) ## change output layer act. function here!

    def changeWeights(self):    # correct the weights
        temp = np.subtract(self.correctOutput, self.output)
        outputError = np.multiply(temp, sigmoid_der(self.outputLayerStimuli)) 
                                                                            
        temp1 = np.multiply(self.outputLayerWeights,outputError)    
        hiddenLayerError = np.multiply( temp1, leakyRELU_der(self.hiddenLayerStimuli))  # change derivative of act. function here!
                                                                                        
        outputLayerWeightChange = np.multiply(self.outputLayerStimuli, outputError) * self.learningRate
        hiddenLayerWeightChange = np.matmul( hiddenLayerError.reshape((self.numHidden,1)) , self.inputs.reshape((self.numInputs,1)).T ).T  * self.learningRate

        self.hiddenLayerWeights = np.add(self.hiddenLayerWeights, hiddenLayerWeightChange)  # korekta wag neuronów w. ukrytej 
        self.outputLayerWeights = np.add(self.outputLayerWeights, outputLayerWeightChange)  # korekta wag neuronu w. wyjściowej
    
    # get weights (after training)
    def getWeights(self):
        return [self.hiddenLayerWeights, self.outputLayerWeights]

    # set weights (after training to testing network)
    def feedWeights(self, hiddenWeights, outputWeights):
        self.hiddenLayerWeights = hiddenWeights
        self.outputLayerWeights = outputWeights

    # go through one full shuffled dataset
    def oneEpoch(self):                                     
        self.dataSet = self.dataSet.sample(frac = 1)        # shuffle the data
        numberRows = self.dataSet.shape[0]
        for index in range(numberRows):
            data = self.dataSet.iloc[index].tolist()        # one row of data
            self.correctOutput = data.pop()

            self.inputs = np.array(data)                    
            self.forwardPropagate()                        
            self.changeWeights()
    
    ## check accuracy on full data set
    def checkAccuracy(self):
        false_pos = 0
        false_neg = 0
        true_pos  = 0
        true_neg  = 0
        numberRows = self.dataSet.shape[0]
        error = 0
        for index in range(numberRows):
            data = self.dataSet.iloc[index].tolist()
            self.correctOutput = data.pop()
            outcome = self.correctOutput

            self.inputs = np.array(data)
            self.forwardPropagate()
            ## no changing weights here!
            output = self.output
            if( output >= 0.5 and outcome == 1):
                true_pos += 1
            elif( output < 0.5 and outcome == 0):
                true_neg += 1
            elif(output >= 0.5 and outcome == 0):
                false_neg += 1
            elif(output < 0.5 and outcome == 1):
                false_pos += 1
            else:
                pass
            error += (self.correctOutput - output)**2
        
        meanSquared_error = round(error/numberRows, 4)  
        acc_diagnoses = round((true_pos+true_neg)/numberRows, 4)
        sensitivity = round(true_pos/(false_neg+true_pos+1),  4)
        specificity = round(true_neg/(true_neg+false_pos+1),  4)
                                                            
        print(f"Accurate diagnoses = {100*acc_diagnoses} %")     # obliczenie dokładności
        print(f"Sensitivity = {sensitivity}")                        # obliczenie czułości
        print(f"Specificity = {specificity}")
        print(f"MSE-error = {meanSquared_error}")                  # obliczenie specyficzności
        print("----")
        results = [acc_diagnoses,sensitivity,specificity,meanSquared_error]
        return results
