import math
import random
import time
import numpy as np
import scipy as sc
import qutip as qt
import matplotlib.pyplot as plt
def partialTraceRem(obj, rem):
    #prepare keep list
    rem.sort(reverse=True)
    keep = list(range(len(obj.dims[0])))
    for x in rem:
        keep.pop(x)
    res = obj;
    #return partial trace:
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;

def partialTraceKeep(obj, keep):
    #return partial trace:
    res = obj;
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;

def swappedOp(obj, i, j):
    if i==j: return obj
    numberOfQubits = len(obj.dims[0])
    permute = list(range(numberOfQubits))
    permute[i], permute[j] = permute[j], permute[i]
    return obj.permute(permute)

def tensoredId(N):
    #Make Identity matrix
    res = qt.qeye(2**N)
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def tensoredQubit0(N):
    #Make Qubit matrix
    res = qt.fock(2**N).proj() #For some reason ran faster than fock_dm(2**N) in tests
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def unitariesCopy(unitaries):
    newUnitaries = []
    for layer in unitaries:
        newLayer = []
        for unitary in layer:
            newLayer.append(unitary.copy())
        newUnitaries.append(newLayer)
    return newUnitaries

def randomQubitUnitary(numQubits):
    dim = 2**numQubits
    #Make unitary matrix
    res = sc.random.normal(size=(dim,dim)) + 1j * sc.random.normal(size=(dim,dim))
    res = sc.linalg.orth(res)
    res = qt.Qobj(res)
    #Make dims list
    dims = [2 for i in range(numQubits)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

# Generating random states for the training data
def randomQubitState(numQubits):
    sq = 2 ** numQubits
    res = qt.rand_dm_hs(sq)
    #Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [2 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    #Return
    return res

#Generation of training data for the bit flip with the operation elements for the bit flip as an input
def TrainingData(listOfOperationElements, N):
    numQubits = len(listOfOperationElements[0].dims[0])
    #print(numQubits)
    trainingData = []
    for i in range(N):
        t = randomQubitState(numQubits)
        utu = 0
        for e in listOfOperationElements:
            utu =  utu + e * t * e.dag()
        trainingData.append([t,utu])

    return trainingData

#The list of operation elements for the bit flip. This list changes depending on the channel.
#For clearer separation and the ability to simultaneously run simulations, all the simulations were put into different files.
bitFlipList = [math.sqrt(0.7) * tensoredId(1), math.sqrt(1-0.7) * qt.sigmax()]



def randomNetwork(qnnArch, numTrainingPairs):
    assert qnnArch[0] == qnnArch[-1], "Not a valid QNN-Architecture."

    # Create the targeted network unitary and corresponding training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkTrainingData = TrainingData(bitFlipList, numTrainingPairs)

    # Create the initial random perceptron unitaries for the network
    networkUnitaries = [[]]
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l - 1]
        numOutputQubits = qnnArch[l]

        networkUnitaries.append([])
        for j in range(numOutputQubits):
            unitary = randomQubitUnitary(numInputQubits + 1)
            if numOutputQubits - 1 != 0:
                unitary = qt.tensor(randomQubitUnitary(numInputQubits + 1), tensoredId(numOutputQubits - 1))
                unitary = swappedOp(unitary, numInputQubits, numInputQubits + j)
            networkUnitaries[l].append(unitary)

    # Return
    return (qnnArch, networkUnitaries, networkTrainingData, networkUnitary)

#The cost function with the schatten-2-norm according to the thesis.
def costFunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        costSum += (((outputStates[i] - trainingData[i][1]) * (outputStates[i] - trainingData[i][1]).dag())).tr()
    return costSum/len(trainingData)


def makeLayerChannel(qnnArch, unitaries, l, inputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Tensor input state
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))

    #Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    #Multiply and tensor out input state
    return partialTraceRem(layerUni * state * layerUni.dag(), list(range(numInputQubits)))


def makeAdjointLayerChannel(qnnArch, unitaries, l, outputState):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Prepare needed states
    inputId = tensoredId(numInputQubits)
    state1 = qt.tensor(inputId, tensoredQubit0(numOutputQubits))
    state2 = qt.tensor(inputId, outputState)

    # Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    # Multiply and tensor out output state
    #
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)))

def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x,storedStates): #new dependency on storedStates
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Calculate sigma state
    state =trainingData[x][1] - storedStates[x][-1] # state changed !!!! hier getauscht
    for i in range(len(qnnArch) - 1, l, -1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    # Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)


    # Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j + 1, numOutputQubits):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni.dag() * state * productUni

def feedforward(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0]
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)

    return storedStates


def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x,storedStates)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(trainingData))) * summ
    erg = summ.expm()
    return erg

def updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Tensor input state
    state = qt.tensor(storedStates[x][l - 1], tensoredQubit0(numOutputQubits))


    # Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j + 1):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni * state * productUni.dag()

def makeUpdateMatrixTensored(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    res = makeUpdateMatrix(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j)
    if numOutputQubits - 1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits - 1))
    return swappedOp(res, numInputQubits, numInputQubits + j)


def qnnTraining(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunction(trainingData, outputStates)]]

    # Optional
    #runtime = time()

    # Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0: print("In training round " + str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        # Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l - 1]
            numOutputQubits = qnnArch[l]

            # Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (makeUpdateMatrixTensored(qnnArch, currentUnitaries, trainingData, storedStates, lda, ep, l,j) * currentUnitaries[l][j])

        ### FEEDFORWARD
        # Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

        # Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])

        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))

    # Return
    return [plotlist, currentUnitaries]

# Calculations for different learning rates with optional plotting.
'''
network121 = randomNetwork([1,2,1], 10)
plotlist121a = qnnTraining(network121[0], network121[1], network121[2], 0.5, 0.1, 500)[0]
plotlist121b = qnnTraining(network121[0], network121[1], network121[2], 1, 0.1, 500)[0]
plotlist121c = qnnTraining(network121[0], network121[1], network121[2], 2, 0.1, 500)[0]
plotlist121d = qnnTraining(network121[0], network121[1], network121[2], 0.1, 0.1, 500)[0]

plotlist121 = [plotlist121a[0],plotlist121a[1],plotlist121b[1],plotlist121c[1],plotlist121d[1]]

columns = "N,a,b,c,d"
np.savetxt("bitflip 1-2-1 network lda05-1-2-10 eps01.csv",
           np.transpose(plotlist121),
           header = columns,
           delimiter =", ",
           fmt ='% s')
'''
# optional plotting
'''
plt.plot(plotlist121[0], plotlist121[1], label = 'Learning rate = 2')
plt.plot(plotlist121[0], plotlist121[2], label = 'Learning rate = 1')
plt.plot(plotlist121[0], plotlist121[3], label = 'Learning rate = 0,5')
plt.plot(plotlist121[0], plotlist121[4], label = 'Learning rate = 10')
plt.xlabel("s")
plt.ylabel("Cost[s]")
plt.legend()
plt.show()
'''
# Caclulations for different architectures.
'''
network121 = randomNetwork([1,2,1], 10)
network11 = randomNetwork([1,1], 10)
network131 = randomNetwork([1,3,1], 10)
network1221 = randomNetwork([1,2,2,1], 10)

pla = qnnTraining(network121[0],network121[1],network121[2],0.5,0.1,500)[0]
print(len(pla[0]))
plb = qnnTraining(network11[0],network11[1],network11[2],0.5,0.1,500)[0]
plc = qnnTraining(network131[0],network131[1],network131[2],0.5,0.1,500)[0]
print(len(pla[1]))
pld = qnnTraining(network1221[0],network1221[1],network1221[2],0.5,0.1,500)[0]

plotlist = [pla[0],pla[1],plb[1],plc[1],pld[1]]

columns = "N,a,b,c,d"
np.savetxt("biflipdifferentnetworks121o11o131o1221.csv",
           np.transpose(plotlist),
           delimiter=", ",
           fmt = '% s')
'''
#Optional Plotting
'''
plt.plot(plotlist[0], plotlist[1], label = '121')
plt.plot(plotlist[0], plotlist[2], label = '11')
plt.plot(plotlist[0], plotlist[3], label = '131')
plt.plot(plotlist[0], plotlist[4], label = '1221')
plt.xlabel("s")
plt.ylabel("Cost[s]")
plt.legend()
plt.show()
'''

# Created for more convenience when testing different parameters.
def qnnTrainingAvg(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert = 0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunction(trainingData, outputStates)]]

    # Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0: print("In training round " + str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        # Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l - 1]
            numOutputQubits = qnnArch[l]

            # Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (makeUpdateMatrixTensored(qnnArch, currentUnitaries, trainingData, storedStates, lda, ep, l,j) * currentUnitaries[l][j])

        ### FEEDFORWARD
        # Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

        # Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])

        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))

    # Return
    return [currentUnitaries, plotlist]


# Tests the trained networks on the test data
def testaverage(numberOfRunsToAverage,maxPairs,untrainedTestData):
    testStates = TrainingData(bitFlipList,untrainedTestData)#List with test data, not used for training
    outputList = [[],[],[]]
    for i in range(1, maxPairs + 1):
        avgSum = 0
        cost = 0
        for j in range(numberOfRunsToAverage):
            network = randomNetwork([1, 2, 1], i)
            trainedNetworkData = qnnTrainingAvg(network[0], network[1], network[2], 2, 0.1, 100)#trained network returns unitaries of the traines network
            trainedNetwork = trainedNetworkData[0]
            outputsWithTestData = feedforward(network[0], trainedNetwork, testStates)
            outputStates = []
            cost = cost + trainedNetworkData[1][1][-1]
            for k in range(len(outputsWithTestData)):
                outputStates.append(outputsWithTestData[k][-1])#loss function of the last value
            avgSum = avgSum + costFunction(testStates, outputStates)
            print(j ,",", i)
        avg = avgSum/numberOfRunsToAverage
        cost = cost/numberOfRunsToAverage
        outputList[0].append(i)
        outputList[1].append(avg)
        outputList[2].append(cost)
    return outputList


network121 = randomNetwork([1,2,1],10)

# Final calculations for the generalisation behaviour, parte where parameters are chosen.
'''
finallist = testaverage(20,10,10)
print(finallist[0])
print(finallist[1])
print(finallist[2])
plt.plot(finallist[0], finallist[1], 'o')
plt.plot(finallist[0],finallist[2],'o')
plt.xlabel("No of training pairs")
plt.ylabel("Average Cost for 10 training epochs")
plt.legend(loc="upper left")
plt.show()

np.savetxt("bitflip 1-2-1 10training pairs 20avg 10trainingdata.csv",
           finallist,
           delimiter =", ",
           fmt ='% s')
testMatrix = TrainingData(bitFlipList,1)
print(testMatrix)
trainedNetwork = qnnTrainingAvg(network121[0], network121[1], network121[2], 0.5, 0.1, 500)
print(trainedNetwork)
storedState = [feedforward(network121[0], trainedNetwork[0], testMatrix)[-1][-1]]

print(storedState)
print(costFunction(testMatrix,storedState))
'''