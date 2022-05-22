import math
import random
import time
import numpy as np
import scipy as sc
import qutip as qt
import matplotlib.pyplot as plt
import tensorflow as tf

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

def randomQubitState(numQubits): #generates a random ket vector, projection is perfromed in training data generation
    dim = 2**numQubits
    #Make normalized state
    res = sc.random.normal(size=(dim,1)) + 1j * sc.random.normal(size=(dim,1))
    res = (1/sc.linalg.norm(res)) * res
    res = qt.Qobj(res)
    #Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [1 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    #Return
    return res

def randomDensityOperator(numQubits):
    sq = 2 ** numQubits
    res = qt.rand_dm_hs(sq)
    # Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [2 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    # Return
    return res

# for pure states
def randomTrainingData(unitary, N):
    numQubits = len(unitary.dims[0])
    trainingData=[]
    #Create training data pairs
    for i in range(N):
        t = randomQubitState(numQubits)
        ut = unitary*t
        trainingData.append([t,ut])
    #Return
    return trainingData
#for mixed states
def randomTrainingDataMixed(unitary, N):
    numQubits = len(unitary.dims[0])
    trainingData=[]
    #Create training data pairs
    for i in range(N):
        t = randomQubitState(numQubits).proj()
        utu = unitary*t*unitary.dag()
        trainingData.append([t,utu])
    #Return
    return trainingData

def randomTrainingDataMixedOnMixed(unitary, N):
    numQubits = len(unitary.dims[0])
    trainingData = []
    for i in range(N):
        t = randomDensityOperator(numQubits)
        utu = unitary*t*unitary.dag()
        trainingData.append([t,utu])
    #Return
    return trainingData

def randomNetwork(qnnArch, numTrainingPairs,unitary):
    assert qnnArch[0] == qnnArch[-1], "Not a valid QNN-Architecture."

    # Create the targeted network unitary and corresponding training data
    #networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkUnitary = unitary
    networkTrainingData = randomTrainingData(networkUnitary, numTrainingPairs)
    networkTrainingDataMixed = randomTrainingDataMixed(networkUnitary, numTrainingPairs)
    networkTrainingDataMixedOnMixed = randomTrainingDataMixedOnMixed(networkUnitary, numTrainingPairs)


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
    return (qnnArch, networkUnitaries, networkTrainingData, networkTrainingDataMixed, networkTrainingDataMixedOnMixed, networkUnitary)


def costFunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        costSum += trainingData[i][1].dag() * outputStates[i] * trainingData[i][1]
    return costSum.tr()/len(trainingData)

def costFunctionMixed(trainingData, outputStates):
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
    #print(partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits))))
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)))


def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Calculate sigma state
    state = trainingData[x][1] * trainingData[x][1].dag()
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

# for mixed states

def updateMatrixSecondPartMixed(qnnArch, unitaries, trainingData, l, j, x,storedStates): #new dependency on storedStates
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Calculate sigma state
    state =trainingData[x][1] - storedStates[x][-1] # state changed !!!! hier getauscht
    #print(state.isherm)
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
        currentState = trainingData[x][0] * trainingData[x][0].dag()
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates


# for mixed states
def feedforwardMixed(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0]
        #print(trainingData[x][1].isherm)
        #print(x)
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
        jj = complex(0,1)
        #for m in storedStates:
        #   print(m[-1].isherm)

    return storedStates


def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(trainingData))) * summ
    return summ.expm()

#for mixed states

def makeUpdateMatrixMixed(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l - 1]
    jj = complex(0, 1)

    # Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPartMixed(qnnArch, unitaries, storedStates, l, j, x)
        #print((firstPart*jj).isherm)
        secondPart = updateMatrixSecondPartMixed(qnnArch, unitaries, trainingData, l, j, x,storedStates)
        #print((secondPart * jj).isherm)
        #print(secondPart.isherm)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)


        #print((mat * jj).isherm)
        #print(l, j, x)
        #print(mat)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(trainingData))) * summ
    #summi = summ * jj
    #print(summi.isherm)
    erg = summ.expm()
    #print((erg*jj).isherm)
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

#for mixed states

def updateMatrixFirstPartMixed(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Tensor input state
    state = qt.tensor(storedStates[x][l - 1], tensoredQubit0(numOutputQubits))


    # Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j + 1):
        productUni = unitaries[l][i] * productUni

    #erg = productUni * state * productUni.dag()
    #print(erg.isherm)
    #print(l,j,x)

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

    # Optional
    #runtime = time() - runtime
    #print("Trained " + str(trainingRounds) + " rounds for a " + str(qnnArch) + " network and " + str(
    #    len(trainingData)) + " training pairs in " + str(round(runtime, 2)) + " seconds")

    # Return
    return [plotlist, currentUnitaries]

#for mixed states

def qnnTrainingMixed(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforwardMixed(qnnArch, currentUnitaries, trainingData)

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunctionMixed(trainingData, outputStates)]]

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
        storedStates = feedforwardMixed(qnnArch, currentUnitaries, trainingData)

        # Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])

        plotlist[0].append(s)
        plotlist[1].append(costFunctionMixed(trainingData, outputStates))

    return [plotlist, currentUnitaries]

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

    # Optional
    #runtime = time() - runtime
    #print("Trained " + str(trainingRounds) + " rounds for a " + str(qnnArch) + " network and " + str(
    #    len(trainingData)) + " training pairs in " + str(round(runtime, 2)) + " seconds")

    # Return
    return [currentUnitaries, plotlist]

#training avg for mixed states

def qnnTrainingAvgMixed(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert = 0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforwardMixed(qnnArch, currentUnitaries, trainingData)

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunctionMixed(trainingData, outputStates)]]

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
        storedStates = feedforwardMixed(qnnArch, currentUnitaries, trainingData)

        # Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])

        plotlist[0].append(s)
        plotlist[1].append(costFunctionMixed(trainingData, outputStates))

    # Optional
    #runtime = time() - runtime
    #print("Trained " + str(trainingRounds) + " rounds for a " + str(qnnArch) + " network and " + str(
    #    len(trainingData)) + " training pairs in " + str(round(runtime, 2)) + " seconds")

    # Return
    return [currentUnitaries, plotlist]

#new cost function for the new fidelity

def finalcost(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        costSum += qt.fidelity(trainingData[i][1],outputStates[i])

    return costSum/len(trainingData)

def finalcostPure(testData, outputStates):
    costSum = 0
    for i in range(len(testData)):
        costSum += qt.fidelity(testData[i][1].proj(),outputStates[i])

    return costSum/len(testData)

'''def finalcost(trainingData, outputStates):
    costSum = 0
    testlist = []
    for i in range(len(trainingData)):
        #costSum += abs(np.trace(sc.linalg.sqrtm(trainingData[i][1]) *outputStates[i] * sc.linalg.sqrtm(trainingData[i][1])))
        #costSum += ((tf.linalg.sqrtm((tf.linalg.sqrtm(trainingData[i][1]) * outputStates[i] * tf.linalg.sqrtm(trainingData[i][1])))).tr())**2
        testlist.append(np.abs(((((trainingData[i][1]).sqrtm() * outputStates[i] * (trainingData[i][1]).sqrtm()).sqrtm())).tr()**2))


        #print(costSum , " ," , i)
    costSum = sum(testlist)
    print(testlist)
    return costSum/len(trainingData)

def finalcostPure(testData, outputStates):
    costSum = 0
    testlist = []
    for i in range(len(testData)):
        testlist.append(np.abs((((testData[i][1].proj()).sqrtm()*outputStates[i]*(testData[i][1].proj()).sqrtm()).sqrtm()).tr()**2))
        print(outputStates[i])
    costSum = sum(testlist)
    return costSum/len(testData)'''

def testaverage(numberOfRunsToAverage, maxpairs, architecture, untrainedTestData,trainingRounds):
    outputList = [[],[],[],[]] #[[No of Training Pairs],[avg cost 10 epochs, pure],[avg cost 10 epochs mixed]]
    unitary = randomQubitUnitary(architecture[-1])
    testData = randomNetwork(architecture, untrainedTestData, unitary)
    #generation of Test Data with the unitary
    testStatesPure = []
    testStatesMixed = []
    testStatesMixedonMixed = []
    for y in range(1, untrainedTestData):
        testStatesPure.append(testData[2][y])
        testStatesMixed.append(testData[3][y])
        testStatesMixedonMixed.append(testData[4][y])

    # training for each amount of training pairs
    for i in range(1, maxpairs + 1):
        avgSumPure = 0
        avgSumMixed = 0
        avgSumMixedonMixed = 0
        # averaging over several training runs
        for j in range(numberOfRunsToAverage):
            network = randomNetwork(architecture,i, unitary) #new training data generated for each averaging run

            trainedNetworkDataPure = qnnTrainingAvg(network[0], network[1], network[2], 2, 0.1, trainingRounds)
            trainedNetworkDataMixed = qnnTrainingAvgMixed(network[0], network[1], network[3], 2, 0.1, trainingRounds)
            trainedNetworkDataMixedOnMixed = qnnTrainingAvgMixed(network[0],network[1],network[4],2,0.1,trainingRounds)
            trainedNetworkPure = trainedNetworkDataPure[0]
            trainedNetworkMixed = trainedNetworkDataMixed[0]
            trainedNetworkMixedonMixed = trainedNetworkDataMixedOnMixed[0]
            outputsWithTestDataPure = feedforward(network[0], trainedNetworkPure, testStatesPure)
            outputsWithTestDataMixed = feedforwardMixed(network[0],trainedNetworkMixed, testStatesMixed)
            outputsWithTestDataMixedonMixed = feedforwardMixed(network[0],trainedNetworkMixedonMixed,testStatesMixedonMixed)
            outputStatesPure = []
            outputStatesMixed = []
            outputStatesMixedonMixed = []
            for k in range(len(outputsWithTestDataPure)):
                outputStatesPure.append(outputsWithTestDataPure[k][-1])
            avgSumPure = avgSumPure + finalcostPure(testStatesPure, outputStatesPure)

            for n in range(len(outputsWithTestDataMixed)):
                outputStatesMixed.append(outputsWithTestDataMixed[n][-1])
            avgSumMixed = avgSumMixed + finalcost(testStatesMixed, outputStatesMixed)
            print(j,",",i)

            for m in range(len(outputsWithTestDataMixedonMixed)):
                outputStatesMixedonMixed.append(outputsWithTestDataMixedonMixed[m][-1])
            avgSumMixedonMixed = avgSumMixedonMixed + finalcost(testStatesMixedonMixed, outputStatesMixedonMixed)

        avgPure = avgSumPure/numberOfRunsToAverage
        avgMixed = avgSumMixed/numberOfRunsToAverage
        avgMixedonMixed = avgSumMixedonMixed/numberOfRunsToAverage
        outputList[0].append(i)
        outputList[1].append(avgPure)
        outputList[2].append(avgMixed)
        outputList[3].append(avgMixedonMixed)
    return outputList

finallist = testaverage(10,10,[1,2,1],10,100)
print(finallist)

plt.plot(finallist[0],finallist[1])
plt.plot(finallist[0],finallist[2])
plt.plot(finallist[0],finallist[3])

plt.xlabel("No of training pairs")
plt.ylabel("Average Cost for 10 training epochs")
plt.legend(loc="upper left")
plt.show()

np.savetxt("Purevsmixed.csv",
           np.transpose(finallist),
           delimiter =", ",
           fmt ='% s')

'''unitary = randomQubitUnitary(2)
network121 = randomNetwork([2,3,2], 10, unitary)
plotlist121 = qnnTrainingMixed(network121[0], network121[1], network121[3], 1, 0.1, 500)[0]

for i in range(len(plotlist121[1])):
    if plotlist121[1][i] >= 0.95:
        print("Exceeds cost of 0.95 at training step "+str(i))
        break

plt.plot(plotlist121[0], plotlist121[1])
plt.xlabel("s")
plt.ylabel("Cost[s]")
plt.show()'''

