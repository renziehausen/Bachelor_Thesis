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

def randomQubitState(numQubits):
    #
    sq = 2 ** numQubits
    res = qt.rand_dm_hs(sq)
    #Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [2 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    #Return
    return res

#def randomTrainingData(unitary, N):
#    numQubits = len(unitary.dims[0])
#    trainingData=[]
#    #Create training data pairs
#    for i in range(N):
#        t = randomQubitState(numQubits)
#        ut = unitary*t*unitary.dag()
#        trainingData.append([t,ut])
#    #Return
#   return trainingData

def TrainingData(input_dim, trace_out_list, N):
    numQubits = input_dim
    #print(numQubits)
    trainingData = []
    for i in range(N):
        t = randomQubitState(numQubits)
        utu = t.ptrace(trace_out_list)

        trainingData.append([t,utu])

    return trainingData

bitFlipList = [math.sqrt(0.7) * tensoredId(1), math.sqrt(1-0.7) * qt.sigmax()]
#print(bitFlipList[0]+bitFlipList[1])

#print(TrainingData(bitFlipList,1))


def randomNetwork(qnnArch, numTrainingPairs):


    # Create the targeted network unitary and corresponding training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkTrainingData = TrainingData(3,[2], numTrainingPairs)

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

    #print(partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits))))
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)))


def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x,storedStates): #new dependency on storedStates
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
    jj = complex(0, 1)

    # Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        #print((firstPart*jj).isherm)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x,storedStates)
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



def testaverage(numberOfRunsToAverage,maxPairs,untrainedTestData):
    testStates = TrainingData(3,[2],untrainedTestData)#Liste mit nicht trainierten Datenpunkten
    outputList = [[],[],[]]
    for i in range(1, maxPairs + 1):
        avgSum = 0
        cost = 0
        for j in range(numberOfRunsToAverage):
            network = randomNetwork([3, 2, 1], i)
            trainedNetworkData = qnnTrainingAvg(network[0], network[1], network[2], 0.5, 0.1, 100)#trained network returns unitaries of the traines network
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

'''
network121 = randomNetwork([3,2,1], 10)
plotlist121 = qnnTraining(network121[0], network121[1], network121[2], 0.5, 0.1, 500)[0]

for i in range(len(plotlist121[1])):
    if plotlist121[1][i] >= 0.95:
        print("Exceeds cost of 0.95 at training step "+str(i))
        break


np.savetxt("ptrace 322 lda05 eps01.csv",
           plotlist121,
           delimiter =", ",
           fmt ='% s')

plt.plot(plotlist121[0], plotlist121[1])
plt.xlabel("s")
plt.ylabel("Cost[s]")
plt.show()
'''
'''
### Testing different learning rates
network121lea = randomNetwork([3,2,1], 10)
plotlist121a = qnnTraining(network121lea[0], network121lea[1], network121lea[2], 0.5, 0.1, 500)[0]
plotlist121b = qnnTraining(network121lea[0], network121lea[1], network121lea[2], 1, 0.1, 500)[0]
plotlist121c = qnnTraining(network121lea[0], network121lea[1], network121lea[2], 2, 0.1, 500)[0]

plotlist121 = [plotlist121a[0],plotlist121a[1],plotlist121b[1],plotlist121c[1]]

for i in range(len(plotlist121[1])):
    if plotlist121[1][i] >= 0.95:
        print("Exceeds cost of 0.95 at training step "+str(i))
        break

columns = "N,a,b,c"
np.savetxt("ptrace 3-2-1 network lda05-1-2 eps01.csv",

           np.transpose(plotlist121),
           header = columns,
           delimiter =", ",
           fmt ='% s')

plt.plot(plotlist121[0], plotlist121[1], label = '2')
plt.plot(plotlist121[0], plotlist121[2], label = '1')
plt.plot(plotlist121[0], plotlist121[3], label = '0,5')
plt.xlabel("s")
plt.ylabel("Cost[s]")
plt.legend()
plt.show()
'''
### Testing different architecetures

network121 = randomNetwork([3,2,1], 10)
network11 = randomNetwork([3,1], 10)
network131 = randomNetwork([3,3,1], 10)
network1221 = randomNetwork([3,2,2,1], 10)

pla = qnnTraining(network121[0],network121[1],network121[2],0.5,0.1,500)[0]
plb = qnnTraining(network11[0],network11[1],network11[2],0.5,0.1,500)[0]
plc = qnnTraining(network131[0],network131[1],network131[2],0.5,0.1,500)[0]
pld = qnnTraining(network1221[0],network1221[1],network1221[2],0.5,0.1,500)[0]

plotlist = [pla[0],pla[1],plb[1],plc[1],pld[1]]

columns = "N,a,b,c,d"
np.savetxt("ptracedifferentnetworks121o11o131o1221.csv",
           np.transpose(plotlist),
           delimiter=", ",
           fmt = '% s')

plt.plot(plotlist[0], plotlist[1], label = '121')
plt.plot(plotlist[0], plotlist[2], label = '11')
plt.plot(plotlist[0], plotlist[3], label = '131')
plt.plot(plotlist[0], plotlist[4], label = '1221')
plt.xlabel("s")
plt.ylabel("Cost[s]")
plt.legend()
plt.show()

### Testing Generalisation
'''
network121gen = randomNetwork([3,2,1],10)

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

np.savetxt("ptrace 1-2-1 10training pairs 20avg 10trainingdata.csv",
           np.transpose(finallist),
           delimiter =", ",
           fmt ='% s')
'''
