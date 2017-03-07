import numpy as np

neurons = []
results = []

firstLayerConnections = 1

def newNeurons(conections, layer, makeRandom = True):
	neurons[layer].append([])
	for x in range(0,conections):
		if makeRandom:
			neurons[layer][len(neurons[layer]) - 1].append(np.random.random())
		else:
			neurons[layer][len(neurons[layer]) - 1].append(1)

def newLayer(neuronNum):
	neurons.append([])
	if len(neurons) == 1:
		for x in range(0, neuronNum):
			newNeurons(firstLayerConnections, 0, False)
	else:
		for x in range(0,neuronNum):
			newNeurons(len(neurons[len(neurons) - 2]),len(neurons) - 1)

def logistic(inval):
	return 1/(1+np.exp(-inval))
def derLogistic(inval):
	return logistic(inval)*(1+logistic(inval))

def resultsSetup():
	for x in range(0,len(neurons)):
		results.append([])
		for y in range(0,len(neurons[x])):
			results[x].append([0])

def test(inList):
	results[0] = inList
	tempRes = 0
	for x in range(1,len(results)):
		for y in range(0,len(results[x])):
			for z in range(0, len(neurons[x][y])):
				tempRes += results[x-1][z] * neurons[x][y][z]
			results[x][y] = logistic(tempRes)
			tempRes = 0

def weightChange(a,trueResult,finalResult,currentInput,currentWeight):
	return a*(trueResult-finalResult)*derLogistic(currentInput*currentWeight)*currentInput

def train(inList, trueResult, trainBase):
	test(inList)
	for x in range(1, len(neurons)):
		for y in range(0, len(neurons[x])):
			for z in range(0, len(neurons[x][y])):
				for i in range(0, len(results[-1])):
					neurons[x][y][z] += weightChange(0.01,trueResult[i],results[-1][trainBase],results[x-1][z],neurons[x][y][z])

def listTrain(inList, inRes):
	for x in range(0,len(results[-1])):
		for y in range(0,len(inList)):
			train(inList[y],inRes[y],x)

#Setup Layers
newLayer(4)
newLayer(1)
#Setup Results Very Important, do not remove!!!!!!!
resultsSetup()

#training set
trainList = [[0,1,0,1],[0,0,1,0],[0,1,1,1],[1,1,0,0],[1,0,1,1],[1,1,0,0]]
trainRes = [[0],[0],[0],[1],[1],[1]]

#print random neuron values
print(neurons)

#train
for x in range(0,10000):
	listTrain(trainList,trainRes)

#print trained neuron values
print(neurons)

#test other values
test([1,1,1,1])
print(results[-1])
test([0,1,0,1])
print(results[-1])








