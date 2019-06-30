import csv
import json
import math
import random
import pprint
import operator


def readFromCSV(filename):
    dataset = []
    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            dataset.append(row[1:])
    csvFile.close()
    return dataset


def writeInJSON(filename, data):
    with open(filename, 'w') as jsonFile:
        json.dump(data, jsonFile)
    jsonFile.close()


def findEntropy(values):
    total = sum(values)
    entropy = 0.0
    for val in values:
        prob = val / total
        if (prob > 0.0):
            entropy += (- prob * math.log(prob, 2))
    return entropy


def setResultLabel(resultColumn):
    idx = 0
    for key, value in resultColumn.items():
        resultLabel[key] = idx
        idx += 1


def getResultLabel():
    return resultLabel


def getMajorityVote(columns):
    return max(columns[-1].items(), key=operator.itemgetter(1))[0]


def extractInfoFormDataset(dataset):
    noOfRows = len(dataset)
    noOfCols = len(dataset[0])
    resultDataIndex = noOfCols - 1
    column = []

    column = [{} for i in range(noOfCols)]

    for data in dataset:
        for colIndex in range(noOfCols):
            if (colIndex == resultDataIndex):
                key = data[resultDataIndex]
            else:
                currentColValue = data[colIndex]
                resultColValue = data[resultDataIndex]
                key = (currentColValue, resultColValue)

            if (key in column[colIndex]):
                column[colIndex][key] = column[colIndex][key] + 1
            else:
                column[colIndex][key] = 1

    return column


def getColumnObjList(columns, resultLabel, resultLabelLen):
    columnObjList = []
    for column in columns:

        columnObj = {}
        for key, value in column.items():
            attributeValue, resLabel = key
            idx = resultLabel[resLabel]

            if (attributeValue not in columnObj):
                columnObj[attributeValue] = [0] * resultLabelLen

            columnObj[attributeValue][idx] = value

        columnObjList.append(columnObj)

    return columnObjList


def getInfoGainList(columnObjList, targetEntropy, noOfRows):
    infoGainList = []
    for columnObj in columnObjList:
        infoGain = 0.0
        for key, value in columnObj.items():
            infoGain = infoGain + ((sum(value) / float(noOfRows)) * findEntropy(value))
        infoGainList.append(targetEntropy - infoGain)

    return infoGainList


def bestSplitNode(dataset, isFirst):
    dataset = dataset[1:]
    noOfRows = len(dataset)
    columns = extractInfoFormDataset(dataset)
    if (len(columns) == 1):
        return getMajorityVote(columns)

    resultColumn = columns.pop()
    if (isFirst):
        setResultLabel(resultColumn)

    resultLabel = getResultLabel()
    resultLabelLen = len(resultLabel)

    if (resultLabelLen != len(resultColumn)):
        return list(resultColumn.keys())[0]

    values = list(resultColumn.values())
    targetEntropy = findEntropy(values)

    columnObjList = getColumnObjList(columns, resultLabel, resultLabelLen)
    infoGainList = getInfoGainList(columnObjList, targetEntropy, noOfRows)

    maxInfoGainList = max(infoGainList)
    if (maxInfoGainList <= 0.0):
        return getMajorityVote(columns)

    bestSplit = infoGainList.index(maxInfoGainList)
    columnObjList[bestSplit].keys()

    node = {}
    node[bestSplit] = columnObjList[bestSplit].keys()

    return node


def createTree(dataset, boolean):
    tree = {}
    node = bestSplitNode(dataset, boolean)
    if (isinstance(node, dict)):
        nodeKey = list(node.keys())[0]
        nodeLabel = dataset[0].pop(nodeKey)
        values = list(node.values())[0]
        tree[nodeLabel] = {}
        for val in values:
            newDataset = []
            newDataset.append(dataset[0])
            referenceLength = len(dataset[0]) + 1
            for row in dataset[1:]:
                if (referenceLength == len(row) and row[nodeKey] == val):
                    row.pop(nodeKey)
                    newDataset.append(row)

            subtree = createTree(newDataset, False)
            tree[nodeLabel][val] = subtree
        dataset[0].insert(nodeKey, nodeLabel)

        return tree
    else:
        return node


def chunks(data, size, resultIdx):
    for i in range(0, len(data), size):
        chunk = data[i:i + size]
        chunk.append(resultIdx)
        yield chunk


def resampling(dataset, noOfSamples):
    columnLabels = dataset[0]

    noOfCols = len(columnLabels) - 1
    colsPerDataset = int(math.floor(math.sqrt(noOfCols)))

    totalColumnsRequired = noOfSamples * colsPerDataset

    multiplyingFactor = 1

    if (totalColumnsRequired > noOfCols):
        multiplyingFactor = int(math.ceil(totalColumnsRequired / float(noOfCols)))

    colIdxList = []
    for i in range(multiplyingFactor):
        colIdxSubList = list(range(0, noOfCols))
        random.shuffle(colIdxSubList)
        # print(colIdxSubList)
        colIdxList = colIdxList + colIdxSubList

    datasetChunksIdx = list(chunks(colIdxList, colsPerDataset, noOfCols))

    if (len(datasetChunksIdx) > noOfSamples):
        datasetChunksIdx = datasetChunksIdx[0: noOfSamples]

    datasetChunks = x = [[] for i in range(len(datasetChunksIdx))]

    for row in dataset:
        for idx in range(len(datasetChunksIdx)):
            rowChunk = []
            for colIdx in datasetChunksIdx[idx]:
                rowChunk.append(row[colIdx])
            datasetChunks[idx].append(rowChunk)

    return datasetChunks


def growRandomForest(datasetChunks):
    forest = []
    isFirst = True
    for datasetChunk in datasetChunks:
        tree = createTree(datasetChunk, isFirst)
        forest.append(tree)
        isFirst = False

    return forest


def splitTrainingAndTestingDataSet(dataset, trainRatio):
    datasetLength = len(dataset) - 1

    columnLabels = dataset.pop(0)
    random.shuffle(dataset)

    dataset.insert(0, columnLabels)

    trainingCount = int(math.ceil(datasetLength * trainRatio))

    trainingDataset = dataset[:trainingCount]

    testingDataset = dataset[trainingCount:]
    testingDataset.insert(0, dataset[0])

    return [trainingDataset, testingDataset]


def buildTestingDatarow(columnLabels, row):
    datarow = {}
    for idx, value in enumerate(columnLabels[:-1]):
        datarow[value] = row[idx]

    return [datarow, row[-1]]


def getLabelFromTree(tree, datarow):
    previous_value = ""
    while (isinstance(tree, dict)):
        key = list(tree.keys())[0]
        value = datarow[key]
        if (value not in tree[key]):
            return previous_value

        previous_value = value
        tree = tree[key][value]

    return tree


def getLabelFromForest(forest, testingDataset):
    resultLabel = getResultLabel()
    reversed_resultLabel = dict(map(reversed, resultLabel.items()))

    predictedLabels = []
    actualLabels = []

    for row in testingDataset[1:]:
        datarow, result = buildTestingDatarow(testingDataset[0], row)

        labelsList = []
        for tree in forest:
            labelsList.append(getLabelFromTree(tree, datarow))

        labelCounts = [0] * len(resultLabel)

        for key, value in resultLabel.items():
            labelCounts[value] = labelsList.count(key)

        predictedLabels.append(reversed_resultLabel[labelCounts.index(max(labelCounts))])
        actualLabels.append(result)

    return [predictedLabels, actualLabels]


def findAccuray(predictedLabels, actualLabels):
    accurayList = []
    for idx, labels in enumerate(predictedLabels):
        accurayList.append(1 if labels == actualLabels[idx] else 0)

    print((sum(accurayList) / float(len(accurayList))) * 100)

    from sklearn.metrics import confusion_matrix
    confusionMatrix = confusion_matrix(actualLabels, predictedLabels)
    print(confusionMatrix)


# Start
resultLabel = {}
dataset = readFromCSV("phishcoop.csv")
trainingDataset, testingDataset = splitTrainingAndTestingDataSet(dataset, 0.75)

datasetChunks = resampling(trainingDataset, 100)

forest = growRandomForest(datasetChunks)

writeInJSON("random-forest.json", forest)

predictedLabels, actualLabels = getLabelFromForest(forest, testingDataset)

findAccuray(predictedLabels, actualLabels)