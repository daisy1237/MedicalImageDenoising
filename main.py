from src import viewImage
from experiments import evaluateFilters,buildProbabilityMap,testSpatialNormalization,testAdaptiveFilter

if __name__ == "__main__":
    #viewImage.visualizeSample("21")
    #evaluateFilters.runExperiments()
    #buildProbabilityMap.runProbMap()
    #testSpatialNormalization.runNormaliztionTest()
    testAdaptiveFilter.runAdaptiveTest(imageId=21,sigma=20,protectionStrength=0.7)