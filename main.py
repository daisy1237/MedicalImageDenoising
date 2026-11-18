from experiments import analyzeFinalVariability,testDriveAlignment,testDriveRegistration,tuneDriveAlpha
from src import viewImage,driveAlignment
from experiments import buildProbabilityMap,evaluateDriveProbabilityMap,evaluateDriveAdaptive

if __name__ == "__main__":
    #buildProbabilityMap.buildDriveProbabilityMaps()
    evaluateDriveAdaptive.evaluateDriveAdaptive()
    #testDriveAlignment.selectOpticDiscCenters()
    #turnDriveAlpha.tuneAlpha()