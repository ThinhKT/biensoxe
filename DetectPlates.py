# DetectPlates.py

import cv2
import numpy as np
import math
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.3

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []             

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if showSteps == True: 
        cv2.imshow("0", imgOriginalScene)

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)     

    if showSteps == True: 
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)

    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if showSteps == True:
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene))) 

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)

        cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
        cv2.imshow("2b", imgContours)

    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    listOfListsOfMatchingCharsInScene = DetectChars.divideListOfListsOfMatchingChars(listOfListsOfMatchingCharsInScene)

    if showSteps == True: 
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene))) 

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

        cv2.imshow("3", imgContours)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                  
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)       

        if possiblePlate.imgPlate is not None:                      
            listOfPossiblePlates.append(possiblePlate)       

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  

    if showSteps == True: 
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)

    return listOfPossiblePlates

def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []              

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                   

        if showSteps == True:
            cv2.drawContours(imgContours, contours, i, SCALAR_WHITE)

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                  
            intCountOfPossibleChars = intCountOfPossibleChars + 1         
            listOfPossibleChars.append(possibleChar)    

    if showSteps == True: 
        print("\nstep 2 - len(contours) = " + str(len(contours)))  
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars)) 
        cv2.imshow("2a", imgContours)

    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()         

    DetectChars.mysort(listOfMatchingChars)  


    minX = listOfMatchingChars[0].intBoundingRectX
    maxX = listOfMatchingChars[0].intBoundingRectX
    minY = listOfMatchingChars[0].intBoundingRectY
    maxY = listOfMatchingChars[0].intBoundingRectY

    width = listOfMatchingChars[0].intBoundingRectWidth
    height = listOfMatchingChars[0].intBoundingRectHeight

    for item in listOfMatchingChars:
        if item.intBoundingRectX < minX:
            minX = item.intBoundingRectX
        if item.intBoundingRectX > maxX:
            maxX = item.intBoundingRectX
        if item.intBoundingRectY < minY:
            minY = item.intBoundingRectY
        if item.intBoundingRectY > maxY:
            maxY = item.intBoundingRectY
    maxX += width
    maxY += height

    fltPlateCenterX = (maxX + minX ) / 2
    fltPlateCenterY = (maxY + minY ) / 2

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    intPlateWidth = int((maxX - minX) * PLATE_WIDTH_PADDING_FACTOR)

    intPlateHeight = int((maxY - minY) * PLATE_HEIGHT_PADDING_FACTOR)

    fltCorrectionAngleInDeg = 0.0

    for i in range(len(listOfMatchingChars) - 1):
        fltOpposite = listOfMatchingChars[i + 1].intCenterY - listOfMatchingChars[i].intCenterY
        fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[i], listOfMatchingChars[i + 1])
        fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
        fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
        if abs(fltCorrectionAngleInDeg) <= 17.0:
            break    

    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape   

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  

    imgCropped = cv2.getRectSubPix(imgRotated, (abs(intPlateWidth), abs(intPlateHeight)), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped        

    return possiblePlate

def GetPILImage(img):
    pilImg = cv2.cvtColor(imgOriginalScene, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(pilImg)
    return pilImage















