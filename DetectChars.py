# DetectChars.py
import os

import cv2
import numpy as np
import math
import random

import Preprocess
import PossibleChar
import DetectPlates

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.7

MAX_CHANGE_IN_WIDTH = 0.7
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 90.0

MIN_NUMBER_OF_MATCHING_CHARS = 6

RESIZED_CHAR_IMAGE_WIDTH = 12
RESIZED_CHAR_IMAGE_HEIGHT = 28

MIN_CONTOUR_AREA = 100
MAX_CHANGE_IN_LINE = 1

MAX_DISTANCE_IN_HEIGHT = 1.5


def trainingData():
    path = './data/'
    npaClassifications = []
    npaFlattenedImages = []
    for dirr in os.listdir(path):
        for file in os.listdir(path + dirr + '/'):
            temp = cv2.imread(path + dirr + '/' + file)
            gray = np.zeros((temp.shape[0], temp.shape[1], 1), np.uint8)
            gray = Preprocess.extractValue(temp)
            npaClassifications.append(ord(dirr[0]))
            npaFlattenedImages.append(np.array(gray).reshape((RESIZED_CHAR_IMAGE_HEIGHT * RESIZED_CHAR_IMAGE_WIDTH)))

    kNearest.setDefaultK(5) 
    kNearest.train(np.array(npaFlattenedImages).astype(np.float32), 0, np.array(npaClassifications).astype(np.float32))
    return True

def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates             

    for possiblePlate in listOfPossiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)    

        if DetectPlates.showSteps == True: 
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)

        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if DetectPlates.showSteps == True:
            cv2.imshow("5d", possiblePlate.imgThresh)

        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if DetectPlates.showSteps == True: 
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                    

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)

            cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
            cv2.imshow("6", imgContours)

        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if DetectPlates.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

            cv2.imshow("7", imgContours)

        if (len(listOfListsOfMatchingCharsInPlate) == 0):		

            if DetectPlates.showSteps == True:
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)

            possiblePlate.strChars = ""
            continue						

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                             
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)       
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              

        if DetectPlates.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

            cv2.imshow("8", imgContours)

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i

        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if DetectPlates.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)

            cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)

            cv2.imshow("9", imgContours)

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if DetectPlates.showSteps == True: 
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)

    if DetectPlates.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)

    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []   
    contours = []
    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                 
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):       
            listOfPossibleChars.append(possibleChar)      

    return listOfPossibleChars

def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

def findListOfListsOfMatchingChars(listOfPossibleChars):

    listOfListsOfMatchingChars = []  

    for possibleChar in listOfPossibleChars:                 
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)     
        listOfMatchingChars.append(possibleChar)    

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:    
            continue                           
        
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = []
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved) 

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:     
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)     
        break  
    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []     

    for possibleMatchingChar in listOfChars:            
        if possibleMatchingChar == possibleChar:        
            continue                          

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        if (
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar) 

    return listOfMatchingChars 

def divideListOfListsOfMatchingChars(listOfListsOfMatchingCharsInScene):
    res = []
    for i in listOfListsOfMatchingCharsInScene:
        temp = recursiveDivideListOfListsOfMatchingChars(i)
        for j in temp:
            res.append(j)
    return res

def recursiveDivideListOfListsOfMatchingChars(listsOfMatchingCharsInScene):
    res = []
    all = listsOfMatchingCharsInScene.copy()
    while len(all) > 0:
        a = []
        recursiveFind(a, all[0], all)
        if (len(a) >= MIN_NUMBER_OF_MATCHING_CHARS):
            res.append(a)
    return res

def recursiveFind(current, i, all):
    current.append(i)
    all.remove(i)
    k = 0
    while k < len(all):
        if all[k] not in current and distanceBetweenChars(i, all[k]) < i.intBoundingRectHeight * MAX_DISTANCE_IN_HEIGHT:
            recursiveFind(current, all[k], all)
        else:
            k += 1

def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                          
        fltAngleInRad = math.atan(fltOpp / fltAdj)     
    else:
        fltAngleInRad = 1.5708    

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)    

    return fltAngleInDeg

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)             

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:      
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:      
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:             
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar) 
                    else:                                                                      
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:              
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar) 

    return listOfMatchingCharsWithInnerCharRemoved

def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""         

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    mysort(listOfMatchingChars)  

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                   

    for currentChar in listOfMatchingChars:                                       
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, SCALAR_GREEN, 2)        

        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))         

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT)) 
        npaROIResized = np.float32(npaROIResized)              
        

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 5)       

        strCurrentChar = str(chr(int(npaResults[0][0])))          

        strChars = strChars + strCurrentChar                       

    if DetectPlates.showSteps == True:
        cv2.imshow("10", imgThreshColor)

    return strChars
    
def mysort(listOfMatchingChars):
    for i in range(len(listOfMatchingChars) - 1):
        for j in range(i + 1, len(listOfMatchingChars)):
            if abs(listOfMatchingChars[i].intCenterY - listOfMatchingChars[j].intCenterY) < listOfMatchingChars[i].intBoundingRectHeight * MAX_CHANGE_IN_LINE:
                if (listOfMatchingChars[i].intCenterX > listOfMatchingChars[j].intCenterX):
                    temp = listOfMatchingChars[i]
                    listOfMatchingChars[i] = listOfMatchingChars[j]
                    listOfMatchingChars[j] = temp
            elif listOfMatchingChars[i].intCenterY > listOfMatchingChars[j].intCenterY:
                temp = listOfMatchingChars[i]
                listOfMatchingChars[i] = listOfMatchingChars[j]
                listOfMatchingChars[j] = temp


def GetPILImage(img):
    pilImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(pilImg)
    return pilImage

