from __future__ import division
import numpy as np
import cv2
import sys
import os

'''global data common to all vision algorithms'''
isTracking = False
r=g=b=0.0
image = np.zeros((640,480,3), np.uint8)
trackedImage = np.zeros((640,480,3), np.uint8)
imageWidth=imageHeight=0
globalx = 0
globaly = 0
targetArea = False
targetWidth = 30
targetHeight = 30
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
MAX_ITERATION = 10

# SOME OPENCV2 MEANSHIFT VIRABLES
target_hist = None
prior_window = None

def captureVideo(src):
    video = True
    terminate = False
    global image, isTracking, trackedImage, imageWidth, imageHeight, globalx, globaly, targetWidth, targetHeight, prior_window
    cap = cv2.VideoCapture(src)
    if cap.isOpened() and src=='0':
        ret = cap.set(3,640) and cap.set(4,480)
        if ret==False:
            video = False
            print( 'Cannot set frame properties, returning' )
            return
    # FILE WITH EXTENSIONS
    elif "." in src:
        frate = cap.get(cv2.CAP_PROP_FPS)
        print( frate, ' is the framerate' )
        waitTime = int( 1000/frate )
    else:
        waitTime = 100
        video = False
    if src == 0:
        waitTime = 1
    if cap:
        print( 'Succesfully set up capture device' ) 
    else:
        print( 'Failed to setup capture device' ) 

    windowName = 'Input View, press q to quit'
    cv2.namedWindow(windowName)
    cv2.setMouseCallback( windowName, clickHandler )
    fileNameLst = []
    while(True):
        if terminate:
            break
        if (video):
            # Capture frame-by-frame
            ret, image = cap.read()
            if ret==False:
                terminate = True
            imageHeight, imageWidth, _ = image.shape
            # Display the resulting frame
            if targetArea and isTracking:
                prior_window = track(prior_window, target_hist)
                cv2.rectangle(image, (globalx, globaly), (globalx + targetWidth, globaly + targetHeight), (0,255,0), 2)
            cv2.imshow(windowName, image)                                        
            inputKey = cv2.waitKey(waitTime) & 0xFF
            if inputKey == ord('q'):
                terminate = True
            elif inputKey == ord('t'):
                isTracking = not isTracking  
        else:
            for filename in os.listdir(src):
                # FINISHED ALL FILES ALREADY
                if (filename in fileNameLst):
                    print("Finished Scanning All Images")
                    terminate = True
                    break
                fileNameLst.append(filename)
                if not filename.endswith("png") and not filename.endswith("jpg") and not filename.endswith("gif"):
                    continue
                image = cv2.imread(os.path.join(src, filename))
                if image is not None:
                    imageHeight, imageWidth, _ = image.shape
                    # Display the resulting frame
                    if targetArea and isTracking:
                        prior_window = track(prior_window, target_hist)
                        cv2.rectangle(image, (globalx, globaly), (globalx + targetWidth, globaly + targetHeight), (0,255,0), 2)
                    cv2.imshow(windowName, image)                                        
                    inputKey = cv2.waitKey(waitTime) & 0xFF
                    if inputKey == ord('q'):
                        terminate = True
                        break
                    elif inputKey == ord('t'):
                        isTracking = not isTracking                
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def track(prior_window, target_hist):
    global globalx, globaly
    current_window = np.copy(prior_window)
    count = 0
    x_old = 0
    y_old = 0
    while True:
        x_new = 0
        y_new = 0
        x_center = int(targetHeight / 2)
        y_center = int(targetWidth / 2)
        # CONVERGE OR CANNOT FAILED FIND OPTIMAL MEAN
        if count > MAX_ITERATION:
            break
        # FIND TARGET SECTION
        [columnCoordinate, rowCoordinate, sectionWidth, sectionHeight] = current_window
        targetSection = image[rowCoordinate:rowCoordinate + sectionHeight, columnCoordinate:columnCoordinate + sectionWidth]
        # CREATE HISTOGRAM
        current_hist = createHistogram(targetSection, x_center, y_center)
        # FIND THE SIMILIARITY OF TARGET HISTGRAM AND CURRENT HISTOGRAM
        Bhattacharyya_distance = find_norm_similarity_histograms(target_hist, current_hist)
        sum_weight = 0
        # Bhattacharyya DISTANCE OF EACH PIXEL
        greyScaleImage = np.copy(targetSection)
        cvt = cv2.cvtColor(greyScaleImage, cv2.COLOR_BGR2GRAY)
        for i in range(targetHeight):
            for j in range(targetWidth):
                sum_weight += Bhattacharyya_distance[cvt[i,j]]
                x_new += Bhattacharyya_distance[cvt[i,j]] * (j - y_center)
                y_new += Bhattacharyya_distance[cvt[i,j]] * (i - x_center)
        x_new /= sum_weight
        y_new /= sum_weight

        # X COORDINATE BOUNDRY
        if (current_window[0] != np.NaN):
            if (current_window[0] + x_new < 0):
                current_window[0] = 0
            elif (current_window[0] + x_new + targetWidth > imageWidth):
                current_window[0] = imageWidth - targetWidth
            else:
                current_window[0] = current_window[0] + x_new

        # Y COORDINATE BOUNDRY
        if (current_window[1] != np.NaN):
            if (current_window[1] + y_new < 0):
                current_window[1] = 0
            elif (current_window[1] + y_new + targetHeight > imageHeight):
                current_window[1] = imageHeight - targetHeight
            else:
                current_window[1] = current_window[1] + y_new
        # 7 - CONVERGE OR CANNOT FAILED FIND OPTIMAL MEAN
        if ((x_new - x_old > -1 or x_new - x_old < 1) and (y_new - y_old > -1 or y_new - y_old < 1)):
            break
        x_old, y_old = x_new, y_new
        count += 1
    # UPDATE RECTANGLE X & Y COORDINATES
    globalx = current_window[0]
    globaly = current_window[1]
    return current_window

def createHistogram(targetImage, x0, y0):
    # WEIGHT CALCULATION - CENTER PIXELS WEIGHTS MORE 
    targetSectionWeight = np.zeros((targetHeight, targetWidth))
    denominator = (x0 * x0 + y0 * y0) ** 0.5
    for i in range(targetHeight):
        for j in range(targetWidth):
            pos_normed = ((i - x0) * (i - x0) + (j - y0) * (j - y0)) ** 0.5 / denominator
            targetSectionWeight[i, j] = 1 - pos_normed
    # GENERATE HISTOGRAM
    hist_size = 256
    targetHistogram = np.zeros(hist_size)
    for i in range(targetHeight):
        for j in range(targetWidth):
            greyScaleImage = np.copy(targetImage)
            cvt = cv2.cvtColor(greyScaleImage, cv2.COLOR_BGR2GRAY)
            targetHistogram[cvt[i,j]] += targetSectionWeight[i, j]
    return targetHistogram / np.sum(targetSectionWeight)

# Bhattacharyya distance BETWEEN HISTOGRAM 1 and HISTOGRAM 2
def find_norm_similarity_histograms(h1, h2):
    size = len(h1)
    similarity_matrix = np.zeros(size)
    for i in range(size):
        similarity_matrix[i] = (h1[i] * h2[i]) ** 0.5
    similarity_matrix = similarity_matrix/sum(similarity_matrix)
    return similarity_matrix

def clickHandler( event, x, y, flags, param):
    global targetArea, image, target_hist, prior_window
    if event == cv2.EVENT_LBUTTONDOWN:
        if targetArea:
            print("Redefine Tracking Area")
        else:
            print("Set Tracking Area")
        # FIND RBG & DEFINE BOUNDING BOX
        TuneTracker(x,y)
        targetArea = True
        if targetArea:
            # SET FIRST WINDOW AND FRAME
            prior_window = [globalx, globaly, targetWidth, targetHeight]
            # INITIALIZE TARGET SECTION
            [columnCoordinate, rowCoordinate, sectionWidth, sectionHeight] = prior_window
            targetSection = image[rowCoordinate:rowCoordinate + sectionHeight, columnCoordinate:columnCoordinate + sectionWidth]
            # CALCUALTE HISTOGRAM FOR THE INITIALIZED TARGET SECTION
            x0 = round(sectionHeight / 2)
            y0 = round(sectionWidth / 2)
            target_hist = createHistogram(targetSection, x0, y0)

'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x,y):
    global r,g,b, image
    b,g,r = image[y,x]
    sumpixels = float(b)+float(g)+float(r)
    if sumpixels != 0:
        b = b/sumpixels,
        r = r/sumpixels
        g = g/sumpixels
    print( r,g,b, 'at location ', x,y )
    print("Calculating optimal target boundry box...")
    defineTrackingBoxDimension(x,y,r,b,g, sumpixels)
    print("Found optimal target boundry box...")
    return r, b, g


# I USED COLOR TO FIND THE BOUNDING BOX COORDINATES
def defineTrackingBoxDimension(x, y, r, b, g, sumpixels):
    global targetWidth, targetHeight, globalx, globaly
    updateX = True
    updateY = True
    colorThreshold = 12/sumpixels
    lstBoundryCoordinates = []
    tempBoundryCoordinates = [x,y]
    # MAXIMUM WIDTH
    targetWidth = round(imageWidth/5)
    # MAXIMUM HEIGHT
    targetHeight = round(imageHeight/5)

    # INITIALIZE LOWER AND HIGHER BOUNDS
    xlower = xhigher = x
    ylower = yhigher = y
    
    while lstBoundryCoordinates != tempBoundryCoordinates:
        if not updateX and not updateY:
            break
        tempBoundryCoordinates = [xlower, xhigher, ylower, yhigher]
        # print("Temp Boundry List: ", tempBoundryCoordinates)
        xlower, xhigher, ylower, yhigher = updateBoundryPoints(xlower, xhigher, ylower, yhigher, r, b, g, colorThreshold, updateX, updateY)
        lstBoundryCoordinates = [xlower, xhigher, ylower, yhigher]
        # print("Boundry List: ", lstBoundryCoordinates)
        if xhigher - xlower > targetWidth:
            updateX = False
        if yhigher - ylower > targetHeight:
            updateY = False
    # CONVERGE - WE FIND BOUNDRIES
    globalx = xlower
    globaly = ylower
    if xhigher - xlower <= targetWidth:
        targetWidth = xhigher - xlower
    if yhigher - ylower <= targetHeight:
        targetHeight = yhigher - ylower
    

def updateBoundryPoints(xlower, xhigher, ylower, yhigher, r, b, g, colorThreshold, updateX, updateY):
    returnXlow = xlower
    returnXhigh = xhigher
    returnYlow = ylower
    returnYhigh = yhigher
    if xlower - 1 < 0:
        xlow = 0
    else:
        xlow = xlower - 1
    if xhigher + 1 > imageWidth:
        xhigh = imageWidth
    else:
        xhigh = xhigher + 1
    if ylower - 1 < 0:
        ylow = 0
    else:
        ylow = ylower - 1
    if yhigher + 1 > imageHeight:
        yhigh = imageHeight
    else:
        yhigh = yhigher + 1
    if (updateX):
        # COLOR CHECK & UPDATE 
        for y in range (ylow, yhigh):
            imageb,imageg,imager = image[y,xlow-1]
            sumpixels = float(imageb)+float(imageg)+float(imager)
            if sumpixels != 0:
                imageb = imageb/sumpixels
                imager = imager/sumpixels
                imageg = imageg/sumpixels
            # SIMILAR COLOR
            if (imageb - colorThreshold <= b and imageg - colorThreshold <= g and imager - colorThreshold <= r and imageb + colorThreshold >= b and imageg + colorThreshold >= g and imager + colorThreshold >= r):
                returnXlow = xlow
                break
        for y in range (ylow, yhigh):
            imageb,imageg,imager = image[y,xhigh-1]
            sumpixels = float(imageb)+float(imageg)+float(imager)
            if sumpixels != 0:
                imageb = imageb/sumpixels
                imager = imager/sumpixels
                imageg = imageg/sumpixels
            # SIMILAR COLOR
            if (imageb - colorThreshold <= b and imageg - colorThreshold <= g and imager - colorThreshold <= r and imageb + colorThreshold >= b and imageg + colorThreshold >= g and imager + colorThreshold >= r):
                returnXhigh = xhigh
                break
    if (updateY):
        for x in range (xlow, xhigh):
            imageb,imageg,imager = image[ylow-1,x]
            sumpixels = float(imageb)+float(imageg)+float(imager)
            if sumpixels != 0:
                imageb = imageb/sumpixels
                imager = imager/sumpixels
                imageg = imageg/sumpixels
            # SIMILAR COLOR
            if (imageb - colorThreshold <= b and imageg - colorThreshold <= g and imager - colorThreshold <= r and imageb + colorThreshold >= b and imageg + colorThreshold >= g and imager + colorThreshold >= r):
                returnYlow = ylow
                break
        for x in range (xlow, xhigh):
            imageb,imageg,imager = image[yhigh-1,x]
            sumpixels = float(imageb)+float(imageg)+float(imager)
            if sumpixels != 0:
                imageb = imageb/sumpixels
                imager = imager/sumpixels
                imageg = imageg/sumpixels
            # SIMILAR COLOR
            if (imageb - colorThreshold <= b and imageg - colorThreshold <= g and imager - colorThreshold <= r and imageb + colorThreshold >= b and imageg + colorThreshold >= g and imager + colorThreshold >= r):
                returnYhigh = yhigh
                break
    return returnXlow, returnXhigh, returnYlow, returnYhigh



print( 'Starting program' )
if __name__ == '__main__':
    arglist = sys.argv
    src = 0
    print( 'Argument count is ', len(arglist) ) 
    if len(arglist) == 2:
        src = arglist[1]
        captureVideo(src)
    else:
        src = 0
        print('No video source input.')
else:
    print( 'Not in main' )