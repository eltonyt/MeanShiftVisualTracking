# MeanShift Visual Tracking from scratch

## How To Run:
	meanShiftTracking.py {Video Path/Image Sequence Directory Name}
	i.e. meanShiftTracking.py boy-walking.mp4

## How To Track:
- Press Button "T" to start Tracking.
- Then click on the target section on the image you are interested in.

## General Idea:
- Find Target Area: 
	According to the coordinate of your click, the color of the pixel is read. I'll scan the color of the surrounded pixels layer by layer and find pixels with similar colors(RBG difference < 12). 
	
	For all pixels found, I'll define the boundry box (Target Section) as [minX, minY, maxX, maxY]
- Use OpenCV2 to read frames of video or each image of the image sequence.
- Calculate the histogram of the target section.(Weights are also considered in this process, when weight of pixels close to the center of the target section is big and weight of pixels far from the center is small)
- Compare latest calculated histogram to the histogram of the previous frame and calculate the Bhattacharyya Distance of the 2 histograms. [1]
- Find the shift vector by using Bhattacharyya DISTANCE calculated. ((Bhattacharyya DISTANCE[histogram_index] * ((x,y)-(x_center,y_center)))/sum(Bhattacharyya DISTANCE))
- The Mean Shift process takes at most 10 iterations if the shift vector is not reaching [0, 0](Shift Converges).
	
## References:
Bhattacharyya distance - https://en.wikipedia.org/wiki/Bhattacharyya_distance
