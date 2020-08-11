import cv2
import numpy as np
from PIL import Image

# vid = cv2.VideoCapture(0)

# while True :
#     ret, frame = vid.read()

#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     image_blur = cv2.GaussianBlur(img, (9,9), 0)
#     image_thresh = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2)
#     contours, hierachy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break

# # After the loop release the cap object 
# vid.release() 
# # Destroy all the windows 
# cv2.destroyAllWindows() 




########################################################################################################################################
# def preprocessImage(image, skip_dilation = False):
#     preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     preprocess = cv2.GaussianBlur(preprocess, (9, 9), 0)
#     preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     if not skip_dilation:
#         kernel = np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], dtype = np.uint8)
#         preprocess = cv2.dilate(preprocess, kernel, iterations = 1)

#     return preprocess

# def getContours(image):
#     image = image.copy()
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     polygon = all_contours[0]
#     return contours, polygon


# def getCoords(image, polygon):
#     sums = []
#     diffs = []

#     for point in polygon:
# 	    for x, y in point:
# 		    sums.append(x + y)
# 		    diffs.append(x - y)

#     top_left = polygon[np.argmin(sums)].squeeze()
#     bottom_right = polygon[np.argmax(sums)].squeeze() 
#     top_right = polygon[np.argmax(diffs)].squeeze()
#     bottom_left = polygon[np.argmin(diffs)].squeeze() 
    
#     return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)

# def warp(image, coords):
#     ratio = 1.0
#     tl, tr, br, bl = coords
#     widthA = np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[1])**2)
#     widthB = np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[1])**2)
#     heightA = np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[1])**2)
#     heightB = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[1])**2)
#     width = max(widthA, widthB) * ratio
#     height = width

#     destination = np.array([
#     [0, 0],
#     [height, 0],
#     [height, width],
#     [0, width]], dtype = np.float32)
#     M = cv2.getPerspectiveTransform(coords, destination)
#     warped = cv2.warpPerspective(image, M, (int(height), int(width)))
#     return warped

# def extractGrid(image, rects):
#     tiles = []
#     for coords in rects:
#         rect = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
#         tiles.append(rect)
#     return tiles
	
# def displayGrid(image):
# 	cell_height = image.shape[0] // 9
# 	cell_width = image.shape[1] // 9
# 	indentation = 0
# 	rects = []

# 	for i in range(9):
# 		for j in range(9):
# 			p1 = (j*cell_height + indentation, i*cell_width + indentation)
# 			p2 = ((j+1)*cell_height - indentation, (i+1)*cell_width - indentation)
# 			rects.append((p1, p2))
# 			cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
# 	return rects


cap = cv2.VideoCapture(0)
detected = False
solved = False
tiles = []
print("Get board closer to webcam until stated otherwise...")
while True:
    retr, frame = cap.read()
    cv2.imshow('frame', frame)
    #####################################################
    # preprocess = preprocessImage(frame)
    # preprocess = cv2.bitwise_not(preprocess.copy(), preprocess.copy())
    # contourImage = preprocess.copy()
    # contourImage = cv2.cvtColor(contourImage, cv2.COLOR_GRAY2BGR)
    # coordsImage = contourImage.copy()
    # contours, polygon = getContours(preprocess)
    #print(cv2.contourArea(polygon))
    #print(cv2.contourArea(polygon))
    # coords = getCoords(contourImage, polygon)
    #print(coords)
    # if cv2.contourArea(polygon) > 80000 and not detected:
    			#coords = getCoords(contourImage, polygon)
        # for coord in coords:
            # cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)
            #cv2.circle(frame, (coord[0], coord[1]), 5, (255, 0, 0), -1)
            # cv2.drawContours(contourImage, polygon, -1, (0, 255, 0), 3)
            # cv2.drawContours(frame, polygon, -1, (0, 255, 0), 3)
            # warpedImage = warp(coordsImage.copy(), coords)
            # warpedImage = cv2.resize(warpedImage, (540, 540))
            # rects = displayGrid(warpedImage)
            # tiles = extractGrid(warpedImage, rects)
            # if cv2.contourArea(polygon) >= 90000:
            #     print("Detected")
            #     detected = True
            #     cv2.imwrite('./frame.png', frame)
            #     cv2.imwrite('./preprocess.png', preprocess)
            #     cv2.imwrite('./contour.png', contourImage)
            #     cv2.imwrite('./coords.png', coordsImage)
                #solved = False
            # else:
            #     print("Bring closer...")
            # for i, tile in enumerate(tiles):
            # 	cv2.imwrite('./tiles/' + str(i) + '.png', tile)
    
    # else:
    #     #print("Show puzzle...")
    #     warpedImage = np.zeros((540, 540))