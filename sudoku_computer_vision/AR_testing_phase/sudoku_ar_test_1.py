import cv2
import numpy as np
from PIL import Image
from cnn import model
from cnn import model
from predict import predict
from digit_extractor import extract_digits
from sudoku_solver import *
from sudoku_helper import *


# detected = None
detected = False
solved = False
tiles = []
model = model.model()



def process(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
    return dilated


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners

def write_image(image, cells, sudoku, sol, dst_img, pts):
    image = image.copy()
    solution = []
    for i in sol:
        for x in i:
            solution.append(x)
    points = cells[80]
    print(points)
    for i in range(len(cells)) :
        points = cells[i]
        x = int((points[0][0] + points[1][0])/ 2) - 5
        y = int((points[0][1] + points[1][1]) / 2) + 9
        print(x,y)
        print("sudoku is", sudoku[i])
        if sudoku[i] == 0 :
            cv2.putText(image, str(solution[i]),(x, y) ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2 )
    


    pts_source = np.array([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]],dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(image, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped


    # round_tripped = cv2.perspectiveTransform(image, inv)
    # cv2.imshow('image', dst_img)
    # cv2.imwrite('testing3.jpg', dst_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst_img



def img_preprocess(image, skip_dilate= False):
    # Converting to a grey scale iage
    image_grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # Using gaussian blur to reduce thw noise from the image
    image_blur = cv2.GaussianBlur(image_grey, (9,9), 0)
    image_thresh = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2)
    image_bitwise = cv2.bitwise_not(image_thresh, image_thresh)

    if not skip_dilate:
		# Dilate the image to increase the size of the grid lines.
	    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
	    image_bitwise = cv2.dilate(image_bitwise, kernel)

    return image_bitwise


def largest_contour(contours):
    """
    contourArea helps in finding the area of the contour.
    We are finding the contur which have the maximun area i.e) outer region.
    """
    return max(contours, key=lambda cnt: cv2.contourArea(cnt))



def find_contours(image):
    """
    Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity. 
    Contours come handy in shape analysis, finding the size of the object of interest, and object detection.
    OpenCV has findContour() function that helps in extracting the contours from the image. 
    It works best on binary images, so we should first apply thresholding techniques, Sobel edges, etc.
    """
    contours, hierachy = cv2.findContours(image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return largest_contour(contours)

def getContours(image):
    image = image.copy()
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    polygon = all_contours[0]
    return contours, polygon

def getCoords(image, polygon):
    sums = []
    diffs = []

    for point in polygon:
	    for x, y in point:
		    sums.append(x + y)
		    diffs.append(x - y)

    top_left = polygon[np.argmin(sums)].squeeze()
    bottom_right = polygon[np.argmax(sums)].squeeze() 
    top_right = polygon[np.argmax(diffs)].squeeze()
    bottom_left = polygon[np.argmin(diffs)].squeeze() 
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)

def warp(image, coords):
    ratio = 1.0
    tl, tr, br, bl = coords
    widthA = np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[1])**2)
    widthB = np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[1])**2)
    heightA = np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[1])**2)
    heightB = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[1])**2)
    width = max(widthA, widthB) * ratio
    height = width

    destination = np.array([
    [0, 0],
    [height, 0],
    [height, width],
    [0, width]], dtype = np.float32)
    M = cv2.getPerspectiveTransform(coords, destination)
    warped = cv2.warpPerspective(image, M, (int(height), int(width)))
    return warped

def extractGrid(image, rects):
    tiles = []
    for coords in rects:
        rect = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
        tiles.append(rect)
    return tiles
	
def displayGrid(image):
	cell_height = image.shape[0] // 9
	cell_width = image.shape[1] // 9
	indentation = 0
	rects = []

	for i in range(9):
		for j in range(9):
			p1 = (j*cell_height + indentation, i*cell_width + indentation)
			p2 = ((j+1)*cell_height - indentation, (i+1)*cell_width - indentation)
			rects.append((p1, p2))
			cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
	return rects

def unwarp(image, coords):
	ratio = 1.0
	tl, tr, br, bl = coords
	widthA = np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[1])**2)
	widthB = np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[1])**2)
	heightA = np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[1])**2)
	heightB = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[1])**2)
	width = max(widthA, widthB) * ratio
	height = width

	destination = np.array([
	[0, 0],
	[height, 0],
	[height, width],
	[0, width]], dtype = np.float32)
	M = cv2.getPerspectiveTransform(coords, destination)
	unwarped = cv2.warpPerspective(image, M, (int(height), int(width)), flags = cv2.WARP_INVERSE_MAP)
	return unwarped


cap = cv2.VideoCapture(0)

while True:
    # retr = True
    retr, frame = cap.read()
    try :
        # image_path = r"C:\Users\Lenovo\Workspace\Machine Learning\Computer vision\Sudoku\image\sudoku_1.JPG"#raw_input("Enter the image file path :")
        # frame = cv2.imread(image_path)
        preprocess = img_preprocess(frame)
        # bitwise = cv2.bitwise_not(preprocess.copy(), preprocess.copy())
        contourImage = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
        coordsImage = contourImage.copy()
        # cv2.imshow("frame", frame)
        # contours = find_contours(contourImage)
        contours, polygon = getContours(preprocess)
        coords = getCoords(contourImage, polygon)
        if detected and solved:
            unwarpedImage = unwarp(solutionImage, coords)
        else:
            unwarpedImage = np.zeros((frame.shape[0], frame.shape[1]))

        if cv2.contourArea(polygon) > 80000 and not detected:
            # coords = getCoords(contourImage, polygon)

            # for coord in coords:
            #     cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)
            #     cv2.circle(frame, (coord[0], coord[1]), 5, (255, 0, 0), -1)
            cv2.drawContours(contourImage, polygon, -1, (0, 255, 0), 3)
            cv2.drawContours(frame, polygon, -1, (0, 255, 0), 3)
            warpedImage = warp(coordsImage.copy(), coords)
            warpedImage = cv2.resize(warpedImage, (540, 540))
            rects = displayGrid(warpedImage)
            tiles = extractGrid(warpedImage, rects)
            if cv2.contourArea(polygon) >= 90000:
                print("Detected")
                detected = True
                # cv2.imwrite('./frame.png', frame)
                # cv2.imwrite('./preprocess.png', preprocess)
                # cv2.imwrite('./contour.png', contourImage)
                # cv2.imwrite('./coords.png', coordsImage)
                solved = False
            else:
                print("Bring closer...")

        else:
            warpedImage = np.zeros((540, 540))

        if detected and not solved:
            # digits = extract_digits(frame)
            digits, cells, cropped = extract_digits(frame)
            values = predict(digits, model)
            actual_input = convert(values)
            answer = solve(actual_input)
            processed = process(frame)
            corners = get_corners(processed)
            solutionImage = write_image(cropped, cells, values, answer, frame, np.array(corners))
            print_board(answer)
            # predictions = getPredictions(tiles)
            #print(predictions)
            # solutionImage = solveSudoku(predictions, coords)
            cv2.imwrite('./solution.png', solutionImage)
            solved = True

        if retr == True:

            if solved:
                cv2.imshow("Solution", solutionImage)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            
        else:
            cap.release()
            cv2.destroyAllWindows()
            break

    except Exception as e:
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if debug:
            print(e)
        continue
# cap.release(0)
# cv2.destroyAllWindows()





while True:
    ret, frame = cap.read()

    try:
        cv2.imshow('hhhhh', frame)
        preprocess = img_preprocess(frame)
        contourImage = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
        coordsImage = contourImage.copy()
        contours, polygon = getContours(preprocess)
        coords = getCoords(contourImage, polygon)

        if cv2.contourArea(polygon) > 80000 :

            cv2.drawContours(contourImage, polygon, -1, (0, 255, 0), 3)
            cv2.drawContours(frame, polygon, -1, (0, 255, 0), 3)
            warpedImage = warp(coordsImage.copy(), coords)
            warpedImage = cv2.resize(warpedImage, (540, 540))
            rects = displayGrid(warpedImage)
            tiles = extractGrid(warpedImage, rects)
            if cv2.contourArea(polygon) >= 90000:
                print("Detected")
                detected = True
                solved = False
            else:
                print("Bring closer...")

        else:
            warpedImage = np.zeros((540, 540))

        if detected:
            digits, cells, cropped = extract_digits(frame)
            values = predict(digits, model)
            actual_input = convert(values)
            answer = solve(actual_input)
            processed = process(frame)
            corners = get_corners(processed)
            solutionImage = write_image(cropped, cells, values, answer, frame, np.array(corners))
            print_board(answer)
            cv2.imshow('qqqqq', solutionImage)
            cv2.imwrite('./solution.png', solutionImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue