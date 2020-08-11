import cv2
import numpy as np


def display_image(image_label, image):
    cv2.imshow(image_label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Preprocessing image
def preprocess(image, skip_dilate= False):
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


def get_contour_corners(contour):
    """
    finding the four corners in the contour
    """
    sum_xy = [pt[0][0] + pt[0][1] for pt in contour]
    diff_xy = [pt[0][0] - pt[0][1] for pt in contour]
    
    indices = [
        np.argmin(sum_xy),
        np.argmax(diff_xy),
        np.argmax(sum_xy),
        np.argmin(diff_xy)
    ]
    
    return [contour[i][0] for i in indices]



def distance_between(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    
	# Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
    side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # inv_trans = np.linalg.pinv(dst)

	# Performs the transformation on the original image
    warp = cv2.warpPerspective(img, m, (int(side), int(side)))

    return warp

def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / 9

	# Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares



def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point

def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]



def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))




def extract_digit(img, rect, size):
    """Extracts a digit (if one exists) from a Sudoku square."""

    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    # Use fill feature finding to get the largest feature in middle of the box
    # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)



def get_digits(img, squares, size):
    """Extracts digits from their cells and builds an array"""
    digits = []
    img = preprocess(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits

def extract_digits(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (500, 500))
    # Preprocessing the image
    processed = preprocess(image)
    # display_image('Original Image', processed)
    contours = find_contours(processed)
    # 4 corners of the sudoku puzzle
    corners = get_contour_corners(contours)
    # cropping the image and warping the image.
    cropped = crop_and_warp(image, corners)
    cells = infer_grid(cropped)
    # print(cells)
    # print(cells[0].shape)
    print(cells[0])
    print(len(cells))
    display_image('cropped and warped Image', cropped)
    digits = get_digits(cropped, cells, 28)
    return digits, cells, cropped