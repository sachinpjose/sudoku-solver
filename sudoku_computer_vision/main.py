from cnn import model
from predict import predict
from digit_extractor import extract_digits
from sudoku_solver import *
from sudoku_helper import *
import cv2
import numpy as np

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
    for i in range(len(cells)) :
        points = cells[i]
        x = int((points[0][0] + points[1][0])/ 2) - 5
        y = int((points[0][1] + points[1][1]) / 2) + 9
        if sudoku[i] == 0 :
            cv2.putText(image, str(solution[i]),(x, y) ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2 )
    

    pts_source = np.array([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]],dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(image, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img
    # cv2.imshow('image', dst_img)
    # cv2.imwrite('testing2.jpg', dst_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\Lenovo\Workspace\Machine Learning\Computer vision\Sudoku\image\sudoku_1.JPG"#raw_input("Enter the image file path :")
    x = cv2.imread(image_path)
    processed = process(x)
    corners = get_corners(processed)

    model = model.model()
    digits, cells, cropped = extract_digits(image_path)
    values = predict(digits, model)
    actual_input = convert(values)
    answer = solve(actual_input)
    original_image = write_image(cropped, cells, values, answer, x, np.array(corners))
    cv2.imwrite('Original_image.jpg', original_image)
    print_board(answer)