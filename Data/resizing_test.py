
import cv2
import pandas as pd
import csv

columnNames = list()
for i in range(784):
    pixel = 'pixel' + str(i)
    columnNames.append(pixel)

#with open("test_data/Vandy_data/test_gray.csv", 'w') as f:
    #writer = csv.writer(f)
    #writer.writerow(columnNames)
with open("Vandy_data/test_data_updated/gray_test_updated.csv", 'a') as f:
    for i in range(0, 91):
        if i == 0:
            writer = csv.writer(f)
            writer.writerow(columnNames)
        else:
            image = cv2.imread('Vandy_data/test_data_updated/image_' + str(i) + '.png')  # read in the image, where img_name is the path to the file
            image = cv2.resize(image, (28, 28))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inverted = (255 - gray)  # this is the inverted image
            inverted_arr = inverted.flatten()
            # print(inverted_arr)
            writer = csv.writer(f)
            writer.writerow(inverted_arr)
