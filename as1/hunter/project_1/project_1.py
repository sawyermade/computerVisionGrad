from functions import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    imageFilename = "images/shift.jpg"

    img = grey_image_read(imageFilename)

    img = mean_shift(img)

    #labImage = rgb_to_lab(img)

    #rgbImage = lab_to_rgb(labImage)

    image_save("shift_test.png", img)


'''
Questions to ask tmrw:

Is it okay to use the openCv way to convert the image to LAB

Do we need to normalize the values from 0-255 before we do the mean shift calculations

Can we get an example of what a correct input / output should look like

'''