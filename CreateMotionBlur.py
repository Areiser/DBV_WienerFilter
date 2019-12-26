import cv2
import numpy as np
from matplotlib import pyplot as plt

imgRGB = cv2.imread('input2.jpg')
img = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)


### Apply Motion Blur

size = 150

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size


# applying the kernel to the input image
imageWithMotionBlur = cv2.filter2D(img, -1, kernel_motion_blur)


### Adding the noise to the blurred image

# Higher is more intense
STD_DEV = 15
# Create noise with the same shape as the image, random normal distribution
noise = np.random.normal(size = img.shape, loc=0, scale=STD_DEV)

output3 =  np.clip((imageWithMotionBlur + noise), 0, 255).astype('uint8')
noisy_output = cv2.cvtColor(output3, cv2.COLOR_GRAY2RGB)


#Displaying the different Images
print("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.show()

print("\n\n\nImage with Motion Blur")
plt.imshow(cv2.cvtColor(imageWithMotionBlur, cv2.COLOR_GRAY2RGB))
plt.show()

print("\n\n\nMotion Blur + Noise")
plt.imshow(noisy_output)
plt.show()
