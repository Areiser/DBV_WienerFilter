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
STD_DEV = 0.0001
# Create noise with the same shape as the image, random normal distribution
noise = np.random.normal(size = img.shape, loc=0, scale=STD_DEV * 255)

output3 =  np.clip((imageWithMotionBlur + noise), 0, 255).astype('uint8')
noisy_output = cv2.cvtColor(output3, cv2.COLOR_GRAY2RGB)


### Wiener Filter
#img_psd, freq = plt.psd(img)
#print(img_psd)
#mean_power_density_spectrum_input = np.fft.fft(image_mean)**2
#mean_power_density_spectrum_noise = np.fft.fft(noise.mean())**2
H = np.fft.fft2(kernel_motion_blur)
# Noise is still present
wiener_filter = (np.conjugate(H) / (np.absolute(H)**2 + 500))
print(wiener_filter)
#restored = np.real(np.fft.ifftn(np.fft.fft2(np.array(img)) * wiener_filter))
#print(wiener_filter)
wiener_filter = np.real(np.fft.ifftn(wiener_filter)) * 255
print(wiener_filter)
print(wiener_filter.shape)
restored = cv2.filter2D(imageWithMotionBlur, -1, wiener_filter)
print(restored)

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

print("\n\nRestored")
plt.imshow(restored)
plt.show()