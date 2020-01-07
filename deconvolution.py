import cv2
import numpy as np
import matplotlib.pyplot as plt

# original image
image=cv2.imread("input2.jpg", 0)

#Define a psf for horizontal blur
psf = np.zeros((image.shape[0], image.shape[1]))
psf[int(psf.shape[0]/2), int(psf.shape[1]/2 - 75):int(psf.shape[1]/2+75)] = np.ones(150)
psf /= 150

# FFT original image
F = np.fft.fft2(image)

# Frequencies of PSF
H=np.fft.fft2(psf)

# Create noise with standard deviation 5
STD_DEV = 5
noise = np.random.normal(size = F.shape, loc=0, scale=STD_DEV)
# F * H = apply filter to motion blur image
# then inverse fourier and shift to get the normal representation instead of frequencies
g = np.fft.fftshift(np.fft.ifft2(F*H).real)
g = g + noise

G=np.fft.fft2(g)

# Constants for simplified wiener filter
LOW_K = 0.01
HIGH_K = 0.1

# Wiener filter (simplified: conjugated part of H / H^2 + K) convolved with the image
low_k_deconv = (np.conj(H)/((np.conj(H) * H) + LOW_K))*G
high_k_deconv = (np.conj(H)/((np.conj(H) * H) + HIGH_K))*G

# Map real part of the image back from frequencies
restored_image_low_k = np.fft.fftshift(np.fft.ifft2(low_k_deconv).real)
restored_image_high_k = np.fft.fftshift(np.fft.ifft2(high_k_deconv).real)

noise_pds = np.square(np.abs(np.fft.fft2(noise)))
image_pds = np.square(np.abs(np.fft.fft2(image)))

perfect_deconv = ((np.conj(H) * image_pds)/(((np.conj(H) * H) * image_pds) + noise_pds))*G
restored_image_perfect = np.fft.fftshift(np.fft.ifft2(perfect_deconv).real)

plt.subplot(231)
plt.imshow(image,cmap="gray")
plt.colorbar()
plt.title("Original image")

plt.subplot(232)
plt.imshow(psf,cmap="gray")
plt.title("Point spread function")
plt.colorbar()
plt.subplot(233)

plt.imshow(g,cmap="gray")
plt.title("Motion blurred and noisy image")
plt.colorbar()

plt.subplot(234)
plt.imshow(restored_image_perfect,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image using Power density spectrum")
plt.colorbar()

plt.subplot(235)
plt.imshow(restored_image_high_k,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image using high K")
plt.colorbar()

plt.subplot(236)
plt.imshow(restored_image_low_k,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image using low K")
plt.colorbar()

### Perfect deconvolution with weaker motion blur

# different PSF: only use 100 px to get weaker blur
psf = np.zeros((image.shape[0], image.shape[1]))
psf[int(psf.shape[0]/2), int(psf.shape[1]/2 - 50):int(psf.shape[1]/2+50)] = np.ones(100)
psf /= 100

H=np.fft.fft2(psf)

g = np.fft.fftshift(np.fft.ifft2(F*H).real)
g = g + noise
weaker_motion_image = g
G=np.fft.fft2(g)


deconv = ((np.conj(H) * image_pds)/(((np.conj(H) * H) * image_pds) + noise_pds))*G
weaker_motion_restored = np.fft.fftshift(np.fft.ifft2(deconv).real)

g = np.fft.fftshift(np.fft.ifft2(F*H).real)

# Weaker noise
STD_DEV = 0.1
noise = np.random.normal(size = F.shape, loc=0, scale=STD_DEV)

g = g + noise
weaker_noise_image = g
G=np.fft.fft2(g)

noise_pds = np.square(np.abs(np.fft.fft2(noise)))

deconv = ((np.conj(H) * image_pds)/(((np.conj(H) * H) * image_pds) + noise_pds))*G
weaker_noise_restored = np.fft.fftshift(np.fft.ifft2(deconv).real)


plt.show()

plt.subplot(221)
plt.imshow(weaker_motion_image,cmap="gray",vmin=0,vmax=255)
plt.title("Image with weaker motion blur")
plt.colorbar()

plt.subplot(222)
plt.imshow(weaker_motion_restored,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image with weaker motion blur")
plt.colorbar()

plt.subplot(223)
plt.imshow(weaker_noise_image,cmap="gray",vmin=0,vmax=255)
plt.title("Image with weaker noise and weaker blur")
plt.colorbar()

plt.subplot(224)
plt.imshow(weaker_noise_restored,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image with weaker noise and blur")
plt.colorbar()

plt.show()