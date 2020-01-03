import imageio as s
import numpy as np
import matplotlib.pyplot as plt

# original image
image=s.imread("input2.jpg", as_gray = True)

#Define a psf for horizontal blur
psf = np.zeros((image.shape[0], image.shape[1]))
psf[int(psf.shape[0]/2), int(psf.shape[1]/2 - 75):int(psf.shape[1]/2+75)] = np.ones(150)
psf /= 150

# FFT original image
H = np.fft.fft2(image)

# Frequencies of PSF
P=np.fft.fft2(psf)

# Create noise with standard deviation
STD_DEV = 5
noise = np.random.normal(size = H.shape, loc=0, scale=STD_DEV)
# H * P = apply point spread function to motion blur image
# then inverse fourier and shift to get the normal representation instead of frequencies
d = np.fft.fftshift(np.fft.ifft2(H*P).real)
d = d + noise

D=np.fft.fft2(d)

# regularization parameter
# (should be one to two orders of magnitude below the largest spectral component of point-spread function)
LOW_K = 0.01
HIGH_K = 0.1

# Wiener filter (simplified: conjugated part of P / P^2 + K) convolved with the image
low_k_deconv = (np.conj(P)/(np.square(np.abs(P)) + LOW_K))*D
high_k_deconv = (np.conj(P)/(np.square(np.abs(P)) + HIGH_K))*D

# Map real part of the image back from frequencies
restored_image_low_k = np.fft.fftshift(np.fft.ifft2(low_k_deconv).real)
restored_image_high_k = np.fft.fftshift(np.fft.ifft2(high_k_deconv).real)

noise_spectrum = np.square(np.abs(np.fft.fft2(noise)))
image_spectrum = np.square(np.abs(np.fft.fft2(image)))

perfect_deconv = ((np.conj(P) * image_spectrum)/((np.square(np.abs(P)) * image_spectrum) + noise_spectrum))*D
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

plt.imshow(d,cmap="gray")
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


plt.show()