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
H=np.fft.fft2(image)

# Frequencies of PSF
P=np.fft.fft2(psf)

# H * P = apply point spread function to motion blur image
# then inverse fourier and shift to get the normal representation instead of frequencies
d=np.fft.fftshift(np.fft.ifft2(H*P).real)
# add noise with standard deviation of 0.001
STD_DEV = 0.1
d=d+np.random.normal(size = d.shape, loc=0, scale=STD_DEV)

D=np.fft.fft2(d)

# regularization parameter
# (should be one to two orders of magnitude below the largest spectral component of point-spread function)
LOW_K = 0.0000001
HIGH_K = 0.00001

# Wiener filter (simplified: conjugated part of P / P^2 + K) convolved with the image
low_k_deconv = (np.conj(P)/((np.abs(P)**2.0) + LOW_K))*D
high_k_deconv = (np.conj(P)/((np.abs(P)**2.0) + HIGH_K))*D

# Map real part of the image back from frequencies
restored_image_low_k=np.fft.fftshift(np.fft.ifft2(low_k_deconv).real)
restored_image_high_k=np.fft.fftshift(np.fft.ifft2(high_k_deconv).real)

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
plt.title("Motion blurred and noisy image (d)")
plt.colorbar()

# sort 2d fourier components by magnitude and convert into 1d vector for plotting.
P2=np.abs(np.copy(P).flatten())
P2=np.sort(P2)[::-1]

plt.subplot(235)
plt.imshow(restored_image_high_k,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image using high K")
plt.colorbar()

plt.subplot(236)
plt.imshow(restored_image_low_k,cmap="gray",vmin=0,vmax=255)
plt.title("Deconvolved image using low K")
plt.colorbar()


plt.show()