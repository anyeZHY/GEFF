# import the opencv library
import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from skimage.filters import gaussian

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Image process
    W = frame.shape[0]
    L = frame.shape[1]
    a = 200
    frame = frame[int(W/2)-a:int(W/2)+a, int(L/2)-a:int(L/2)+a,:]
    # frame = random_noise(frame)
    # patch_kw = dict(patch_size=5,  # 5x5 patches
    #                 patch_distance=6,  # 13x13 search area
    #                 channel_axis=-1)
    # sigma_est = np.mean(estimate_sigma(frame, channel_axis=-1))
    # denoise_fast = denoise_nl_means(frame, h=0.6 * sigma_est, fast_mode=True, sigma = sigma_est,
    #                                 **patch_kw)
    # cv2.imshow('denoise', denoise_fast)
    gamma_cor = frame**(1/2.2)/(np.max(frame)**(1/2.2))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('gamma', gamma_cor)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
