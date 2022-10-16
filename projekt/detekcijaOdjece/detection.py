#%%
from pathlib import Path
import pathlib
import matplotlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import shutil
import json
import urllib
import PIL.Image as Image
import cv2
import requests
from IPython.display import display
from sklearn.model_selection import train_test_split

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 16, 10

np.random.seed(42)

path = "F:/AI edukacija/projekt/detekcijaOdjece/fashion-dataset/"
dataset_path = pathlib.Path(path)
images = os.listdir(dataset_path)
images


# %%
import matplotlib.image as mpimg
plt.figure(figsize=(20,20))
for i in range(10,20):
    plt.subplot(6,10,i-10+1)
    cloth_img = mpimg.imread(path + 'images/100'+ str(i) +'.jpg')
    plt.imshow(cloth_img)
    plt.axis("off")
plt.subplots_adjust(wspace=-0.5, hspace=1)
plt.show()
# %%
