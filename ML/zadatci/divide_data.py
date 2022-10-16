import os
import splitfolders

splitfolders.ratio("gtsrb_dataset/Train", output="gtsrb_dataset/Train_divide", seed=1337, ratio=(0.85,0.15))