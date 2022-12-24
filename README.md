# HOG-method
HOG is a feature descriptor. It is used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in the localized portion of an image. This method is quite similar to Edge Orientation Histograms and Scale Invariant aFeature Transformation (SIFT). The HOG descriptor focuses on the structure or the shape of an object. It is better than any edge descriptor as it uses magnitude as well as angle of the gradient to compute the features. For the regions of the image it generates histograms using the magnitude and orientations of the gradient.

Article : N. Dalal et al., “Histograms of Oriented Gradients for Human Detection,” IEEE, 2005, doi: 
10.1109/CVPR.2005.177.

fscore file : calculates Accuracy, Sensitivity, Specificity, Precision, Recall, F-Measure, G-mean of the model.

Other files contain noisy model.
