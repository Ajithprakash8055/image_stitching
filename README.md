# image_stitching
Image Stitching Pipeline for High-Quality Panoramic Images


# project overview :
This project implements an Image Stitching algorithm, which combines multiple overlapping images into a single seamless panorama. The process involves feature detection, matching, and blending techniques to align and merge the images, producing a continuous and visually appealing panoramic image.

# Detailed Description :
Creating panoramas using computer vision is a well-established technique, yet most algorithms focus on incorporating all images into the final panorama.

In this project, we approach stitching as a multi-image matching problem. By leveraging invariant local features,it finds matches between images with robustness to variations in ordering, orientation, scale, and illumination. This approach ensures that the stitching process is less sensitive to these factors, leading to more accurate and seamless panoramic images.

## Results
Evaluating the performance of an image stitching algorithm, especially when no ground truth is available is very challenging. However, you can use several metrics to assess different aspects of the quality of the stitched panorama like 
*Entropy 
*SSIM (structual similarity index)
*Seam line quality

Entropy measures the amount of information in the image. Higher entropy often indicates more details, contrast, and texture
I have used Entropy to evaluate the image which was arround 7.5

![Final Image](/output.jpg)

## Reference :
SIFT - Research on Image Matching of Improved SIFT Algorithm
Based on Stability Factor and Feature Descriptor Simplification (Liang Tang, Shuhua Ma , Xianchun Ma
and Hairong You)

STITCHING - Automatic Panoramic Image Stitching using Invariant Features ( MATTHEW BROWN ∗ AND DAVID G. LOWE Department of Computer Science, University of British Columbia, Vancouver, Canada)

REGISTRATION - Image registration methods: a survey( Barbara Zitová*, Jan Flusser)

AFFINE TRANSFORMATIONS - Affine Transformation, Landmarks registration, Non linear Warping
(Arthur Coste)