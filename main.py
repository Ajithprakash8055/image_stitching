from matplotlib import pyplot as plt
import cv2 as cv 
import numpy as np 
from pathlib import Path
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.timelapser import Timelapser
from stitching.cropper import Cropper
from stitching.blender import Blender
import os
import glob
from skimage.measure import shannon_entropy
import cv2



##### plot a single image 

def plot_image(img, figsize_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
###### plot multiple images 

def plot_images(imgs, figsize_inches=(5,5)):
    fig, axs = plt.subplots(1,len(imgs),figsize=figsize_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()   
    
## locating the images 

## renaming the images

# def rename():
#     folder_path = '/home/madmax/stitching/data'
#     image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

#     for i, filepath in enumerate(image_files,start=1):
#         new_name=f"parking{i}.jpg"
#         new_path=os.path.join(folder_path,new_name)
#         os.rename(filepath,new_name)
#         print("done")


# parking_imgs= get_image_paths('park')



# Folder where your images are stored
folder_path = '/home/madmax/stitching/image_stitching/data'

# List to store the images
images = []

# Loop over each file in the folder
for filename in os.listdir(folder_path):
    # Create full file path
    file_path = os.path.join(folder_path, filename)
    
    # Check if it's an image file (you can specify extensions as needed)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        img = cv2.imread(file_path)
        
        if img is not None:
            # Append the image to the list
            images.append(img)
        else:
            print(f"Failed to load image: {file_path}")

# Now 'images' contains all the images in the folder
print(f"Loaded {len(images)} images.")



print(images)
# exposure_error_imgs = get_image_paths('exp')
# barcode_imgs = get_image_paths('barc')
# barcode_masks = get_image_paths('mask')

# resize the images 
from stitching.images import Images
images=Images.of(images)

medium_images=list(images.resize(Images.Resolution.MEDIUM))
low_images =list(images.resize(Images.Resolution.LOW))
final_images= list(images.resize(Images.Resolution.FINAL))



original_size = images.sizes[0]
low_size = images.get_image_size(low_images[0])
medium_size = images.get_image_size(medium_images[0])
final_size = images.get_image_size(final_images[0])

# print(f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px ~ 1 MP")
# print(f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px ~ 0.6 MP")
# print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px ~ 0.1 MP")
# print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px ~ 1 MP")


## find the features 

finder = FeatureDetector(detector='sift', nfeatures=700)
features =[ finder.detect_features(img) for img in medium_images]
keypoints_center_images= finder.draw_keypoints(medium_images[1],features[1])
# plot_image(keypoints_center_images,(15,10))

## feature Matcher
 
matcher = FeatureMatcher()
matches = matcher.match_features(features)
# print(matcher.get_confidence_matrix(matches))


all_relevant_matches = matcher.draw_matches_matrix(medium_images, features, matches, conf_thresh=1, 
                                                   inliers=True, matchColor=(0, 255, 0))

for idx1, idx2, img in all_relevant_matches:
    print(f"Matches Image {idx1+1} to Image {idx2+1}")
    # plot_image(img, (20,10))

## creating a subset 

subsetter = Subsetter()
dot_notation = subsetter.get_matches_graph(images.names, matches)
print(dot_notation)


indices = subsetter.get_indices_to_keep(features, matches)

medium_imgs = subsetter.subset_list(medium_images, indices)
low_imgs = subsetter.subset_list(low_images, indices)
final_imgs = subsetter.subset_list(final_images, indices)
features = subsetter.subset_list(features, indices)
matches = subsetter.subset_matches(matches, indices)

images.subset(indices)

print(images.names)
print(matcher.get_confidence_matrix(matches))


## camera parameters estimation 

CameraEstimator(estimator='homography')
CameraAdjuster(adjuster='ray', refinement_mask='xxxxx')
WaveCorrector(wave_correct_kind='horiz')



camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster()
wave_corrector = WaveCorrector()

cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)


## Warping images 
warper = Warper()
warper.set_scale(cameras)


low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)  # since cameras were obtained on medium imgs

warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)


# plot_images(warped_low_imgs, (10,10))
# plot_images(warped_low_masks, (10,10))

print(final_corners)
print(final_sizes)


# excursion Timelapser 
## it is used to visualize how the image will look in the in final plane


timelapser = Timelapser('as_is')
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(warped_final_imgs, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    # plot_image(frame, (10,10))
    
## croping the images based on the interior rectangle 

cropper = Cropper()
mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
# plot_image(mask, (5,5))

lir = cropper.estimate_largest_interior_rectangle(mask)
print(lir)

plot = lir.draw_on(mask, size=2)
# plot_image(plot, (5,5))

low_corners = cropper.get_zero_center_corners(low_corners)
rectangles = cropper.get_rectangles(low_corners, low_sizes)

plot = rectangles[1].draw_on(plot, (0, 255, 0), 2)  # The rectangle of the center img
# plot_image(plot, (5,5))

## estimating the overlap new corners and size 

overlap = cropper.get_overlap(rectangles[1], lir)
plot = overlap.draw_on(plot, (255, 0, 0), 2)
# plot_image(plot, (5,5))

intersection = cropper.get_intersection(rectangles[1], overlap)
plot = intersection.draw_on(warped_low_masks[1], (255, 0, 0), 2)
# plot_image(plot, (2.5,2.5))

cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

cropped_low_masks = list(cropper.crop_images(warped_low_masks))
cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)  # since lir was obtained on low imgs
cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

timelapser = Timelapser('as_is')
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(cropped_final_imgs, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    # plot_image(frame, (10,10))
    
    
#####composition of the image

# Seam Masks
# Exposure Error Compensation
# Blending

## seam mask 
## Seam masks find a transition line between images with the least amount of interference.

from stitching.seam_finder import SeamFinder

seam_finder = SeamFinder()

seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]

seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(cropped_final_imgs, seam_masks)]
# plot_images(seam_masks_plots, (15,10))

## Exposure Error Compensation
##exposure errors respectively exposure differences between images occur which lead to artefacts in the final panorama. 

from stitching.exposure_error_compensator import ExposureErrorCompensator

compensator = ExposureErrorCompensator()

compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

compensated_imgs = [compensator.apply(idx, corner, img, mask) 
                    for idx, (img, mask, corner) 
                    in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]


###Blending
# With all the previous processing the images can finally be blended to a whole panorama

blender = Blender()
blender.prepare(final_corners, final_sizes)
for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
    blender.feed(img, mask, corner)
panorama, _ = blender.blend()
cv2.imwrite('output.jpg', panorama)
# plot_image(panorama, (20,20))


## evaluating the image

entropy_value = shannon_entropy(panorama)
print(f"Entropy: {entropy_value}")




# if __name__ =='__main__':
#     rename()