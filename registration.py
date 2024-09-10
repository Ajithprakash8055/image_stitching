### read 

from pathlib import Path
def get_image_paths(img_set):
    return [str(path.relative_to('/home/madmax/stitching/data')) for path in Path('/home/madmax/stitching/data').rglob(f'{img_set}*')]




### resize
# resizing the images can be done with normal resize function check the code in main function 


### find features




### match features 


### select subsets 


### camera parameters estimation and correction


### registration
