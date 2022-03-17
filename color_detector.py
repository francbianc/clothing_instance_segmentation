from PIL import Image, ImageStat
import functools
import os
import tqdm

path = '...' #@@@ OVERRIDE: Path of a folder containing images 

def detect_color_image(file_path, MONOCHROMATIC_MAX_VARIANCE = 0.005, COLOR = 1000, MAYBE_COLOR = 100):
    """ Given a path of an image, returns if it is colored or not 

    :return: integer
    """
    v = ImageStat.Stat(Image.open(file_path)).var # Intensity variance over each channel (or band in ImageStat)
    types = {0:'no_idea', 1:'black_white', 2:'greyscale', 3:'monochromatic', 4:'maybe_color', 5:'color'}
    is_monochromatic = functools.reduce(lambda x, y: x and y < MONOCHROMATIC_MAX_VARIANCE, v, True) 
    if is_monochromatic: # If var of at least 2 channels are less than MONOCHROMATIC_MAX_VARIANCE
        type = 3
    else:
        if len(v)==3:
            maxmin = abs(max(v) - min(v))
            print(maxmin)
            if maxmin > COLOR: # If intensity among 2 channels has high variance (> COLOR threshold)
                type = 5
            elif maxmin > MAYBE_COLOR: # If intensity among 2 channels has medium variance (> MAYBE_COLOR threshold)
                type = 4
            else: # If intensity among 2 channels has low variance (<= MAYBE_COLOR threshold)
                type = 2
        elif len(v)==1: # If there is just 1 channel (rarely even if image appears b/w)
            type = 1 
        else:
            type = 0
    #print(types[type])
    return type

all_ids = sorted(os.listdir(path))

# Create 4 lists to store the ids of images for each color type
ids_maybe_color = []
ids_bw_grey = []
ids_no_idea_color = []
ids_monocolor = []
#for id in tqdm.tqdm(all_ids):
for id in all_ids:
    t = detect_color_image(os.path.join(path, id))
    if t == 0: 
        ids_no_idea_color.append(id)
    elif t == 1 or t == 2: 
        ids_bw_grey.append(id)
    elif t == 3: 
        ids_monocolor.append(id)
    elif t == 4: 
        ids_maybe_color.append(id)

# Save in a txt file the ids of images by color type
with open(os.path.join(path, 'ids_no_idea_color.txt'), 'w') as f:
    f.write('\n'.join(ids_no_idea_color))

with open(os.path.join(path, 'ids_bw_grey.txt'), 'w') as f:
    f.write('\n'.join(ids_bw_grey))

with open(os.path.join(path, 'ids_monocolor.txt'), 'w') as f:
    f.write('\n'.join(ids_monocolor))

with open(os.path.join(path, 'ids_maybe_color.txt'), 'w') as f:
    f.write('\n'.join(ids_maybe_color))