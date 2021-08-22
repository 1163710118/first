import random
from random import randint
import math
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from fastai.vision import ItemLists, bb_pad_collate, tensor
from object_detection_fastai.helper.wsi_loader import SlideContainer, ObjectItemListSlide, SlideObjectCategoryList



def sample_function_train(y, classes, size, level_dimensions, level):
    width, height = level_dimensions[level]
    if len(y[0]) == 0:
        return random.randint(0, width - size[0]), random.randint(0, height -size[1])
    else:
        if randint(0, 5) < 2: ##upsample patches containing annotations
            class_id = np.random.choice(classes, 1)[0] # select a random class
            ids = np.array(y[1]) == class_id # filter the annotations according to the selected class
            xmin, ymin, _, _ = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)] # randomly select one of the filtered annotatons as seed for the training patch
            
            # To have the selected annotation not in the center of the patch and an random offset.
            xmin += random.randint(-size[0]/2, size[0]/2) 
            ymin += random.randint(-size[1]/2, size[1]/2)
            xmin, ymin = max(0, int(xmin - size[0] / 2)), max(0, int(ymin -size[1] / 2))
            xmin, ymin = min(xmin, width - size[0]), min(ymin, height - size[1])
            return xmin, ymin
        else:
            return random.randint(0, width - size[0]), random.randint(0, height -size[1])
       
    
    
def sample_function_test(y, classes, size, level_dimensions, level):
    width, height = level_dimensions[level]
    return random.randint(0, width - size[0]), random.randint(0, height -size[1])

          
          
def create_wsi_container(annotations_df: pd.DataFrame, res_level, patch_size, input_folder="/drive/MyDrive/MIDOG_Challenge/images/",train=True):

    container = []

    for image_name in tqdm(annotations_df["file_name"].unique()):

        image_annos = annotations_df[annotations_df["file_name"] == image_name]

        bboxes = [box   for box   in image_annos["box"]]
        labels = [label for label in image_annos["cat"]]
        
        if train==True:
            container.append(SlideContainer(input_folder+str(image_name), y=[bboxes, labels], level=res_level,width=patch_size, height=patch_size, sample_func=sample_function_train))
        else:
            container.append(SlideContainer(input_folder+str(image_name), y=[bboxes, labels], level=res_level,width=patch_size, height=patch_size, sample_func=sample_function_test))
    return container
  
  
  
def sample_selector(dataframe, train_samples, val_samples, batch_size, transforms, patch_size=256, res_level=0, 
                    train_scanner = "Hamamatsu XR", val_scanner = "Hamamatsu S360",
                    random_seed = None, normalise = True, testdataframe = None,
                    input_folder="/drive/MyDrive/MIDOG_Challenge/images/"):
  
  if patch_size not in [256,512,1024]:
    print("Suggested patch sizes are 256, 512 or 1024")
  if random_seed:
    np.random.seed(random_seed)

  train_annos = dataframe[dataframe["scanner"].isin(train_scanner.split(","))]
  train_container = create_wsi_container(train_annos, res_level, patch_size,input_folder,train=True)

  if testdataframe is not None:
    val_annos = testdataframe[testdataframe["scanner"].isin(val_scanner.split(","))]
    valid_container = create_wsi_container(val_annos, res_level, patch_size,input_folder,train=False)
  else:
    val_annos = dataframe[dataframe["scanner"].isin(val_scanner.split(","))]
    valid_container = create_wsi_container(val_annos, res_level, patch_size,input_folder,train=False)

  train_images = list(np.random.choice(train_container, train_samples))
  valid_images = list(np.random.choice(valid_container, val_samples))

  train, valid = ObjectItemListSlide(train_images), ObjectItemListSlide(valid_images)

  item_list = ItemLists(".", train, valid)
  lls = item_list.label_from_func(lambda x: x.y, label_cls=SlideObjectCategoryList)
  lls = lls.transform(transforms, tfm_y=True, size=train_images[0].height)

  norm_mean=[0,0,0]
  norm_sd=[0,0,0]
  if "Hamamatsu XR" in train_scanner:
    norm_mean=[sum(x) for x in zip(norm_mean, [197.53/255,143.54/255,202.30/255])]
    norm_sd=[sum(x) for x in zip(norm_sd,[math.sqrt(690.74)/255,math.sqrt(1279.30)/255,math.sqrt(237.16)/255])]
  if "Hamamatsu S360" in train_scanner:
    norm_mean=[sum(x) for x in zip(norm_mean,[206.11/255,144.28/255,187.62/255])]
    norm_sd=[sum(x) for x in zip(norm_sd,[math.sqrt(670.45)/255,math.sqrt(1522.41)/255,math.sqrt(601.91)/255])]
  if "Aperio CS" in train_scanner:
    norm_mean=[sum(x) for x in zip(norm_mean,[202.74/255,149.97/255,174.83/255])]
    norm_sd=[sum(x) for x in zip(norm_sd,[math.sqrt(731.78)/255,math.sqrt(1480.70)/255,math.sqrt(855.76)/255])]
  if "Leica GT450" in train_scanner:
    norm_mean=[sum(x) for x in zip(norm_mean,[231.52/255,197.26/255,230.18/255])]
    norm_sd=[sum(x) for x in zip(norm_sd,[math.sqrt(317.51)/255,math.sqrt(797.85)/255,math.sqrt(134.20)/255])]

  norm_mean=[i/len(train_scanner.split(",")) for i in norm_mean]
  norm_sd=[i/len(train_scanner.split(",")) for i in norm_sd]
    
  if normalise:
    data = lls.databunch(bs=batch_size, collate_fn=bb_pad_collate,num_workers=0
                     ).normalize([tensor(norm_mean),tensor(norm_sd)])
    print("Validation data normalised on scanner "+str(train_scanner))
  else:
    data = lls.databunch(bs=batch_size, collate_fn=bb_pad_collate,num_workers=0
                     )

  return(data)
