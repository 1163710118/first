import json
import pandas as pd

def get_bbox_df():

  image_folder = "/drive/MyDrive/MIDOG_Challenge/images"

  hamamatsu_rx_ids = list(range(0, 51))
  hamamatsu_360_ids = list(range(51, 101))
  aperio_ids = list(range(101, 151))
  leica_ids = list(range(151, 201))

  annotation_file = "/drive/MyDrive/MIDOG_Challenge/MIDOG.json"
  rows = []
  with open(annotation_file) as f:
      data = json.load(f)
      categories = {1: 'mitotic figure', 2: 'hard negative'}

      for row in data["images"]:
          file_name = row["file_name"]
          image_id = row["id"]
          width = row["width"]
          height = row["height"]

          scanner  = "Hamamatsu XR"
          if image_id in hamamatsu_360_ids:
              scanner  = "Hamamatsu S360"
          if image_id in aperio_ids:
              scanner  = "Aperio CS"
          if image_id in leica_ids:
              scanner  = "Leica GT450"
          
          for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
              box = annotation["bbox"]
              cat = categories[annotation["category_id"]]
              point = [0.5*(box[0]+box[2]),0.5*(box[1]+box[3])]
              rows.append([file_name, image_id, width, height, box, point, cat, scanner])

  df = pd.DataFrame(rows, columns=["file_name", "image_id", "width", "height", "box", "point", "cat", "scanner"])
  return(df)


def train_test_split(df):
  test_files=[]
  for i in range(10):
    test_files.append("0"+str(i+41)+".tiff")
  for i in range(9):
    test_files.append("0"+str(i+91)+".tiff")
  test_files.append("100.tiff")
  for i in range(10):
    test_files.append(str(i+141)+".tiff")
  for i in range(9):
    test_files.append(str(i+191)+".tiff")
  test_files.append("200.tiff")

  df_test = df[df['file_name'].isin(test_files)].reset_index()
  df_train = df[-df['file_name'].isin(test_files)].reset_index()
  return df_train, df_test
