import os, json
from pathlib import Path
# from PIL import Image

def get_wh(path):
  # img = Image.open(path)

  # # get width and height
  # width = img.width
  # height = img.height

  # return width, height
  return 2048, 1024


def write_data_odgt(image_path, segm_path, file_path, DATA_SIZE=10):
  with open(file_path, 'w') as fw:
    for subdir, dirs, files in os.walk(image_path):
      for dir in dirs:
        img_path_list = sorted(list(Path(image_path + '/' + dir).glob('*.png'))) #[:DATA_SIZE]
        annot_path_list = sorted(list(Path(segm_path + '/' + dir).glob('*labelIds.png'))) #[:DATA_SIZE]

        assert len(img_path_list) == len(annot_path_list)

        for i in range(len(img_path_list)):
          elm = {}
          img_path = str(img_path_list[i])
          annot_path = str(annot_path_list[i])

          elm['fpath_img'] = img_path
          elm['fpath_segm'] = annot_path

          width, height = get_wh(img_path)
          elm['width'] = width
          elm['height'] = height
          fw.write(json.dumps(elm) + '\n')
  print(f'Have succesfully wrote data to {file_path}')

write_data_odgt('./data/city_scapes/leftImg8bit_trainvaltest/leftImg8bit/train', './data/city_scapes/gtFine_trainvaltest/gtFine/train', './data/city_training.odgt', DATA_SIZE=30)
write_data_odgt('./data/city_scapes/leftImg8bit_trainvaltest/leftImg8bit/val', './data/city_scapes/gtFine_trainvaltest/gtFine/val', './data/city_validation.odgt', DATA_SIZE=30)
write_data_odgt('./data/city_scapes/leftImg8bit_trainvaltest/leftImg8bit/test', './data/city_scapes/gtFine_trainvaltest/gtFine/test', './data/city_testing.odgt', DATA_SIZE=30)


