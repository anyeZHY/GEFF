import dlib
import face_recognition as fr
from pathlib import Path
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
import os


def generate_columbia_cut_set(data_load_folder, data_save_folder, required_size, err_log_save_path,
                              default_pos=[1200, 1950], default_size=[1380, 1380]):
    """
    Args:
        required_size: a list[height,width] where the height and width must be equal to call this function
        data_load_folder, data_save_folder: the directories(in string) where you want to load the original images
            and save the cropped images
    ---------------------------------------------------------------------------------------------------------------------
    Outputs:
        This function has no return value.
        Generate file which contains all cropped images for columbia dataset.
        Generate a log file(in txt) where all directories for unsuccessful cropped images are stored
    """
    err_log_file = open(err_log_save_path, "w+")
    for curr_dir in os.listdir(data_load_folder):
        if curr_dir == '.DS_Store':
            continue
        folder = os.listdir(os.path.join(data_load_folder, curr_dir))
        file_save_path = os.path.join(data_save_folder, curr_dir)
        if os.path.exists(file_save_path):
            print("images in directory " + str(curr_dir) + " have already been generated")
            continue
        else:
            print("we are currently processing images in directory " + str(curr_dir))
        os.makedirs(file_save_path)
        for file_name in folder:
            if file_name.endswith(".jpg"):
                file_path = os.path.join(data_load_folder, curr_dir) + '/' + file_name
                img_array = fr.load_image_file(file_path)  # face_recognition library requires ndarray
                img_int = read_image(file_path)  # pytorch library requires tensor to cut
                bounding_box = fr.face_locations(img_array)
                if len(bounding_box):
                    img_crop = F.resized_crop(img_int, bounding_box[0][0], bounding_box[0][3], bounding_box[0][2] -
                                              bounding_box[0][0], bounding_box[0][1] - bounding_box[0][3],
                                              required_size)
                    save_image(img_crop/255, file_save_path + '/' + file_name)
                else:
                    err_log_file.write(file_path+'\n')
                    img_crop = F.resized_crop(img_int, default_pos[0], default_pos[1], default_size[0], default_size[1],
                                              required_size)
                    save_image(img_crop/255, file_save_path + '/' + file_name)
    err_log_file.close()


if __name__ == '__main__':
    size = [224, 224]
    data_folder = str(Path.cwd().parent.parent)+'/assets/ColumbiaGazeDataSet'
    save_folder = str(Path.cwd().parent.parent)+'/assets/ColumbiaGazeCutSet'
    err_log_path = str(Path.cwd().parent.parent) + '/assets/face_cut_err_log.txt'
    generate_columbia_cut_set(data_folder, save_folder, size, err_log_path)
