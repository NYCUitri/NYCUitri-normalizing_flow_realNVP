"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""




########################################################################
# import additional python-library
########################################################################

# from import
from tqdm import tqdm

########################################################################

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse

import os

# additional
import numpy
# import librosa
# import librosa.core
# import librosa.feature
import common as com
import json
import re
########################################################################



########################################################################


########################################################################
# feature extractor
########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         get_label=False):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    labels_file = open("jsons\\labels_for_test.json", "w")
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    labels_out = []
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        labels_list = []
        if get_label:
            for  i in range(len(vector_array)):
                regex = re.compile(r'normal')
                s = regex.search(file_list[idx])
                if s != None:
                    label = 0
                else:
                    regex = re.compile(r'anomaly')
                    s = regex.search(file_list[idx])
                    if s != None:
                        label = 1
                    else:
                        label = -1
                labels_list.append(label)
        labels_out.append(labels_list)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    json.dump(labels_out, labels_file)

    return dataset


def file_list_generator(target_dir,
                        dir_name="target_test",
                        ext="wav",
                        begin=0,
                        end=None):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    # logger.info("target_dir : {}".format(target_dir))
    print("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    
    if end == None:
        end = len(files)
    print('I sample', str(end-begin), 'flies, from ', str(begin), 'to', str(end))
    files=files[begin:end]

    if len(files) == 0:
        # logger.exception("no_wav_file!!")
        print("no_wav_file!!")

    print("train_file num : {num}".format(num=len(files)))
    # logger.info("train_file num : {num}".format(num=len(files)))
    return files
########################################################################



def dataloader(machine_type, begin=0, end=None, get_label = False):
    param = com.yaml_load()

    dir_path = os.path.abspath("E:/dcase2021/dev_data/*")
    dirs = sorted(glob.glob(dir_path))

    # dirs = select_dirs(param=param, mode=True)
    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        if target_dir!='E:\\dcase2021\\dev_data\\'+machine_type:
            print(target_dir)
            continue
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            print("model exists")
            continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir, begin=begin, end=end)
        # for f in files:
        #     print(f)
        
        
        train_data = list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"],
                                          get_label=get_label)

    return train_data

# dataloader()

