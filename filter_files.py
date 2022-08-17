import re
import glob
import pandas as pd


# get path to all files
PATH = '../data/synthetic/spectra/simulated_data/'
INDICES_PATH = '../data/synthetic/params/'

file_indices = pd.read_csv(INDICES_PATH + 'nn_testing_data_indices.csv', header=None, skiprows=1)
files = sorted(glob.glob(PATH + 'model_parameters_data*.txt'), key=lambda x: int(re.search(r'\d+', x).group()))


filename_list = []

def get_filenames(PATH):
    """
    Get the file names of the files in the directory.
    """
    file_indices = pd.read_csv(INDICES_PATH + 'nn_testing_data_indices.csv', header=None)
    files = sorted(glob.glob(PATH + 'model_parameters_data*.txt'), key=lambda x: int(re.search(r'\d+', x).group()))
    for file_index in file_indices[1]:
        for index, file in enumerate(files):
            if file_index == index:
                filename_list.append(file.split('/')[-1])

    return filename_list[1:]