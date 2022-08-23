import re
import glob
import pandas as pd


# get path to all files
SIMULATED_FILES_PATH = '../data/synthetic/spectra/simulated_data/'
INDICES_FILE_PATH = '../data/synthetic/params/'

tree_based_file_indices_model = pd.read_csv(INDICES_FILE_PATH + 'testing_data_indices.csv', header=None, skiprows=1)
nn_based_file_indices_model = pd.read_csv(INDICES_FILE_PATH + 'nn_testing_data_indices.csv', header=None, skiprows=1)


files = sorted(glob.glob(SIMULATED_FILES_PATH + 'model_parameters_data*.txt'), key=lambda x: int(re.search(r'\d+', x).group()))


filename_list = []

def get_filenames(SIMULATED_FILES_PATH, model_type):
    """
    Get the file names of the files in the directory.
    """
    if model_type == 'tree':
        filename_list.clear() # clear the list
        tree_file_indices = tree_based_file_indices_model
        for file_index in tree_file_indices[1]:
            for index, file in enumerate(files):
                if file_index == index:
                    filename_list.append(file.split('/')[-1])
    
    elif model_type == 'nn':
        filename_list.clear() # clear the list
        nn_file_indices = nn_based_file_indices_model
        for file_index in nn_file_indices[1]:
            for index, file in enumerate(files):
                if file_index == index:
                    filename_list.append(file.split('/')[-1])

    elif model_type == 'lte':
        filename_list.clear() # clear the list
        tree_file_indices = tree_based_file_indices_model
        nn_file_indices = nn_based_file_indices_model

        list_of_model_file_indices = [tree_file_indices, nn_file_indices]
        for model_file in list_of_model_file_indices:
            for file_index in model_file[1]:
                for index, file in enumerate(files):
                    if file_index == index:
                        filename_list.append(file.split('/')[-1])
    else:
        raise ValueError('model_type must be either "lte", "tree" or  "nn"')
    
    return filename_list[:]