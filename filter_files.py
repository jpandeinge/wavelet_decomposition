import re
import glob
import pandas as pd


# get path to all files
path = 'data/spectra/simulated_data/'

vlsr_list = []
filename_list = []

def get_filenames(path):
    """
    Get the file names of the files in the directory.
    """
    files = sorted(glob.glob(path + 'model_parameters_data*.txt'), key=lambda x: int(re.search(r'\d+', x).group()))
    for file in files:
        data = pd.read_csv(file, sep='\t', header=None)

        # get the postive and negative float values of line 6 using regular expression
        vlsr_list.append(float(re.findall(r'[+-]?\d+\.\d+', str(data.iloc[5, :]))[0]))

        # append filename to list
        filename_list.append(file.split('/')[-1])

    # put the filename and vlsr values in a dataframe
    vlsr_df = pd.DataFrame({'filename': filename_list, 'vlsr': vlsr_list})
    vlsr_df = vlsr_df[vlsr_df['vlsr']>-60].reset_index(drop=True)
    
    filenames  = vlsr_df['filename']
   
    return filenames