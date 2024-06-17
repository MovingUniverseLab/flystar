import pickle

# Need to add these functions to a utility .py file rather than storing them in general structure. 
def open_archive(file_name):
    """
    Helper function to open archived files. 
    """
    with open(file_name, 'rb') as file_archive:
        file_dict = pickle.load(file_archive)
    return file_dict

def save_archive(file_name, save_data):
    """
    Helper function to archive a file. 
    """
    with open(file_name, 'wb') as outfile:
        pickle.dump(save_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return