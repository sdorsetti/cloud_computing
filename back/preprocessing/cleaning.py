import os

def get_size(midi_file):
    """_summary_

    Args:
        midi_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.getsize(midi_file)

def sort_by_size(folder, threshold):
    """_summary_

    Args:
        folder (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_list = []
    for item in list(sorted(os.listdir(folder))):
        if get_size(folder + item) < threshold:
            new_list.append(folder + item)
    return new_list
