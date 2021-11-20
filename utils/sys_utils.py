import torch


def get_device(debug=False):
    if debug:
        if torch.cuda.is_available():
            print("The code will run on GPU.")
        else:
            print("The code will run on CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
