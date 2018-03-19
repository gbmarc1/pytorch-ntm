"""Utilities for ntm module
"""


def calculate_num_params(o):
    """Returns the total number of parameters."""
    num_params = 0
    for p in o.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params
