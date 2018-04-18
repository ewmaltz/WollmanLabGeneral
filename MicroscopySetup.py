import numpy as np
import pandas as pd


def print_96_well_names(lets, nums):
    """
    Prints well names for direct reference in matlab.
    :param lets: list of well letters.
    :param nums: list of well numbers.
    :return: None. Just prints the list.
    """
    wells = []
    for let in lets:
        for i in nums:
            wells.append((let + str(i).zfill(2)))
    print(wells)


def plate2dict(plate_path):
    """
    Converts an excel spreadsheet of your plate into a dictionary of which wells correspond to which treatments.
    :param plate_file: excel spreadsheet with labeled rows and columns corresponding to the plate
    :return treat_dict: a dictionary with well name and corresponding treatment (e.g. {'C03': 'TNF'})
    """

    # get plate; get column, and row names
    plate = pd.read_excel(plate_path)
    rows = plate.index.tolist()
    columns = plate.columns.tolist()

    # iterate through column/row names and add them to the dict if they have treatment values in that cell
    treat_dict = {}
    for column in columns:
        for row in rows:
            if pd.notna(plate.loc[row, column]):
                cell_name = row + '0' + str(column)
                treat_dict[cell_name] = plate.loc[row, column].replace('\n', '')
    return treat_dict
