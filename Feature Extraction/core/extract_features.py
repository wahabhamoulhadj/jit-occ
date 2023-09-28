"""
Script for main feature extraction process.
"""

__author__ = "Mohammed Shehab"
__copyright__ = "Copyright (c) 2023"
__license__ = "MIT"

from core.config import project_name, repository_path_location, repository_branch, szz_results

import gc
import os

import pandas as pd
from core.experience import get_part2_features
from core.purpose import get_purpose_features
from core.size import get_part1_features


def extract_features():
    """
    The main function to extract the 14 features from the code repository
    :rtype:
    """
    if not os.path.exists(szz_results + project_name + '_History_Experience.csv'):
        experience, history = get_part2_features()
        df11 = pd.DataFrame(columns=['commit_hex', 'exp', 'rexp', 'sexp'], data=experience)
        df12 = pd.DataFrame(columns=['commit_hex', 'NDEV', 'AGE', 'NUC'], data=history)
        df1 = pd.merge(df11, df12, on='commit_hex')
        df1.to_csv(szz_results + project_name + '_History_Experience.csv', header=True, index=False)
        gc.collect()
    else:
        print("Skip build experience, the results are in {}".format('./' + project_name + '_History_Experience.csv'))
        df1 = pd.read_csv(szz_results + project_name + '_History_Experience.csv')

    bug_path = './results/' + project_name + '/' + project_name + '_SZZ_results'

    if not os.path.exists(szz_results + project_name + 'code_size.csv'):
        code_size = get_part1_features('.', bug_path)
        df2 = pd.DataFrame(
            columns=['commit_hex', 'LA', 'LD', 'NF', 'LT', 'entropy', 'NS', 'ND', 'commit_time', 'label'],
            data=code_size)
        df2.to_csv(szz_results + project_name + 'code_size.csv', header=True, index=False)
        gc.collect()
    else:
        print("Skip build code size, the results are in {}".format('./code_size.csv'))
        df2 = pd.read_csv(szz_results + project_name + 'code_size.csv')

    if not os.path.exists(szz_results + project_name + 'purpose.csv'):
        purpose = get_purpose_features()
        df3 = pd.DataFrame(columns=['commit_hex', 'fix'], data=purpose)
        df3.to_csv(szz_results + project_name + 'purpose.csv', header=True, index=False)
        gc.collect()
    else:
        print("Skip build purpose, the results are in {}".format('./purpose.csv'))
        df3 = pd.read_csv(szz_results + project_name + 'purpose.csv')

    df = pd.merge(df1, df2, on='commit_hex')
    gc.collect()
    df = pd.merge(df, df3, on='commit_hex')
    gc.collect()
    df.to_csv(szz_results + project_name + '_sample_features.csv', header=True, index=False)
