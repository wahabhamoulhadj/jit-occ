"""
Script to extract the purpose features.
"""
from core.config import repository_branch, repository_path_location

__author__ = "Oscar Svensson"
__copyright__ = "Copyright (c) 2018 Axis Communications AB"
__license__ = "MIT"

import csv
import re
from tqdm import tqdm
from pygit2 import Repository, GIT_SORT_TOPOLOGICAL, GIT_SORT_REVERSE

PATTERNS = [r"bug", r"fix", r"defect", r"patch"]


def is_fix(message):
    """
    Check if a message contains any of the fix patterns.
    """
    for pattern in PATTERNS:
        if re.search(pattern, message):
            return True
    return False


def get_purpose_features():
    """
    Extract the purpose features for each commit.
    """
    branch = "refs/heads/"+repository_branch
    repo = Repository(repository_path_location)
    head = repo.references.get(branch)

    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))

    features = []
    for i, commit in enumerate(tqdm(commits)):
        message = commit.message

        fix = 1.0 if (is_fix(message)) else 0.0

        feat = {"commit_hex":str(commit.hex),"fix":int(fix)}
        features.append(feat)

    return features


def save_features(purpose_features, path="./results/purpose_features.csv"):
    """
    Save the purpose features to a csv file.
    """
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["commit", "purpose"])
        for row in purpose_features:
            if row:
                writer.writerow([row[0], row[1]])
