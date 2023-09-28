"""
Script for extracting diffusion features from a git repository.
"""
from core.config import repository_path_location, repository_branch

__author__ = "Oscar Svensson"
__copyright__ = "Copyright (c) 2018 Axis Communications AB"
__license__ = "MIT"

import gc
import json
import time
from multiprocessing import Process, Manager, cpu_count

from numpy import log2
from pygit2 import Repository, GIT_SORT_REVERSE, GIT_SORT_TOPOLOGICAL
from tqdm import tqdm

"""
This section is for size implementation
"""


def count_files(tree, repo):
    """
    Count how many files there are in a repository.
    """
    num_files = 0
    trees = []
    visited = set()
    visited.add(tree.id)
    trees.append(tree)

    while trees:
        current_tree = trees.pop()
        for entry in current_tree:
            if entry.type == "tree":
                if entry.id not in visited:
                    trees.append(repo[entry.id])
                    visited.add(entry.id)
            else:
                num_files += 1
    return num_files


def get_file_lines_of_code(repo, tree, dfile):
    """
    Count how many lines of code there are in a file.
    """
    tloc = 0
    try:
        blob = repo[tree[dfile.path].id]

        tloc = len(str(blob.data).split('\\n'))
    except Exception as _:
        return tloc
    return tloc


def load_fixed_bugs(path, filename):
    with open(path + '/' + filename+'.json') as f:
        json_fixed = json.loads(f.read())
    return json_fixed

"""
This part for diffusions implementation
"""


def count_diffing_subsystems(subsystems):
    """
    Function for counting the number of subsystems in a repository.
    """
    number = 0
    for system in subsystems.values():
        number = number + count_diffing_subsystems(system)

    return number + len(subsystems.keys())


def count_entropy(file_changes, total_change):
    """
    Function to count entropy for some file changes.
    """
    if total_change == 0:
        return 0
    return sum([
        -1 * (float(x) / total_change) * (log2(float(x) / total_change)
                                          if x > 0 else 0)
        for x in file_changes
    ])


def parse_tree(tree, repo):
    """
    Parse a git tree and get the number of files, the number of systems and
    the number of subdirectories.
    """
    found_sub_entries = 0
    additions = 0
    file_additions = []
    tree = repo[tree.id]

    for entry in tree:
        if entry.type == "bin":
            continue
        if entry.type == "tree":
            sub_additions, sub_file_additions, sub_entries = parse_tree(
                entry, repo)
            found_sub_entries += (1 + sub_entries)
            additions += sub_additions
            file_additions.extend(sub_file_additions)
        else:
            try:
                sub_addition = len(str(repo[entry.id]).split('\n'))
                additions += sub_addition
                file_additions.append(sub_addition)
            except Exception as ex:
                print(ex)
                continue

    return additions, file_additions, found_sub_entries


"""
The core function that call upper functions
"""


def parse_diffusion_features(pid, repo_path, branch, RES, bug_json, start, stop=-1):
    """
    Function to extract diffusion features from a set of commits.
    """
    repo = Repository(repo_path)

    head = repo.references.get(branch)
    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))

    start = start - 1 if (start > 0) else start
    commits = commits[start:stop] if (stop != -1) else commits[start:]

    features = [{} for c in range(len(commits))]
    for i, commit in enumerate(tqdm(commits[1:], position=pid)):
        diff = repo.diff(commits[i], commit)

        patches = [p for p in diff]

        # Set label for commit
        hex = str(commit.hex)
        found = 0
        for b in bug_json:
            if len(b) <= 0:
                continue
            if b['hash_intro'] == hex:
                found = 1
                break
        # Extract Size features
        tree = commit.tree
        stats = diff.stats

        # Count the total lines of code and find the biggest file that have been changed
        total_tloc = 0
        line_of_code_old = 0

        # Extract all different subsystems that have been modified
        modules = set([])
        subsystems_mapping = {}
        entropy_change = 0

        file_changes = []
        total_change = 0
        for patch in patches:
            # Skip binary files
            if patch.delta.is_binary:
                continue
            _, addition, deletions = patch.line_stats
            total_change = total_change + (addition + deletions)
            file_changes.append(addition + deletions)

            # Store all subsystems
            fpath = patch.delta.new_file.path
            subsystems = fpath.split('/')[:-1]

            root = subsystems_mapping
            for system in subsystems:
                if system not in root:
                    root[system] = {}
                root = root[system]
            if len(subsystems) > 0:
                modules.add(subsystems[0])
            # Store size
            new_file = patch.delta.new_file

            # Total lines of code
            total_tloc += get_file_lines_of_code(repo, tree, new_file)

            old_file = patch.delta.old_file
            # Total lines of code in the old file
            line_of_code_old = max(
                line_of_code_old, get_file_lines_of_code(repo, tree, old_file))

        # Churned lines of code
        cloc = stats.insertions
        # Deleted lines of code
        dloc = stats.deletions
        # Churned files
        files_churned = len(patches)
        # File count
        num_files = count_files(tree, repo)

        # Check how many subsystems that have been touched
        modified_systems = count_diffing_subsystems(subsystems_mapping)

        # Calculate the entropy for the commit
        entropy_change = count_entropy(file_changes, total_change)

        # Apply relative code churns
        LA = float(cloc) / total_tloc if (total_tloc > 0) else float(cloc) #LA
        LD = float(dloc) / total_tloc if (total_tloc > 0) else float(cloc) #LD
        NF = (float(files_churned) / num_files if (num_files > 0)          #NF
              else float(files_churned))

        LT = float(line_of_code_old)                                        #LT
        feature_dict = {"commit_hex":str(commit.hex),
                        "LA":LA,
                        "LD":LD,
                        "NF":NF,
                        "LT":LT,
                        "entropy":float(entropy_change),
                        "NS":float(modified_systems),
                        "ND":float(len(modules)),
                        "commit_time": commit.commit_time,
                        "label":found
                        }
        features[i].update(feature_dict)

    RES[pid] = features


def free_memory():
    c = gc.collect()
    print('Free {} memory allocation..'.format(c))


def get_part1_features(bugs_fix_path, bugs_file_name):
    """
    Function that extracts the first commits diffusion features. It then starts
    a number of processes(equal to the number of cores on the computer), and then
    distributes the remaining commits to them.
    """
    branch = "refs/heads/"+repository_branch
    repo = Repository(repository_path_location)

    head = repo.references.get(branch)

    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))

    manager = Manager()
    res = manager.dict()
    bug_json = load_fixed_bugs(bugs_fix_path, bugs_file_name)

    # Check how many processes that could be spawned
    cpus = cpu_count()
    print("Using {} cpus...".format(cpus))
    # Divide the commits equally between the processes.
    quote, remainder = divmod(len(commits), cpus)

    # Test single process
    # parse_diffusion_features(0, repository_path_location, branch, res, bug_json, 0, 100)

    processes = [
        Process(
            target=parse_diffusion_features,
            args=(i, repository_path_location, branch, res, bug_json, i * quote + min(i, remainder),
                  (i + 1) * quote + min(i + 1, remainder))) for i in range(cpus)
    ]

    for process in processes:
        process.start()

    start_time = time.time()
    for process in processes:
        process.join()
    end_time = time.time()

    print("Done")
    print("Overall processing time {}".format(end_time - start_time))
    free_memory()
    # Assemble the results
    features = []
    for _, feat in res.items():
        features.extend(feat)
    features = list(reversed(features))
    return features
