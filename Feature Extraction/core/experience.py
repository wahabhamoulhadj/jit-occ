"""
Script for extracting the experience features in a software repository.
"""


__author__ = "Oscar Svensson"
__copyright__ = "Copyright (c) 2018 Axis Communications AB"
__license__ = "MIT"

import csv
import gc
import json
import os
import time
from datetime import datetime
from pygit2 import Repository, GIT_SORT_TOPOLOGICAL, GIT_SORT_REVERSE
from tqdm import tqdm
from math import floor
from core.config import repository_path_location, repository_branch, project_name


def set_to_list(obj):
    """
    Helper function to turn sets to lists and floats to strings.
    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, float):
        return str('%.15g' % obj)
    raise TypeError


def get_files_in_tree(tree, repo):
    """
    Function to get the files in a tree.
    """
    files = set()
    for entry in tree:
        if entry.type == "tree":
            sub_files = [(f[0], "{}/{}".format(entry.name, f[1]))
                         for f in get_files_in_tree(repo[entry.id], repo)]
            files.update(sub_files)
        else:
            blob = repo[entry.id]
            if not blob.is_binary:
                if entry.name.endswith("java"):
                    files.add((entry.hex, entry.name))
    return files


def get_diffing_files(commit, parent, repo):
    """
    Function to get the files that differs between two commits.
    """
    diff = repo.diff(parent, commit)

    patches = [p for p in diff]

    files = set()

    for patch in patches:
        if patch.delta.is_binary:
            continue
        nfile = patch.delta.new_file
        files.add((nfile.id, nfile.path, patch.delta.status))

    return files


def count_diffing_subsystems(subsystems):
    """
    Function for counting the number of subsystems in a repository.
    """
    number = 0
    for system in subsystems.values():
        number = number + count_diffing_subsystems(system)

    return number + len(subsystems.keys())


def init_dir(full_path):
    folder_output = ''
    for index, dir in enumerate(full_path.split('/')):
        if index > 0 and '.' not in dir:
            folder_output += dir + '/'
            try:
                current_directory = os.getcwd()
                final_directory = os.path.join(current_directory, folder_output)
                # Create target Directory
                os.mkdir(final_directory)
                print("Directory ", final_directory, " Created ")
            except FileExistsError:
                print("Directory ", final_directory, " already exists")


def save_experience_features_graph(branch, graph_author_path, graph_file_path, type):
    """
    Function to get and save the experience graph.
    """

    init_dir(graph_author_path)
    init_dir(graph_file_path)

    repo = Repository(repository_path_location)
    head = repo.references.get(branch)
    if head is None:
        print("Please check the branch name !!")
        exit(1)
    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))
    current_commit = repo.head.target

    start_time = time.time()

    current_commit = repo.get(str(current_commit))
    files = get_files_in_tree(current_commit.tree, repo)
    if type == 1:
        all_authors = {}
        author = current_commit.committer.name.strip().lower()
        all_authors[author] = {}
        all_authors[author]['lastcommit'] = current_commit.hex
        all_authors[author][current_commit.hex] = {}
        all_authors[author][current_commit.hex]["exp"] = 1.0
        all_authors[author][current_commit.hex]["rexp"] = len(files)  # [[len(files), 1]]
        all_authors[author][current_commit.hex]["sexp"] = 1.0
        list_size = len(all_authors)
        temp = list_size
    else:
        all_files = {}
        for (_, name) in tqdm(files):
            all_files[name] = {}
            all_files[name]['lastcommit'] = current_commit.hex
            all_files[name][current_commit.hex] = {}
            all_files[name][current_commit.hex]["prevcommit"] = ""
            all_files[name][current_commit.hex]["authors"] = [
                current_commit.committer.name
            ]
        list_size = len(all_files)
        temp = list_size

    for i, commit in enumerate(tqdm(commits[1:])):
        # Build experince features
        files = get_diffing_files(commit, commits[i], repo)
        if type == 1:
            author = commit.committer.name.strip().lower()
            if author not in all_authors:
                all_authors[author] = {}
                all_authors[author]['lastcommit'] = commit.hex
                all_authors[author][commit.hex] = {}
                all_authors[author][commit.hex]["exp"] = 1.0
                all_authors[author][commit.hex]["rexp"] = len(files) / 1.0  # [[len(files), 1]]
                all_authors[author][commit.hex]["sexp"] = 1.0

            else:

                last_commit = all_authors[author]['lastcommit']
                all_authors[author]['lastcommit'] = commit.hex
                all_authors[author][commit.hex] = {}
                all_authors[author][commit.hex]["exp"] = 1.0 + all_authors[author][last_commit]['exp']
                # Start rexp
                date_current = datetime.fromtimestamp(commit.commit_time)
                date_last = datetime.fromtimestamp(repo.get(last_commit).commit_time)

                diffing_years = abs(floor(float((date_current - date_last).days) / 365))
                overall = all_authors[author][last_commit]['rexp']
                if overall <= 0:
                    overall = 1
                all_authors[author][commit.hex]["rexp"] = float(len(files) / ((overall + diffing_years) + overall))
                # End rexp

                diff = repo.diff(commits[i], commit)
                patches = [p for p in diff]
                # Extract all different subsystems that have been modified
                modules = set([])
                subsystems_mapping = {}
                for patch in patches:
                    # Skip binary files
                    if patch.delta.is_binary:
                        continue
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

                # Check how many subsystems that have been touched
                modified_systems = count_diffing_subsystems(subsystems_mapping)
                overall = all_authors[author][last_commit]["sexp"]
                if overall <= 0:
                    overall = 1
                all_authors[author][commit.hex]["sexp"] = float(modified_systems / ((overall + diffing_years) + overall))
            if i % 10 == 0:
                gc.collect()

            list_size = len(all_authors)
            if temp > list_size:
                print('bug !!')
            else:
                temp = list_size
        else:

            for (_, name, _) in files:
                if name not in all_files:
                    all_files[name] = {}

                last_commit = ""
                if 'lastcommit' not in all_files[name]:
                    all_files[name]['lastcommit'] = commit.hex
                else:
                    last_commit = all_files[name]['lastcommit']

                all_files[name][commit.hex] = {}
                all_files[name][commit.hex]["prevcommit"] = last_commit

                authors = set([commit.committer.name])
                if last_commit:
                    try:
                        shi = all_files[name][last_commit]["authors"]
                    except Exception as e:
                        all_files[name][commit.hex]["authors"] = authors
                        # print(e)
                    authors.update(all_files[name][last_commit]["authors"])

                all_files[name][commit.hex]["authors"] = authors

                all_files[name]['lastcommit'] = commit.hex
            if i % 10 == 0:
                gc.collect()

    print("Exporting JSON ...")
    if type == 1:
        with open(graph_author_path + ".json", 'w') as output:
            json.dump(all_authors, output, default=set_to_list)
    else:

        with open(graph_file_path + ".json", 'w') as output:
            json.dump(all_files, output, default=set_to_list)

    end_time = time.time()

    print("Done")
    print("Overall processing time {}".format(end_time - start_time))


def save_experience_features_graph1(graph_author_path, graph_file_path, type):
    """
    Function to get and save the experience graph.
    """

    init_dir(graph_author_path)
    init_dir(graph_file_path)

    repo = Repository(repository_path_location)
    head = repo.references.get(repository_branch)

    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))
    current_commit = repo.head.target

    start_time = time.time()

    current_commit = repo.get(str(current_commit))
    files = get_files_in_tree(current_commit.tree, repo)
    if type == 1:
        all_authors = {}
        author = current_commit.committer.name
        all_authors[author] = {}
        all_authors[author]['lastcommit'] = current_commit.hex
        all_authors[author][current_commit.hex] = {}
        all_authors[author][current_commit.hex]['prevcommit'] = ""
        all_authors[author][current_commit.hex]["exp"] = 1
        all_authors[author][current_commit.hex]["rexp"] = [[len(files), 1]]
        all_authors[author][current_commit.hex]["sexp"] = {}
    else:
        all_files = {}

        for (_, name) in tqdm(files):
            all_files[name] = {}
            all_files[name]['lastcommit'] = current_commit.hex
            all_files[name][current_commit.hex] = {}
            all_files[name][current_commit.hex]["prevcommit"] = ""
            all_files[name][current_commit.hex]["authors"] = [
                current_commit.committer.name
            ]

    for i, commit in enumerate(tqdm(commits[1:])):
        # Build experince features
        files = get_diffing_files(commit, commits[i], repo)
        if type == 1:
            author = commit.committer.name
            if author not in all_authors:
                all_authors[author] = {}
                all_authors[author]['lastcommit'] = commit.hex
                all_authors[author][commit.hex] = {}
                all_authors[author][commit.hex]['prevcommit'] = ""
                all_authors[author][commit.hex]["exp"] = 1
                all_authors[author][commit.hex]["rexp"] = [[len(files), 1.0]]
                all_authors[author][commit.hex]["sexp"] = {}
            else:
                last_commit = all_authors[author]["lastcommit"]
                all_authors[author]["lastcommit"] = commit.hex
                all_authors[author][commit.hex] = {}
                all_authors[author][commit.hex]['prevcommit'] = last_commit
                try:
                    all_authors[author][commit.hex]['exp'] = 1 + all_authors[author][last_commit]['exp']
                except:
                    all_authors[author][commit.hex]['exp'] = 1

                date_current = datetime.fromtimestamp(commit.commit_time)
                date_last = datetime.fromtimestamp(repo.get(last_commit).commit_time)

                diffing_years = abs(floor(float((date_current - date_last).days) / 365))
                try:
                    overall = all_authors[author][last_commit]['rexp']
                    all_authors[author][commit.hex][
                        'rexp'] = [[len(files), 1.0]] + [[e[0], e[1] + diffing_years]
                                                         for e in overall]
                except:
                    all_authors[author][commit.hex]["rexp"] = [[len(files), 1.0]]
                    all_authors[author][commit.hex]["sexp"] = {}

                diff = repo.diff(commits[i], commit)

                patches = [p for p in diff]
                # Extract all different subsystems that have been modified
                modules = set([])
                subsystems_mapping = {}

                for patch in patches:
                    # Skip binary files
                    if patch.delta.is_binary:
                        continue
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

                # Check how many subsystems that have been touched
                modified_systems = count_diffing_subsystems(subsystems_mapping)
                all_authors[author][commit.hex][
                    'sexp'] = [[modified_systems, 1.0]] + [[e[0], e[1] + diffing_years]
                                                           for e in overall]
            if i % 10 == 0:
                gc.collect()
        else:

            for (_, name, _) in files:
                if name not in all_files:
                    all_files[name] = {}

                last_commit = ""
                if 'lastcommit' not in all_files[name]:
                    all_files[name]['lastcommit'] = commit.hex
                else:
                    last_commit = all_files[name]['lastcommit']

                all_files[name][commit.hex] = {}
                all_files[name][commit.hex]["prevcommit"] = last_commit

                authors = set([commit.committer.name])
                if last_commit:
                    try:
                        shi = all_files[name][last_commit]["authors"]
                    except Exception as e:
                        all_files[name][commit.hex]["authors"] = authors
                        # print(e)
                    authors.update(all_files[name][last_commit]["authors"])

                all_files[name][commit.hex]["authors"] = authors

                all_files[name]['lastcommit'] = commit.hex
            if i % 10 == 0:
                gc.collect()

    print("Exporting JSON ...")
    if type == 1:
        with open(graph_author_path + ".json", 'w') as output:
            json.dump(all_authors, output, default=set_to_list)
    else:

        with open(graph_file_path + ".json", 'w') as output:
            json.dump(all_files, output, default=set_to_list)

    end_time = time.time()

    print("Done")
    print("Overall processing time {}".format(end_time - start_time))


def load_experience_features_graph(path="./results/author_graph.json"):
    """
    Function to load the feeatures graph.
    """
    print("loading file {}".format(path), end='\r')
    with open(path + '.json', 'r') as inp:
        file_graph = json.load(inp, parse_float=lambda x: float(x))
    print("loaded file {}".format(path), end='\r')
    return file_graph


def get_experience_features(graph_authors):
    """
    Function that extracts the experience features from a experience graph.
    """
    branch = "refs/heads/"+repository_branch
    repo = Repository(repository_path_location)
    head = repo.references.get(branch)

    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))
    current_commit = repo.head.target

    files = get_files_in_tree(repo.get(str(current_commit)).tree, repo)

    features = []

    commit_feat = {"commit_hex": str(commits[0].hex),
                   "exp": float(1.0),
                   "rexp": float(0.0),
                   "sexp": float(0.0),
                   # "NDEV":float(1.0),
                   # "AGE":float(0.0),
                   # "NUC":float(len(files))
                   }
    features.append(commit_feat)

    for i, commit in enumerate(tqdm(commits[1:])):
        # Experience section
        author = commit.committer.name
        if author in graph_authors.keys():
            exp = graph_authors[author][commit.hex]['exp']
            rexp = graph_authors[author][commit.hex]['rexp']
            sexp = graph_authors[author][commit.hex]['sexp']
        else:
            exp = 0
            rexp = 0
            sexp = 0

        commit_feat = {"commit_hex": str(commit.hex),
                       "exp": float(exp),
                       "rexp": float(rexp),
                       "sexp": float(sexp),
                       }

        features.append(commit_feat)
        if i % 10 == 0:
            gc.collect()

        # This condition for test only
        # if i >= 3500:
        #        break
    return features


def get_history_features(graph_files):
    """
    Function that extracts the experience features from a experience graph.
    """
    branch = "refs/heads/"+repository_branch
    repo = Repository(repository_path_location)
    head = repo.references.get(branch)

    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))
    current_commit = repo.head.target

    files = get_files_in_tree(repo.get(str(current_commit)).tree, repo)

    features = []

    commit_feat = {"commit_hex": str(commits[0].hex),
                   "NDEV": float(1.0),
                   "AGE": float(0.0),
                   "NUC": float(len(files))
                   }

    features.append(commit_feat)

    for i, commit in enumerate(tqdm(commits[1:])):
        # History section
        files = get_diffing_files(commit, commits[i], repo)

        total_number_of_authors = set()
        total_age = []
        total_unique_changes = set()

        for (_, name, _) in files:
            sub_graph = graph_files[name][commit.hex]
            total_number_of_authors.update(sub_graph['authors'])

            prev_commit = sub_graph['prevcommit']
            if prev_commit:
                total_unique_changes.add(prev_commit)

                prev_commit_obj = repo.get(prev_commit)

                total_age.append(commit.commit_time -
                                 prev_commit_obj.commit_time)

        total_age = float(sum(total_age)) / len(total_age) if total_age else 0
        total_age = (total_age / 60 / 60 / 24) if total_age >= 0 else 0  # convert from timestamp unit to days
        commit_feat = {"commit_hex": str(commit.hex),
                       "NDEV": float(len(total_number_of_authors)),
                       "AGE": float(total_age),
                       "NUC": float(len(total_unique_changes))
                       }

        features.append(commit_feat)
        if i % 10 == 0:
            gc.collect()

    return features


def save_experience_features(history_features, path):
    """
    Save the experience features to a csv file.
    """
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["commit", "experience", "rexp", "sexp"])
        for row in history_features:
            if row:
                writer.writerow([row[0], row[1], row[2], row[3]])


def get_part2_features():
    repo_path = repository_path_location
    branch = "refs/heads/"+repository_branch  # "refs/heads/trunk"#ARGS.branch
    graph_path_authors = "./results/" + project_name + "/authors/author_graph_" + project_name  # +".json"
    graph_path_files = "./results/" + project_name + "/files/files_graph_" + project_name  # +".json"

    if not os.path.exists(graph_path_authors + ".json"):
        print(
            "The graph file does not exists, we need to build it, it will take long time ...\nStart build authors ...")
        save_experience_features_graph(branch, graph_path_authors, graph_path_files, 1)
        gc.collect()
    if not os.path.exists(graph_path_files + ".json"):
        print("The graph file does not exists, we need to build it, it will take long time ...\nStart build files ...")
        save_experience_features_graph(branch, graph_path_authors, graph_path_files, 2)
        gc.collect()
    graph_authors = load_experience_features_graph(graph_path_authors)

    experience_features = get_experience_features(graph_authors)
    del graph_authors
    gc.collect()

    file_authors = load_experience_features_graph(graph_path_files)
    history_features = get_history_features(file_authors)
    del file_authors
    gc.collect()

    return experience_features, history_features
