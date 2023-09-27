# One class SVM applied to the prediction of risky commits
# Authors: Members of the SRT lab at Concordia University
import math
import os
import gc
import time
import datetime
import pandas as pd
import numpy as np
from sklearn import metrics
from pyod.models.ocsvm import OCSVM
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

from file_io import create_nested_dir

warnings.filterwarnings('ignore')

# Global variables to generate the final report as CSV file.
tp_s = []  # True positive measurement
tn_s = []  # True negative measurement
fp_s = []  # False positive measurement
fn_s = []  # False negative measurement
precisions = []  # Precision measurement
recalls = []  # Recall measurement
f1_s = []  # F1 score measurement
accuracies = []  # Accuracy measurement
mcc_s = []  # MCC measurement
auc_s = []  # AUC measurement
projects = []  # Project names
ir_s = []  # Imbalanced ratio
cv_s = []
training_times = []
buggies = []
normals = []

mcc_s_validation = []  # MCC measurement
auc_s_validation = []  # AUC measurement
mcc_m_validation = []  # MCC measurement
auc_m_validation = []  # AUC measurement

mcc_ci = []
auc_ci = []
f1_ci = []
accuracies_ci = []

test_mode = False
ClusterCommitMode = True

# https://www.mathsisfun.com/data/confidence-interval.html
confidence_interval = {"80": 1.282,
                       "85": 1.440,
                       "90": 1.645,
                       "95": 1.960,
                       "99": 2.576,
                       "99.5": 2.807,
                       "99.9": 3.291}

figures_path = './figures'
results_path = './results'


def split_random(dataset, label, test_size, sort_time=False):
    if sort_time:
        dataset = dataset.sort_values(by="commit_time")
    features = dataset.drop([label], axis=1)
    targets = dataset[label]
    x_train, x_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=test_size,
                                                        shuffle=False,
                                                        # stratify=targets,
                                                        )

    return x_train, x_test, y_train, y_test


def split_table(path_data, scaling=False, test_size=0.35):
    df1 = pd.read_csv(path_data)
    # sort data
    df1 = df1.sort_values(by="commit_time")
    test_len = math.floor(len(df1) * test_size)
    train_len = len(df1) - test_len
    df1 = df1.drop(["commit_hex", "commit_time", "bug_id"], axis=1)
    label = "isBug"
    X = df1.drop([label], axis=1)
    Y = df1[label]

    X = X.to_numpy()
    Y = Y.to_numpy().ravel()

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_variable)
    X_train = X[0:train_len]
    Y_train = Y[0:train_len]
    X_test = X[train_len:]
    Y_test = Y[train_len:]

    if scaling:
        fit_param = MinMaxScaler().fit(X_train)  # Pre Processing the TRAINING DATA to avoid data leakage
        X_train = fit_param.transform(X_train)
        X_test = fit_param.transform(X_test)

    return X_train, X_test, Y_train, Y_test


def perform_hyper_turning_search(x_train, x_validation, nus, gammas, contaminations,scoring):
    label = "isBug"
    best_auc = 0.0
    best_model = OCSVM(
        max_iter=100
    )
    target_class = 1
    commits_healthy = x_train[x_train[label] == 0]
    commits_healthy_v = x_validation[x_validation[label] == 0].drop([label], axis=1)
    # for c in contaminations:
    for nu in nus:
        for gamma in gammas:
            # Train the model
            model = OCSVM(
                nu=nu,
                gamma=gamma,
                shrinking=False,
            )
            train_healthy_commits = commits_healthy.drop([label], axis=1)
            model.fit(train_healthy_commits)
            # Validate the model
            # merge the risk and healthy for validation
            # validation_commits = pd.concat([commits_risky, x_validation], axis=0)
            validation_commits = x_validation
            validation_commits_targets = validation_commits[label]
            validation_commits_features = validation_commits.drop([label], axis=1)
            predictions_proba = model.predict_proba(validation_commits_features)

            fpr, tpr, thresholds = metrics.roc_curve(validation_commits_targets,
                                                     predictions_proba[:, target_class],
                                                     pos_label=target_class)
            auc = metrics.auc(fpr, tpr)
            print(f"Current AUC = {auc:.4f}", end='\r')
            if best_auc < auc:
                best_auc = auc
                best_model = model
                print(f"---> Highest AUC during tuning = {best_auc:.4f}", end="\n")
    commits_healthy = x_train[x_train[label] == 0].drop([label], axis=1)
    commits_healthy = pd.concat([commits_healthy, commits_healthy_v], ignore_index=True)
    model = OCSVM(
        nu=best_model.nu,
        contamination=best_model.contamination,
        gamma=best_model.gamma,
        # max_iter=1000,
        shrinking=False
        # contamination=c
    )
    model.fit(commits_healthy)
    # print(f"{model}")
    return model


def train_ocsvm(X_train, nu, c=0.1, gamma='auto'):
    label = 'isBug'
    model = OCSVM(nu=nu,
                  contamination=c,
                  gamma=gamma,

                  )
    # healthy_commits = X_train[X_train[label] == 0]
    model.fit(X_train)
    return model


def calculate_f1_score(tp, fp, fn, tn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_mcc(tp, fp, fn, tn):
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator
    return mcc


def evaluate_ocsvm(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    # use
    auc_roc = roc_auc_score(y_test, y_scores)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    tnr = 1 - fpr
    fnr = 1 - tpr

    idx = np.argmax(np.sqrt(tpr * (1 - fpr)))
    fp = fpr[idx]
    tp = tpr[idx]
    th = thresholds[idx]
    tn = tnr[idx]
    fn = fnr[idx]
    # fp, tp, thres = roc_curve(y_test, y_predict)

    # Calculate the F1-score for each threshold
    f1 = calculate_f1_score(tp, fp, fn, tn)  # 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))

    mcc = calculate_mcc(tp, fp, fn,
                        tn)  # [matthews_corrcoef(y_test, y_scores >= threshold) for threshold in thresholds]

    return auc_roc, f1, mcc, fpr, tpr


def plot_roc_curve(y_test, y_scores, project_name, display=False):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    tnr = 1 - fpr
    fnr = 1 - tpr
    # Get the count of each unique value
    class_counts = y_test.value_counts()
    idx = np.argmax(np.sqrt(tpr * (1 - fpr)))
    fp = fpr[idx]
    tp = tpr[idx]
    th = thresholds[idx]
    tn = tnr[idx]
    fn = fnr[idx]

    plt.figure(figsize=(6, 6), dpi=300, edgecolor="black")
    plt.plot([0, 1], [0, 1], linestyle=":", lw=1.5, color="r", label="Random Model", alpha=0.8)
    labels = {"0": "Normal", "1": "Buggy"}
    for index, value in enumerate(class_counts.values):
        plt.plot(fp, tp, "*", label=f"{labels[str(index)]} = {value}")
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_scores))
    # print(f"fp: {fp}")

    # Calculate the F1-score for each threshold
    f1 = calculate_f1_score(tp, fp, fn, tn)
    mcc = calculate_mcc(tp, fp, fn, tn)
    plt.plot(fp, tp, "o", label='F1 score = %0.2f' % f1)
    plt.plot(fp, tp, "x", label='MCC = %0.2f' % mcc)
    # Add labels to each bar in the plot

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{project_name}')
    plt.legend(loc="lower right")
    plt.grid(axis='both')
    # plt.colorbar(label='Threshold')
    plt.savefig(f"./figures/TV/OCSVM_{project_name}.png")
    if display:
        plt.show()
    plt.clf()


def get_mean_std(li):
    mean = np.mean(li)
    std = np.std(li)
    return mean, std


def bootstrap(data, num_samples, statistic_function):
    samples = []
    for _ in range(num_samples):
        sample = data.sample(n=len(data), replace=True)
        statistic = statistic_function(sample)
        samples.append(statistic)
    return samples

def start(file_path: str, display=False):
    label = 'isBug'
    project_name = file_path.split('/')[-1].split('_')[0]
    test_size = 0.30

    print('Processing project: ', project_name)

    # 1. Data loading and preprocessing
    top_features = ['NF',
                    'NS',
                    'LT',
                    'LA',
                    'entropy',
                    'AGE',
                    'LD',
                    'NDEV',
                    'NUC',
                    'ND',
                    label]
    use_top_features = True

    if use_top_features:
        dataset = pd.read_csv(file_path)[top_features]
    else:
        dataset = pd.read_csv(file_path)

    scaling_data = True
    scaler = MinMaxScaler()

    training_set_features, testing_set_features, training_set_targets, testing_set_targets = split_random(dataset,
                                                                                                          label,
                                                                                                          test_size,
                                                                                                          True)
    training_set_features = training_set_features.drop(["commit_hex", "bug_id", "commit_time"], axis=1)
    testing_set_features = testing_set_features.drop(["commit_hex", "bug_id", "commit_time"], axis=1)
    class_counts = testing_set_targets.value_counts()
    if scaling_data:
        columns = training_set_features.columns
        training_set_features = scaler.fit_transform(training_set_features)
        training_set_features = pd.DataFrame(training_set_features, columns=columns)

        testing_set_features = scaler.transform(testing_set_features)
        testing_set_features = pd.DataFrame(testing_set_features)
    # Define the range of nu values to test
    contaminations = [0.1, 0.2,0.3,0.4,0.5]
    nus = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.6, 0.70, 0.8, 0.95 ]
    gammas = [0.000001, 0.0001, 0.001, 0.0015, 0.01, 0.1, 0.15, 1.0, 10.0, 100, 1000, 10000]
    training_set_features[label] = training_set_targets
    training_set_features[label].fillna(0, inplace=True)
    tscv = TimeSeriesSplit(n_splits=2)
    local_auc = []
    local_f1 = []
    local_mcc = []
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(6, 6), dpi=300, edgecolor="black")
    plt.plot([0, 1], [0, 1], linestyle=":", lw=1.5, color="r", label="Random Model", alpha=0.8)
    validation_size = 0.20

    x_train, x_validation, y_train, y_validation = split_random(training_set_features, label, validation_size, sort_time=False)
    x_train[label] = y_train
    x_validation[label] = y_validation
    for test_case in range(1,31):
        print(f"*** Sample {test_case} ***",end='\n')
        train_sample = x_train.sample(n=len(x_train), replace=True, random_state=test_case) #bootstrap(data=x_train, num_samples=10, statistic_function=np.mean)#x_train.sample()
        validation_sample = x_validation.sample(n=len(x_validation), replace=True)
        best_model = perform_hyper_turning_search(train_sample, validation_sample,
                                                  nus, gammas,contaminations, scoring='roc_auc')
        # Evaluate the model on the test set
        auc_roc, f1, mcc, fpr, tpr = evaluate_ocsvm(best_model, testing_set_features, testing_set_targets)
        target = 1
        if auc_roc < 0.5:
            auc_roc = 1 - auc_roc
            target = 0
            y_scores = best_model.predict_proba(testing_set_features)[:, target]
            fpr, tpr, thresholds = roc_curve(testing_set_targets, y_scores)
        best_tpr = np.interp(base_fpr, fpr, tpr)
        best_tpr[0] = 0.0
        tprs.append(best_tpr)
        local_auc.append(auc_roc)
        local_f1.append(f1)
        local_mcc.append(mcc)

        # End folds here
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_auc = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std_auc, 1)
    tprs_lower = mean_tprs - std_auc

    mean_auc, std_auc = get_mean_std(local_auc)
    mean_f1, std_f1 = get_mean_std(local_f1)
    mean_mcc, std_mcc = get_mean_std(local_mcc)

    # Find the optimal point to measure F1 and MCC.
    # Get the count of each unique value
    class_counts = testing_set_targets.value_counts()
    idx = np.argmax(np.sqrt(mean_tprs * (1 - base_fpr)))
    fp = base_fpr[idx]
    tp = mean_tprs[idx]
    labels = {"0": "Normal", "1": "Buggy"}
    for index, value in enumerate(class_counts.values):
        plt.plot(fp, tp, "*", label=f"{labels[str(index)]} = {value}")
    plt.plot(
        base_fpr,
        mean_tprs,
        linestyle='--',
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=1.8,
        alpha=0.80,
    )
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot(fp, tp, "o", lw=1.8, label=r'Mean F1 score = (%0.2f $\pm$ %0.2f)' % (mean_f1, std_f1))
    plt.plot(fp, tp, "x", lw=1.8, label=r'Mean MCC = (%0.2f $\pm$ %0.2f)' % (mean_mcc, std_mcc))
    plt.suptitle(f"Project {project_name}")
    plt.legend(loc="lower right")
    plt.grid(axis='both')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(f"./figures/TV/OCSVM_{project_name}_boot.png")
    if display:
        plt.show()
    plt.clf()

    print(f"Best AUC-ROC: {mean_auc} \u00B1 {std_auc}")
    print(f"Best F1-score:  {mean_f1} \u00B1 {std_f1}")
    print(f"Best MCC: {mean_mcc} \u00B1 {std_auc}")
    # Get the count of each unique value
    class_counts = testing_set_targets.value_counts()
    # save data for report export
    projects.append(project_name)
    normals.append(class_counts[0])
    buggies.append(class_counts[1])
    f1_s.append(mean_f1)
    auc_s.append(mean_auc)
    mcc_s.append(mean_mcc)

    f1_ci.append(std_f1)
    mcc_ci.append(std_mcc)
    auc_ci.append(std_auc)


if __name__ == '__main__':
    if not test_mode:
        data_dir = './dataset/ClusterCommitData/'
        code_metrics = [
            "All",
        ]
        # Get the current time before the process starts
        start_time = datetime.datetime.now()
        for code_metric in code_metrics:
            path_to_file = data_dir + code_metric

            for data_file in os.listdir(path_to_file):
                project_name = data_file.split('/')[-1].split('_')[0]
                if project_name == "ambari":
                    start(path_to_file + '/' + data_file, display=False)

            # Export report
            frame = {"Project name": projects,
                     "Normal": normals,
                     "Buggy": buggies,
                     "f1 score": f1_s,
                     "mcc": mcc_s,
                     "auc": auc_s,
                     "f1 score STD": f1_ci,
                     "mcc STD": mcc_ci,
                     "auc STD": auc_ci
                     }
            report = pd.DataFrame(frame)
            report.to_csv(f"./results/final_report_ocsvm_TV_{code_metric}_bootstrap.csv", index=None, header=True)
            print("Collecting...")
            gc.collect()
            # Reset report for next run
            tp_s = []
            tn_s = []
            fp_s = []
            fn_s = []
            precisions = []
            recalls = []
            f1_s = []
            accuracies = []
            mcc_s = []
            auc_s = []
            projects = []
            ir_s = []
            accuracies_ci = []
            f1_ci = []
            mcc_ci = []
            auc_ci = []
            cv_s = []
            mcc_m_validation = []
            mcc_s_validation = []
            auc_m_validation = []
            auc_s_validation = []

        # Get the current time after the process finishes
        end_time = datetime.datetime.now()
        # Calculate the duration of the process
        duration = end_time - start_time

        # Calculate hours, minutes, and seconds from the duration
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the duration as HH:mm:ss
        formatted_duration = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

        # Print the start and end times along with the formatted duration
        print("Process started at:", start_time)
        print("Process ended at:", end_time)
        print("Duration:", formatted_duration)

