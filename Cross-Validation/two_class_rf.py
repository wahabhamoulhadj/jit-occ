# Binary RandomForest applied to the prediction of risky commits
# Authors: Members of the SRT lab at Concordia University

import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import warnings

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

mcc_s_validation = []  # MCC measurement
auc_s_validation = []  # AUC measurement
mcc_m_validation = []  # MCC measurement
auc_m_validation = []  # AUC measurement

mcc_ci = []
auc_ci = []
f1_ci = []
accuracies_ci = []
feature_names_dic_global = {}
test_mode = False

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


def split_random(dataset, label, test_size, random_state=None):
    features = dataset.drop([label], axis=1)
    targets = dataset[label]
    x_train, x_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=test_size,
                                                        # shuffle=True,
                                                        stratify=targets,
                                                        random_state=random_state
                                                        )
    return x_train, x_test, y_train, y_test


def balancing(df, label, method="down"):
    if df is None:
        print("- Data set is not found.")
        raise
    df_fix = df[df[label] == 0]
    df_bug = df[df[label] == 1]

    if method == 'over':
        # over sampling process
        # print("+ Over-sampling.")
        # logging.debug("+ Over-sampling.")

        if len(df_fix) < len(df_bug):
            df_fix = df_fix.sample(len(df_bug), replace=True)
        elif len(df_fix) >= len(df_bug):
            df_bug = df_bug.sample(len(df_fix), replace=True)
        df = pd.concat([df_bug, df_fix])
    elif method == 'down':
        # down sampling method
        # logging.debug("+ Down-sampling.")
        # print("+ Down-sampling.")
        if len(df_fix) > len(df_bug):
            df_fix = df_fix.sample(len(df_bug), replace=True)
        elif len(df_fix) <= len(df_bug):
            df_bug = df_bug.sample(len(df_fix), replace=True)
        df = pd.concat([df_bug, df_fix])
    elif method=="SMOTE":
        print("+ Over-sampling (SMOTE).")
        oversample = SMOTE()
        X = df.drop([label], axis=1)
        y = df[label]
        df, y = oversample.fit_resample(X, y)
        df[label] = y
    else:
        raise "- Exception: please set the balancing method (over, down or SMOTE)"
    return df


def start(file_path: str, balancing_method=None):
    label = 'isBug'
    test_size = 0.30
    project_name = file_path.split('/')[-1]

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

    dataset = pd.read_csv(file_path)
    dataset = dataset.drop(["commit_hex", "commit_time", "bug_id"],  axis=1)
    if use_top_features:
        dataset = dataset[top_features]

    healthy_commits = dataset[dataset[label] == 0]  # Extract healthy commits from the dataset
    buggy_commits = dataset[dataset[label] == 1]  # Extract buggy commits from the dataset
    target_class = 1
    scaling_data = True
    scaler = MinMaxScaler()

    # Local variables to measure the average of all model performance measurements
    local_tp_s = []  # True positive measurement
    local_tn_s = []  # True negative measurement
    local_fp_s = []  # False positive measurement
    local_fn_s = []  # False negative measurement
    local_precisions = []  # Precision measurement
    local_recalls = []  # Recall measurement
    local_f1_s = []  # F1 score measurement
    local_accuracies = []  # Accuracy measurement
    local_mcc_s = []  # MCC measurement
    local_auc_s = []  # AUC measurement

    for test_case in range(1, 31):  # no need for this

        # Calculate the imbalanced
        ir = round((len(healthy_commits) / len(buggy_commits)), 2)
        cross_validation = 10
        if ir > 8:
            cross_validation = 5

        # 2. Build training, validation, and testing sets

        training_set_features, testing_set_features, training_set_targets, testing_set_targets = split_random(
            dataset=dataset,
            label=label,
            test_size=test_size,
            random_state=test_case)
        feature_names = training_set_features.columns
        if balancing_method is not None:
            training_set_features[label] = training_set_targets
            training_set_features = balancing(df=training_set_features, label=label, method=balancing_method)
            # 2.2 Creating features and target labels
            training_set_targets = training_set_features[label]
            training_set_features = training_set_features.drop([label], axis=1)

        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        # here we need to run it 10 times
        # plt.figure(figsize=(10, 10), dpi=300, edgecolor="black")
        # plt.plot([0, 1], [0, 1], linestyle=":", lw=2, color="r", label="Random Model", alpha=0.8)

        kf = StratifiedKFold(n_splits=cross_validation)

        # 3. Parameter tuning using validation sets
        tuning_features = training_set_features
        tuning_targets = training_set_targets
        best_mcc = -1
        best_auc = -1
        best_samples = 1
        # best_max_depth = 1
        fold_id = 1
        fold_auc_s = []
        fold_aucs = []
        fold_mccs = []
        feature_names_dic = {}
        # 3.2 Cross-validation test different healthy folds, while risky is only for validation.
        for train_index, validate_index in kf.split(tuning_features, tuning_targets):
            train_commits_features, validation_commits_features = tuning_features.iloc[train_index], \
                                                                  tuning_features.iloc[
                                                                      validate_index]
            train_commits_targets, validate_commits_targets = tuning_targets.iloc[train_index], tuning_targets.iloc[
                validate_index]
            # train_commits_features = tuning_targets.drop([label], axis=1)
            # 4. Train a kNN detector
            if scaling_data:
                train_commits_features = scaler.fit_transform(train_commits_features)
            model_name = 'Random Forest Classifier'
            fold_samples = 1
            # fold_max_depth = 1
            mcc_fold = -1
            auc_fold = -1
            best_fpr = best_tpr = [0]
            max_sample = 17
            # if max_sample >= 35:
            #     max_sample = 35
            for samples in range(1, max_sample, 1):
                # for depth in range(2, 15, 1):
                etc = RandomForestClassifier(max_depth=samples, n_estimators=100, n_jobs=-1)
                # Train model with healthy only
                etc.fit(train_commits_features, train_commits_targets)
                predictions = etc.predict(validation_commits_features)
                predictions_proba = etc.predict_proba(validation_commits_features)
                mcc = metrics.matthews_corrcoef(validate_commits_targets, predictions)
                # NOTE: the roc_curve take data label not the predictions, with the probability of the predictions.
                fpr, tpr, thresholds = metrics.roc_curve(validate_commits_targets, predictions_proba[:, target_class],
                                                         pos_label=target_class)

                auc = metrics.auc(fpr, tpr)
                # This to get best samples for each fold
                if mcc_fold < mcc:
                    mcc_fold = mcc
                    fold_samples = samples
                    # fold_max_depth = depth
                    best_fpr = fpr
                    best_tpr = tpr
                    auc_fold = auc
                # This to get best samples over all folds
                if best_mcc < mcc:
                    best_mcc = mcc
                    best_samples = samples
                    # best_max_depth = depth
                    best_auc = auc

                # End test k values
            feature_importance_normalized = np.mean([tree.feature_importances_ for tree in
                                                     etc.estimators_],
                                                    axis=0)
            # print("\n--------------Feature importance of best model--------------")
            for importance, name in sorted(zip(feature_importance_normalized, feature_names), reverse=True)[:10]:
                if name in feature_names_dic:
                    feature_names_dic[name] += importance
                else:
                    feature_names_dic.update({name: importance})

                if name in feature_names_dic_global:
                    feature_names_dic_global[name] += importance
                else:
                    feature_names_dic_global.update({name: importance})
            fold_auc_s.append(best_auc)
            fold_id = fold_id + 1
            fold_aucs.append(best_auc)
            fold_mccs.append(best_mcc)
            best_tpr = np.interp(base_fpr, best_fpr, best_tpr)
            best_tpr[0] = 0.0
            tprs.append(best_tpr)
        print(f"Best best max depth {best_samples}, MCC = {best_mcc} and AUC {best_auc}")

        # AUC
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        auc_s_validation.append(std_auc)
        auc_m_validation.append(mean_auc)

        # MCC
        mean_mcc = np.mean(fold_mccs)
        std_mcc = np.std(fold_mccs)
        mcc_s_validation.append(std_mcc)
        mcc_m_validation.append(mean_mcc)

        if scaling_data:
            tuning_features = scaler.fit_transform(tuning_features)
            testing_set_features = scaler.transform(testing_set_features)
        sorted_dic = sorted(feature_names_dic.items(), key=lambda item: item[1], reverse=True)
        # 5. Train a RandomForestClassifier
        etc = RandomForestClassifier(max_samples=best_samples, n_jobs=-1)
        print(f'Local features for project {project_name}\n{sorted_dic}')
        # training_set_features[label] = training_set_targets.values
        # healthy_commits = training_set_features[training_set_features[label] == 0].drop(label, axis=1)
        # We need to train the best model
        etc.fit(tuning_features, tuning_targets)
        # 5. Test the model using 20% of healthy commits and 90% of buggy commits
        predictions = etc.predict(testing_set_features)
        predictions_proba = etc.predict_proba(testing_set_features)
        # 6. Outputting the results

        tn, fp, fn, tp = confusion_matrix(testing_set_targets, predictions).ravel()
        local_tp_s.append(tp)
        local_fp_s.append(fp)
        local_tn_s.append(tn)
        local_fn_s.append(fn)
        # Calculate MCC. Ref:# https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7
        mcc = metrics.matthews_corrcoef(testing_set_targets, predictions)

        local_mcc_s.append(mcc)
        # Calculate AUC
        # NOTE: the roc_curve take data label not the predictions, with the probability of the predictions.

        fpr, tpr, thresholds = metrics.roc_curve(testing_set_targets, predictions_proba[:, target_class],
                                                 pos_label=target_class)
        auc = metrics.auc(fpr, tpr)
        print(f"Test {test_case} results = MCC = {mcc}, AUC = {auc}")

        local_auc_s.append(metrics.auc(fpr, tpr))
        # Calculate f1
        local_f1_s.append(metrics.f1_score(testing_set_targets, predictions))
        # Calculate accuracy
        local_accuracies.append(metrics.accuracy_score(testing_set_targets, predictions))
        local_precisions.append(metrics.precision_score(testing_set_targets, predictions))
        local_recalls.append(metrics.recall_score(testing_set_targets, predictions))

    # end loop of 10 tests

    # Export test set.
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_auc = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std_auc, 1)
    tprs_lower = mean_tprs - std_auc
    mean_auc = np.mean(fold_auc_s)
    std_auc = np.std(fold_auc_s)

    testing_set_features[label] = testing_set_targets.values
    testing_set_features.to_csv(f"./test_set_{project_name}", index=None, header=True)

    # Update the final report
    projects.append(project_name)
    tp_s.append(np.mean(local_tp_s))
    fp_s.append(np.mean(local_fp_s))
    fn_s.append(np.mean(local_fn_s))
    tn_s.append(np.mean(local_tn_s))
    precisions.append(np.mean(local_precisions))
    recalls.append(np.mean(local_recalls))
    f1_s.append(np.mean(local_f1_s))
    accuracies.append(np.mean(local_accuracies))
    auc_s.append(np.mean(local_auc_s))
    mcc_s.append(np.mean(local_mcc_s))
    ir_s.append(ir)
    cv_s.append(cross_validation)

    print(f"MCC (mean) = {np.mean(local_mcc_s)}, AUC (mean) = {np.mean(local_auc_s)}")
    # measure the CI of 10 test cases
    f1_std = np.std(local_f1_s)
    auc_std = np.std(local_auc_s)
    mcc_std = np.std(local_mcc_s)
    accuracy_std = np.std(local_accuracies)
    #  show the 95% confidence intervals around the median.
    f1_CI = confidence_interval["95"] * (f1_std / np.sqrt(len(local_f1_s)))
    auc_CI = confidence_interval["95"] * (auc_std / np.sqrt(len(local_auc_s)))
    mcc_CI = confidence_interval["95"] * (mcc_std / np.sqrt(len(local_mcc_s)))
    accuracies_CI = confidence_interval["95"] * (accuracy_std / np.sqrt(len(local_accuracies)))
    f1_ci.append(f1_CI)
    auc_ci.append(auc_CI)
    mcc_ci.append(mcc_CI)
    accuracies_ci.append(accuracies_CI)


if __name__ == '__main__':
    if not test_mode:
        balance = None
        data_dir = './datasets/'
        code_metrics = [
            'All/',
        ]
        feature_names_dic_global = {}
        for code_metric in code_metrics:
            path_to_file = data_dir + code_metric
            # run 10 times (10 test cases)
            for data_file in os.listdir(path_to_file):
                start(path_to_file + '/' + data_file, balancing_method=balance)

            # Export report
            frame = {"Project name": projects,
                     "tp": tp_s,
                     "tn": tn_s,
                     "fp": fp_s,
                     "fn": fn_s,
                     "imbalanced ratio": ir_s,
                     "cross-validation": cv_s,
                     "precision": precisions,
                     "recall": recalls,
                     "accuracy": accuracies,
                     "f1 score": f1_s,
                     "mcc": mcc_s,
                     "auc": auc_s,
                     "accuracy CI": accuracies_ci,
                     "f1 CI": f1_ci,
                     "mcc CI": mcc_ci,
                     "auc CI": auc_ci,
                     }
            report = pd.DataFrame(frame)
            report.to_csv(f"./results/final_report_RF_Binary_{code_metric}_{balance}.csv", index=None, header=True)

            frame_validaion ={"Project name": projects,
                              "mcc_mean": mcc_m_validation,
                              "mcc_std": mcc_s_validation,
                              "auc_mean":auc_m_validation,
                              "auc_std": auc_s_validation
                              }
            report = pd.DataFrame(frame_validaion)
            report.to_csv(f"./results/final_validation_report_RF_Binary_{code_metric}.csv", index=None, header=True)
            print("Collecting...")
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
            mcc_m_validation=[]
            mcc_s_validation=[]
            auc_m_validation=[]
            auc_s_validation =[]
            sorted_dic = sorted(feature_names_dic_global.items(), key=lambda item: item[1], reverse=True)
            print(f'Global features of all projects \n{feature_names_dic_global}')
            n_items = list(sorted_dic)[:10]
            print(n_items)
    else:

        start('dataset/CK_NET_PROC/jedit-4.3--CK_NET_PROC.csv', balancing_method=None)
    exit(0)
