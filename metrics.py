import uproot
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Let's inspect one of the prediction files to understand its structure.
sample_root_file = r"C:\Users\henti\ParT\particle_transformer\training\JetClass\Pythia\full\ParT\20250504-161729_example_ParticleTransformer_ranger_lr0.001_batch512\predict_output\pred_HToBB.root" # <<<--- UPDATE THIS PATH

print(f"Inspecting file: {sample_root_file}")
try:
    with uproot.open(sample_root_file) as file:
        print("\nFile contents (keys):")
        for key in file.keys():
            print(f"- {key}: {file[key].classname}")

        tree_name = 'Events' 

        if tree_name in file:
            print(f"\nInspecting Tree: {tree_name}")
            tree = file[tree_name]
            print("Tree branches:")
            for branch in tree.keys():
                print(f"- {branch}")
            print("\n")
        else:
            print(f"\nError: Tree '{tree_name}' not found in the file.")
            print("Please manually inspect the file structure to find the correct tree name.")
            tree_name = None 


except FileNotFoundError:
    print(f"Error: Sample file not found at '{sample_root_file}')")
except Exception as e:
    print(f"An error occurred during file inspection: {e}")

class_names = [
    'QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl'
]

file_class_parts = [
    'HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q',
    'TTBar', 'TTBarLep', 'WToQQ', 'ZJetsToNuNu', 'ZToQQ'
]

# Define the pattern for your ROOT files. Use {} as a placeholder for the file class part.
file_pattern = r"C:\Users\henti\ParT\particle_transformer\training\JetClass\Pythia\full\ParT\20250504-161729_example_ParticleTransformer_ranger_lr0.001_batch512\predict_output/pred_{}.root" # <<<--- UPDATE THIS PATTERN

score_branches = [
    'score_label_QCD', 'score_label_Hbb', 'score_label_Hcc', 'score_label_Hgg',
    'score_label_H4q', 'score_label_Hqql', 'score_label_Zqq', 'score_label_Wqq',
    'score_label_Tbqq', 'score_label_Tbl'
]

label_branch_name = '_label_' 

all_true_labels = [] 
all_predicted_scores_full = [] 

print("\nLoading data from ROOT files...")

if tree_name is None:
    print("Cannot load data because tree name was not identified during inspection.")
else:
    for file_class_part in file_class_parts:
        file_path = file_pattern.format(file_class_part)
        print(f"Loading data from {file_path}")

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue

        try:
            with uproot.open(file_path) as file:
                if tree_name not in file:
                    print(f"Error: Tree '{tree_name}' not found in file '{file_path}'. Skipping file.")
                    continue

                tree = file[tree_name]

                if label_branch_name not in tree:
                    print(f"Error: Branch '{label_branch_name}' not found in tree '{tree_name}' in file '{file_path}'. Cannot determine true labels. Skipping file.")
                    continue
                true_labels = tree[label_branch_name].array(library="np")

                current_file_scores = []
                missing_score_branch = False
                for branch_name in score_branches:
                    if branch_name not in tree:
                        print(f"Error: Score branch '{branch_name}' not found in tree '{tree_name}' in file '{file_path}'. Cannot load all scores. Skipping file.")
                        missing_score_branch = True
                        break
                    scores = tree[branch_name].array(library="np")
                    current_file_scores.append(scores)

                if missing_score_branch:
                    continue 

                if not current_file_scores:
                     print(f"Warning: No score data loaded from file '{file_path}'. Skipping.")
                     continue


                predicted_scores_full = np.vstack(current_file_scores).T 

                all_true_labels.append(true_labels)
                all_predicted_scores_full.append(predicted_scores_full)

                print(f"  Loaded {len(true_labels)} events.")


        except Exception as e:
            print(f"An error occurred while processing file {file_path}: {e}")


# Concatenate data from all files
if not all_true_labels:
    print("\nNo data loaded successfully. Please check file paths, tree name, branch names, and file_class_parts.")
else:
    combined_true_labels = np.concatenate(all_true_labels)
    combined_predicted_scores_full = np.concatenate(all_predicted_scores_full)

    print("\nCalculating metrics...")

    # Accuracy
    predicted_class_indices = np.argmax(combined_predicted_scores_full, axis=1)

    accuracy = accuracy_score(combined_true_labels, predicted_class_indices)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(combined_true_labels, predicted_class_indices)

    print("\nConfusion Matrix:")
    print("True \\ Predicted")
    print(" | ".join(class_names))
    for i, row in enumerate(conf_matrix):
        print(f"{class_names[i]} | " + " | ".join(map(str, row)))


    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # AUC (One-vs-Rest)
    print("\nCalculating One-vs-Rest AUC for each class:")

    auc_values = [] 

    for i, class_name in enumerate(class_names):
        binary_true_labels = (combined_true_labels == i).astype(int)
        class_scores = combined_predicted_scores_full[:, i]

        if len(np.unique(binary_true_labels)) < 2:
             print(f"  Skipping AUC for {class_name} vs Rest: Only one class present in true labels.")
             continue

        try:
            auc = roc_auc_score(binary_true_labels, class_scores)
            print(f"  AUC for {class_name} vs Rest: {auc:.4f}")
            auc_values.append(auc) 

        except ValueError as e:
            print(f"  Could not calculate AUC for {class_name} vs Rest: {e}")

    # Calculate and print the average AUC
    if auc_values:
        average_auc = np.mean(auc_values)
        print(f"\nAverage AUC across all classes: {average_auc:.4f}")
    else:
        print("\nNo AUC values were calculated.")


# --- Signal vs Background (QCD) Background Rejection ---
    print("\nCalculating Background Rejection for each signal class vs QCD background:")

    # Define target signal efficiencies
    target_signal_efficiencies = {
        'Tbl': 0.995,
        'Hqql': 0.99,
        # All other classes will use the default target
    }
    default_signal_efficiency_target = 0.50 # 50% for most classes

    try:
        qcd_class_index = class_names.index('QCD')
    except ValueError:
        print("Error: 'QCD' class not found in class_names. Cannot calculate signal vs QCD rejection.")
        qcd_class_index = None 

    if qcd_class_index is not None:
        for i, signal_class_name in enumerate(class_names):
            if signal_class_name == 'QCD':
                print(f"  Skipping background rejection for {signal_class_name} (it is the background).")
                continue

            signal_vs_qcd_indices = np.where(
                np.logical_or(combined_true_labels == i, combined_true_labels == qcd_class_index)
            )[0]

            if len(signal_vs_qcd_indices) == 0:
                print(f"  Skipping rejection for {signal_class_name} vs QCD: No relevant events found.")
                continue

            binary_true_labels = (combined_true_labels[signal_vs_qcd_indices] == i).astype(int) 

            scores_S = combined_predicted_scores_full[signal_vs_qcd_indices, i]
            scores_B = combined_predicted_scores_full[signal_vs_qcd_indices, qcd_class_index]

            denominator = scores_S + scores_B
            binary_scores_SvsB = np.zeros_like(scores_S, dtype=float) 
            
            valid_denominator_mask = denominator > 1e-9 
            
            binary_scores_SvsB[valid_denominator_mask] = scores_S[valid_denominator_mask] / denominator[valid_denominator_mask]
            
            if len(np.unique(binary_true_labels)) < 2:
                print(f"  Skipping rejection for {signal_class_name} vs QCD: Only one class present in selected events for ROC.")
                continue

            try:
                fpr, tpr, thresholds = roc_curve(binary_true_labels, binary_scores_SvsB)

                signal_efficiency_target = target_signal_efficiencies.get(signal_class_name, default_signal_efficiency_target)

                indices_above_target_tpr = np.where(tpr >= signal_efficiency_target)[0]

                if len(indices_above_target_tpr) == 0:
                    max_tpr_achieved = tpr.max() if len(tpr) > 0 else 0
                    print(f"  Could not reach {signal_efficiency_target*100:.1f}% signal efficiency for {signal_class_name} vs QCD. Max TPR achieved: {max_tpr_achieved*100:.1f}%. Skipping rejection calculation.")
                    continue

                relevant_fprs = fpr[indices_above_target_tpr]
                
                min_fpr_index_in_subset = np.argmin(relevant_fprs)
                
                best_idx = indices_above_target_tpr[min_fpr_index_in_subset]

                false_positive_rate = fpr[best_idx]
                actual_tpr_at_best_idx = tpr[best_idx] 

                # Handle the case where FPR is zero (infinite rejection)
                if false_positive_rate < 1e-9: 
                    bg_rejection_display = "Infinite"
                    print(f"  Background Rejection for {signal_class_name} vs QCD at actual TPR {actual_tpr_at_best_idx*100:.1f}% (target {signal_efficiency_target*100:.1f}%): {bg_rejection_display} (FPR ~ 0)")
                else:
                    bg_rejection = 1.0 / false_positive_rate
                    bg_rejection_rounded = round(bg_rejection)
                    print(f"  Background Rejection for {signal_class_name} vs QCD at actual TPR {actual_tpr_at_best_idx*100:.1f}% (target {signal_efficiency_target*100:.1f}%): {bg_rejection_rounded:.0f} (FPR: {false_positive_rate:.4e})")


            except ValueError as e:
                print(f"  Could not calculate rejection for {signal_class_name} vs QCD: {e}")
            except Exception as e:
                print(f"  An unexpected error occurred during rejection calculation for {signal_class_name} vs QCD: {e}")