import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df_counts, title, filename):
    """Generates and saves a bar plot for the label distribution (counts)."""
    if df_counts is None or df_counts.empty:
        print(f"Skipping plot generation for '{title}': No data.")
        return

    plt.figure(figsize=(10, 6))
    # Plotting counts is generally clearer for direct comparison
    sns.barplot(x='Count', y='Group Activity Label', data=df_counts, palette='viridis', hue='Group Activity Label', dodge=False, legend=False)
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Group Activity Label')
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close() # Close the plot to free memory

def analyze_group_activity(base_path, folder_ids):
    """
    Analyzes the distribution of group activity labels in annotation files.
    Looks for 'annotations.txt' directly within each specified folder ID directory.

    Args:
        base_path (str): The base directory containing the video folders (e.g., 'data/videos').
        folder_ids (list): A list of folder IDs (integers) to process.

    Returns:
        dict: A dictionary containing label counts for each annotation line processed.
              Keys are tuples (folder_id, line_index), values are group_label.
              Returns None if no annotation files are found or accessible.
    """
    all_annotations = {} # Store label per annotation line: {(folder_id, line_idx): label}
    processed_folders = 0
    found_files_count = 0

    for folder_id in folder_ids:
        current_folder_id = int(folder_id)
        folder_path = os.path.join(base_path, str(current_folder_id))
        annotation_file_path = os.path.join(folder_path, "annotations.txt")

        processed_folders += 1

        if os.path.exists(annotation_file_path):
            found_files_count += 1
            try:
                with open(annotation_file_path, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            group_label = parts[1]
                            all_annotations[(current_folder_id, i)] = group_label
                        else:
                            print(f"Warning: Skipping malformed line in {annotation_file_path}: {line.strip()}")
            except Exception as e:
                print(f"Error reading {annotation_file_path}: {e}")
        else:
            pass # Continue silently if file not found

    print(f"\nProcessed {processed_folders} folders (IDs: {min(folder_ids)}-{max(folder_ids)}).")
    print(f"Found and processed {found_files_count} 'annotations.txt' files.")
    total_parsed = len(all_annotations)
    print(f"Total annotations parsed: {total_parsed}")

    if not all_annotations:
        print("No annotations found or parsed.")
        return None, 0

    return all_annotations, total_parsed

def get_distribution_df(annotations, relevant_folder_ids=None):
    """
    Calculates label distribution (count and percentage) for a subset of annotations.
    """
    counts = Counter()
    subset_annotations = []
    if relevant_folder_ids is None: # Calculate for all
        subset_annotations = list(annotations.values())
        counts = Counter(subset_annotations)
    else:
        relevant_ids_set = set(relevant_folder_ids)
        for (folder_id, _), label in annotations.items():
            if folder_id in relevant_ids_set:
                 subset_annotations.append(label)
        counts = Counter(subset_annotations)


    if not counts:
        return None, 0

    total_count_subset = len(subset_annotations)
    df_counts = pd.DataFrame(counts.items(), columns=['Group Activity Label', 'Count'])
    # Calculate Percentage
    df_counts['Percentage'] = df_counts['Count'].apply(lambda x: f"{(x / total_count_subset) * 100:.2f}%" if total_count_subset > 0 else "0.00%")
    df_counts = df_counts.sort_values(by='Count', ascending=False).reset_index(drop=True)

    return df_counts, total_count_subset


if __name__ == "__main__":
    DATA_BASE_PATH = "data/videos"
    ALL_FOLDER_IDS = list(range(55)) # Folders 0 to 54

    # Provided Train/Validation IDs
    TRAIN_IDS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    VAL_IDS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

    print(f"Starting EDA for group activity labels in folders 0-54 within '{DATA_BASE_PATH}'...")
    print("Searching for 'annotations.txt' directly within each folder ID directory.")

    # 1. Get all annotations first
    all_annotations_data, total_overall_annotations = analyze_group_activity(DATA_BASE_PATH, ALL_FOLDER_IDS)

    if all_annotations_data:
        # 2. Calculate and plot Overall distribution
        print(f"\n--- Overall Distribution (Total: {total_overall_annotations}) ---")
        df_overall, _ = get_distribution_df(all_annotations_data)
        if df_overall is not None:
            print(df_overall.to_string(index=False))
            plot_distribution(df_overall, 'Overall Group Activity Label Distribution', 'overall_distribution.png')
        else:
            print("No data for overall distribution.")

        # 3. Calculate and plot Training set distribution
        print(f"\n--- Training Set Distribution (Folders: {len(TRAIN_IDS)}) ---")
        df_train, total_train_annotations = get_distribution_df(all_annotations_data, TRAIN_IDS)
        if df_train is not None:
            print(f"(Total Annotations in Train Set: {total_train_annotations})")
            print(df_train.to_string(index=False))
            plot_distribution(df_train, f'Training Set (IDs: {len(TRAIN_IDS)}) Group Activity Label Distribution', 'train_distribution.png')
        else:
            print("No data for training set distribution.")

        # 4. Calculate and plot Validation set distribution
        print(f"\n--- Validation Set Distribution (Folders: {len(VAL_IDS)}) ---")
        df_val, total_val_annotations = get_distribution_df(all_annotations_data, VAL_IDS)
        if df_val is not None:
            print(f"(Total Annotations in Validation Set: {total_val_annotations})")
            print(df_val.to_string(index=False))
            plot_distribution(df_val, f'Validation Set (IDs: {len(VAL_IDS)}) Group Activity Label Distribution', 'validation_distribution.png')
        else:
            print("No data for validation set distribution.")

    print("\nEDA finished.")