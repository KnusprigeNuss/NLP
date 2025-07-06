import csv

def load_labels(file_path):
    labels = {}
    with open(file_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row['index'])
            label = row['real_news'].strip().lower()
            labels[index] = label
    return labels

def compare_files(ground_truth_path, prediction_path, total_entries=150):
    ground_truth = load_labels(ground_truth_path)
    predictions = load_labels(prediction_path)

    correct = 0
    ambig = 0
    for i in range(1, total_entries + 1):
        gt_label = ground_truth.get(i)
        pred_label = predictions.get(i)
        if pred_label == "":
            pred_label = "yes"
            ambig += 1
        if gt_label == pred_label:
            correct += 1

    print(f"Correct predictions: {correct} out of {total_entries}")
    print(f"Accuracy: {correct / total_entries:.2%}")
    print(f"Ambigious: {ambig}")
    print(f"Ambigious percent: {ambig / total_entries:.2%}")

if __name__ == '__main__':
    ground_truth_file = 'group44_stage1.csv'
    predictions_file = 'group44_stage3.csv'
    compare_files(ground_truth_file, predictions_file)
