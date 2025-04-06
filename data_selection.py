import json
import numpy as np
import os
import pandas as pd
import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., CodeBERT)", default="UniXcoder")
    parser.add_argument("--seed", type=str, required=True, help="Seed value (e.g., 42)", default="42")
    args = parser.parse_args()

    model_name = args.model
    seed = args.seed

    data_path = 'dataset/csn/train.jsonl'
    td_dir = os.path.join('save', model_name, seed, 'training_dynamics')
    subset_dir = os.path.join('save', model_name, seed, 'subsets')
    os.makedirs(subset_dir, exist_ok=True)

    data = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)

    def read_training_dynamics():
        id_field = "guid"
        train_dynamics = {}
        num_epochs = len([f for f in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, f))])

        for epoch_num in tqdm.tqdm(range(num_epochs)):
            epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.jsonl")
            assert os.path.exists(epoch_file)

            with open(epoch_file, "r") as infile:
                for line in infile:
                    record = json.loads(line.strip())
                    guid = record[id_field]
                    if guid[:7] == "tensor(":
                        guid = guid[7:-1]

                    if guid not in train_dynamics:
                        train_dynamics[guid] = {"probability": []}    
                    train_dynamics[guid]["probability"].append(record[f"prob_epoch_{epoch_num}"])

        return train_dynamics

    uncertainty_records = read_training_dynamics()

    def compute_train_dy_metrics(training_dynamics):
        confidence_ = {}
        variability_ = {}

        variability_func = lambda conf: np.std(conf)

        for guid in tqdm.tqdm(training_dynamics):
            record = training_dynamics[guid]
            confidence_[guid] = np.mean(record['probability'])
            variability_[guid] = variability_func(record['probability'])

        df = pd.DataFrame({
            'guid': list(confidence_.keys()),
            'mean_confidence': list(confidence_.values()),
            'variability': list(variability_.values())
        })

        return df

    uncertainty_metrics = compute_train_dy_metrics(uncertainty_records)

    def filter_data(df, metric, percentage, ascending=True):
        sorted_df = df.sort_values(by=[metric], ascending=ascending)
        num_selected = int(percentage * len(sorted_df))
        return sorted_df.iloc[:num_selected]

    hard_to_learn = filter_data(uncertainty_metrics, 'mean_confidence', 0.33, ascending=True)
    ambiguous = filter_data(uncertainty_metrics, 'variability', 0.33, ascending=False)
    easy_to_learn = filter_data(uncertainty_metrics, 'mean_confidence', 0.33, ascending=False)
    easy_to_learn = easy_to_learn[easy_to_learn['variability'] <= easy_to_learn['variability'].median()]
    hard_ambiguous = pd.concat([hard_to_learn, ambiguous]).drop_duplicates()

    datasets = {
        "hard_to_learn.jsonl": hard_to_learn,
        "ambiguous.jsonl": ambiguous,
        "easy_to_learn.jsonl": easy_to_learn,
        "hard_ambiguous.jsonl": hard_ambiguous,
    }

    for filename, df in datasets.items():
        selected_guid = set(df['guid'])
        selected_data = [d for d in data if str(d['url']) in selected_guid]

        output_path = os.path.join(subset_dir, filename)
        with open(output_path, 'w') as outfile:
            for entry in selected_data:
                json.dump(entry, outfile)
                outfile.write('\n')
        print(f"Saved {output_path} with {len(selected_data)} samples.")

if __name__ == "__main__":
    main()
