import json

def save_results(results, path):
    results_str = {str(k): v for k, v in results.items()}
    with open(path, 'w') as f:
        json.dump(results_str, f)

def load_results(path, key_type=float):
    with open(path, 'r') as f:
        data = json.load(f)
    return {key_type(k): {int(s): v for s, v in seed_dict.items()}
            for k, seed_dict in data.items()}