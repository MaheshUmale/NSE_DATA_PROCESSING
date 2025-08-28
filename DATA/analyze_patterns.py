import numpy as np
import pandas as pd
from collections import defaultdict, Counter
def default_pattern_stat():
    return {"counts": Counter(), "total": 0, "timestamps": []}



def analyze_pattern_accuracy_full(
    df,
    seq_len=5,
    future_steps=1,
    min_occurrences=5,
    top_n_visualize=10,
    plot=False
):
    df = df.copy()

    # Encode candles by normalized body size
    df['bodysize'] = df['Close'] - df['Open']
    atr = df['bodysize'].abs().rolling(window=14).mean()
    df['size_category'] = (df['bodysize'] / atr).fillna(0)
    df['size_category'] = df['size_category'].apply(
        lambda x: 0 if abs(x) < 0.5 else int(np.sign(x) * min(abs(x), 3))
    )

    # Encode volume into categories
    vol_mean = df['Volume'].mean()
    vol_std = df['Volume'].std()
    df['volume_category'] = ((df['Volume'] - vol_mean) / vol_std).apply(
        lambda x: 1 if x < -0.5 else (3 if x > 0.5 else 2)
    )

    # Sequence = (body_cat, volume_cat)
    sequence = list(zip(df['size_category'], df['volume_category']))
    timestamps = df.index.tolist()

    total_len = seq_len + future_steps
    ngram_data = [
        tuple(sequence[i:i + total_len])
        for i in range(len(sequence) - total_len + 1)
    ]
    ngram_timestamps = [
        tuple(timestamps[i:i + total_len])
        for i in range(len(sequence) - total_len + 1)
    ]

    # pattern_stats = defaultdict(lambda: {"counts": Counter(), "total": 0, "timestamps": []})
    pattern_stats = defaultdict(default_pattern_stat)
    # OUT_pattern_stats = defaultdict(default_pattern_stat)
    results = []
    for ngram_seq, ts_seq in zip(ngram_data, ngram_timestamps):
        pattern = tuple(ngram_seq[:seq_len])
        actual_future = ngram_seq[-future_steps]
        future_timestamp = ts_seq[-future_steps]

        pattern_stats[pattern]["counts"][actual_future] += 1
        pattern_stats[pattern]["total"] += 1
        pattern_stats[pattern]["timestamps"].append(future_timestamp)

    # results = []
    # for pattern, data in pattern_stats.items():
        total = pattern_stats[pattern]["total"]
        if total < min_occurrences:
            continue

        most_common, count = pattern_stats[pattern]["counts"].most_common(1)[0]
        direction_accuracy = count / total

        predicted_dir = np.sign(most_common[0])
        direction_counts = sum(
            v for actual, v in pattern_stats[pattern]["counts"].items()
            if np.sign(actual[0]) == predicted_dir
        )
        direction_only_accuracy = direction_counts / total

        results.append({
            "pattern": pattern,
            "predicted": most_common,
            "accuracy": round(direction_accuracy * 100, 2),
            "direction_only_accuracy": round(direction_only_accuracy * 100, 2),
            "predicted_direction": predicted_dir,
            "predicted_volume_cat": most_common[1],
            "correct": count,
            "total": total,
            "timestamps": pattern_stats[pattern]["timestamps"]
        })
        # if direction_accuracy > 0.07 :
        #     # print("added pattern -- ")
        #     OUT_pattern_stats[pattern]["counts"][actual_future] += 1
        #     OUT_pattern_stats[pattern]["total"] += 1
        #     OUT_pattern_stats[pattern]["timestamps"].append(future_timestamp) 


    result_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    # print("-------------FINAL OUT_pattern_stats --------------------")
    # print(OUT_pattern_stats)
    return result_df, pattern_stats
