import numpy as np
import json
import os
import random

if __name__ == '__main__':
    filename = os.path.join("e://data/aic_datasets", "predictions.json")
    cnt = 0
    with open(filename, encoding="utf-8") as fh:
        examples = json.load(fh)
        examples_wrong = []
        examples_right = []
        for idx, example in enumerate(examples):
            if example["predict"] != example["orig_answer"]:
                examples_wrong.append(example)
                cnt += 1
            else:
                examples_right.append(example)

    print("wrong num %d" % cnt)
    filename_wrong = os.path.join("e://data/aic_datasets", "pred_wrong.json")
    with open(filename_wrong, encoding="utf-8", mode="w") as fh:
        json.dump(examples_wrong, fh, ensure_ascii=False)

    print("random pick 100 wrong exmaples")
    filename_wrong_random = os.path.join("e://data/aic_datasets", "pred_wrong_100_2.json")
    with open(filename_wrong_random, encoding="utf-8", mode="w") as fh:
        random.shuffle(examples_wrong)
        json.dump(examples_wrong[:100], fh, ensure_ascii=False)

    print("random pick 100 right exmaples")
    filename_wrong_random = os.path.join("e://data/aic_datasets", "pred_right.json")
    with open(filename_wrong_random, encoding="utf-8", mode="w") as fh:
        random.shuffle(examples_right)
        json.dump(examples_right[:100], fh, ensure_ascii=False)
