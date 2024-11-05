import json
import logging
import os.path
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
import re
from tqdm import tqdm

model_folder = sys.argv[1] if len(sys.argv) > 1 else not None
evaluate_type = sys.argv[2] if len(sys.argv) > 2 else "full"

if model_folder is None:
    print("Please provide a model folder")
    sys.exit()
assert os.path.exists(model_folder), f"Folder {model_folder} does not exist"

if evaluate_type == "full":
    dataset_dict = {
        "stsb": "sentence-transformers/stsb",
        "sts12": "mteb/sts12-sts",
        "sts13": "mteb/sts13-sts",
        "sts14": "mteb/sts14-sts",
        "sts15": "mteb/sts15-sts",
        "sts16": "mteb/sts16-sts",
        "sickr": "mteb/sickr-sts"
    }
elif evaluate_type == "low":
    dataset_dict = {
        "stsb": "sentence-transformers/stsb",
    }


matryoshka_dims = [8, 16, 32, 64, 128, 256, 512, 768]
matryoshka_layers = list(range(2, 13, 2))
# matryoshka_dims = [768]
# matryoshka_layers = [12, 10]

#
result_dict = {}
final_result_dict = {}
for layer in tqdm(matryoshka_layers):
    result_dict[layer] = {}
    for dim in tqdm(matryoshka_dims):
        # if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
        #     continue
        model_name = os.path.join(model_folder, f"sts_bert-base-uncased_layer_{layer}_dim_{dim}/final")
        if not os.path.exists(model_name):
            model_name = os.path.join(model_folder, f"layer_{layer}_dim_{dim}/final")
        if not os.path.exists(model_name):
            continue

        layer_dim_str = str(layer) + "_" + str(dim)
        tem_output = os.path.join(model_name, "tem_output.json")
        if os.path.exists(tem_output):
            result_dict[layer][dim] = {}
            with open(tem_output, "r") as f:
                tem_dict = json.load(f)
            result_dict[layer][dim] = tem_dict
            print("Loaded from tem_output")
            continue


        print("Layer:", layer, "Dim:", dim)

        model = SentenceTransformer(model_name, truncate_dim=dim)
        model[0].auto_model.encoder.layer = model[0].auto_model.encoder.layer[:layer]
        if not os.path.exists(model_name):
            continue
        for dataset in dataset_dict.keys():
            dataset_loading_name = dataset_dict[dataset]
            print("Dataset:", dataset)
            test_dataset = load_dataset(dataset_loading_name, split="test")
            evaluators = []

            evaluators.append(
                EmbeddingSimilarityEvaluator(
                    sentences1=test_dataset["sentence1"],
                    sentences2=test_dataset["sentence2"],
                    scores=test_dataset["score"],
                    main_similarity=SimilarityFunction.COSINE,
                    name=f"sts-test-{dim}",
                    truncate_dim=dim,
                )
            )
            test_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
            results = test_evaluator(model)

            for result_key in list(results.keys()):
                if "spearman_cosine" in result_key:
                    # first copy the key
                    #result_key save is only the number in the key
                    #result_key_save = re.findall(r'\d+', result_key)[0]
                    if dim not in result_dict[layer]:
                        result_dict[layer][dim] = {dataset: results[result_key]}
                    else:
                        result_dict[layer][dim][dataset] = results[result_key]
        with open(tem_output, "w") as f:
            json.dump(result_dict[layer][dim], f, indent=2)



        # with open(os.path.join(model_name, "sts_results.json"), "w") as f:
        #     json.dump(result_dict[layer_dim_str], f, indent=2)

for dataset in dataset_dict.keys():
    final_result_dict[dataset] = {}


for layer in result_dict:
    for dim in result_dict[layer]:
        for dataset in result_dict[layer][dim]:
            if dataset not in final_result_dict:
                final_result_dict[dataset] = {}
            if layer not in final_result_dict[dataset]:
                final_result_dict[dataset][layer] = {}
            final_result_dict[dataset][layer][dim] = result_dict[layer][dim][dataset]

print(final_result_dict)
final_result_dict["average"] = {}
for layer in final_result_dict["stsb"]:
    final_result_dict["average"][layer] = {}
    for dim in final_result_dict["stsb"][layer]:
        final_result_dict["average"][layer][dim] = []
        for dataset in final_result_dict:
            if dataset == "average":
                continue
            final_result_dict["average"][layer][dim].append(final_result_dict[dataset][layer][dim])
        final_result_dict["average"][layer][dim] = sum(final_result_dict["average"][layer][dim]) / len(final_result_dict["average"][layer][dim])
print(final_result_dict)
# another average for all sizes of dataset
final_result_dict["average_dataset"] = {}
for dataset in dataset_dict.keys():
    final_result_dict["average_dataset"][dataset] = []
    for layer in final_result_dict["average"]:
        for dim in final_result_dict["average"][layer]:
            final_result_dict["average_dataset"][dataset].append(final_result_dict[dataset][layer][dim])
    final_result_dict["average_dataset"][dataset] = sum(final_result_dict["average_dataset"][dataset]) / len(final_result_dict["average_dataset"][dataset])


out_file = os.path.join(model_folder, "sts_results_full.json")
json.dump(final_result_dict, open(out_file, "w"), indent=2)



