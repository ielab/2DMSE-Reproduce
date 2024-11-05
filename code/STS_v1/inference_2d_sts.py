import json
import os.path
import sys
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
import re

model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
evaluate_type = sys.argv[2] if len(sys.argv) > 2 else "full"
layer_dim_type = sys.argv[3] if len(sys.argv) > 3 else "diaganol"
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

final_result_dict = {}

matryoshka_dims = []

matryoshka_layers = []

matryoshka_dims += [8, 16, 32, 64, 128, 256, 512, 768]
matryoshka_layers += list(range(2, 13, 2))

# for i in range(2, 13, 2):
#     for j in [768, 512, 256, 128, 64, 32]:
#         matryoshka_layers.append(i)
#         matryoshka_dims.append(j)


for dataset in tqdm(dataset_dict.keys()):
    dataset_loading_name = dataset_dict[dataset]

    test_dataset = load_dataset(dataset_loading_name, split="test")

    # evaluators = []
    # for dim in matryoshka_dims:
    #     evaluators.append(
    #         EmbeddingSimilarityEvaluator(
    #             sentences1=test_dataset["sentence1"],
    #             sentences2=test_dataset["sentence2"],
    #             scores=test_dataset["score"],
    #             main_similarity=SimilarityFunction.COSINE,
    #             name=f"sts-test-{dim}",
    #             truncate_dim=dim,
    #         )
    #     )
    # #
    result_dict = {}
    for layer_i, layer in enumerate(matryoshka_layers):
        evaluators = []
        for dim in matryoshka_dims:
            if layer_dim_type == "diaganol":
                if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
                    continue
            #dim = matryoshka_dims[layer_i]
            evaluators.append(
                EmbeddingSimilarityEvaluator(
                    sentences1=test_dataset["sentence1"],
                    sentences2=test_dataset["sentence2"],
                    scores=test_dataset["score"],
                    main_similarity=SimilarityFunction.COSINE,
                    name=f"sts-test-{dim}",
                    truncate_dim=dim
                )
            )
        model = SentenceTransformer(model_name)
        model[0].auto_model.encoder.layer = model[0].auto_model.encoder.layer[:layer]
        test_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
        results = test_evaluator(model)
        #print("Layer:", layer)
        result_dict[layer] = {}
        for result_key in list(results.keys()):
            if "spearman_cosine" in result_key:
                # first copy the key
                #result_key save is only the number in the key
                result_key_save = re.findall(r'\d+', result_key)[0]
                #print(result_key_save)
                result_dict[layer][result_key_save] = results[result_key]
    final_result_dict[dataset] = result_dict

final_result_dict["average"] = {}
for layer_i, layer in enumerate(matryoshka_layers):
    final_result_dict["average"][layer] = {}
    #dim = matryoshka_dims[layer_i]
    for dim in matryoshka_dims:
        if layer_dim_type == "diaganol":
            if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
                continue
        final_result_dict["average"][layer][dim] = sum([final_result_dict[dataset][layer][str(dim)] for dataset in dataset_dict.keys()]) / len(dataset_dict.keys())

final_result_dict["average_dataset"] = {}
for dataset in dataset_dict.keys():
    final_result_dict["average_dataset"][dataset] = []
    for layer_i, layer in enumerate(matryoshka_layers):
        for dim in matryoshka_dims:
            if layer_dim_type == "diaganol":
                if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
                    continue
            final_result_dict["average_dataset"][dataset].append(final_result_dict[dataset][layer][str(dim)])
    final_result_dict["average_dataset"][dataset] = sum(final_result_dict["average_dataset"][dataset]) / len(final_result_dict["average_dataset"][dataset])

if layer_dim_type == "diaganol":
    out_file = os.path.join(model_name, "sts_results.json")
else:
    out_file = os.path.join(model_name, "sts_results_full.json")
json.dump(final_result_dict, open(out_file, "w"), indent=2)




