# 2DMSE-Reproduce for STS_V2
Reproduction code of 2DMSE STS_V2 results


## Code for Reproduction

### Model training

#### Training Base Model (full model and seperate model)

This will train a set of small-scale models, including the full-size model.

```bash
sh train_baseline.sh
```

#### Training MSE model (1D):
    
```bash
sh train_mse.sh
```

#### Training 2DMSE model (2D):
    
```bash
sh train_2dmse.sh
```


### Model Conversion

#### Convert v2 models to v1 models

```bash
python3 scripts/convert_to_sentence_transformer.py --model_name_or_path MODEL_PATH --output_dir OUTPUT_DIRß
```


### Inference

For inference, please see V1 README.md [STS_v1](../../code/STS_v1/README.md) for details.




