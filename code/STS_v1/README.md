# 2DMSE-Reproduce for STS_V1
Reproduction code of 2DMSE STS_V1 results


## Code for Reproduction

### Model training

#### Training Base Model (full model and seperate model)

This will train a set of small-scale models, including the full-size model.

```bash
python3 baseline.py
```


#### Training MSE model (1D):
    
```bash
python3 matyoshka_sts.py
```

#### Training 2DMSE model (2D):
    
```bash
python3 2d_matyoshka_sts.py
```

### Inference

#### Inference Base Model (full model and seperate model)

```bash
python3 inference_baseline.py MODEL_PATH
```

#### Inference MSE model:

```bash
python3 inference_matyoshka_sts.py MODEL_PATH
```







