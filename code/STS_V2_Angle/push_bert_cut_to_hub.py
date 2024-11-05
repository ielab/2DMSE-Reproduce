from transformers import AutoModel

bert_model = AutoModel.from_pretrained("bert-base-uncased")


for layer_num in range(1, 13):
    # bert cut only the first layer_num layers
    bert_model_cut = AutoModel.from_pretrained("bert-base-uncased")
    bert_model_cut.encoder.layer = bert_model.encoder.layer[:layer_num]
    #upload to hub
    bert_model_cut.push_to_hub(f"bert-base-uncased-{layer_num}layers", private=True)