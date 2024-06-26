from flask import Flask, request
import pickle
from transformers import AutoModel, AutoTokenizer
import torch
import json

app=Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained('/share/home/liuxiaoyan/BAAI/bge-small-en-v1.5')
tokenizer.pad_token_id = tokenizer.sep_token_id
model = AutoModel.from_pretrained('/share/home/liuxiaoyan/BAAI/bge-small-en-v1.5')
model = model.to("cuda:3")
model.eval()

@app.route('/embedding', methods=["POST"])
def embedding():
    data = json.loads(request.get_json())
    field = data['field'] 
    with torch.no_grad():
        token_seq = tokenizer(text=field, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        inputs = model(token_seq['input_ids'].cuda(3))["last_hidden_state"].cpu().numpy()
    return pickle.dumps({"input_ids": token_seq["input_ids"], "embeddings": inputs})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
