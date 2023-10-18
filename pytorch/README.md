# Learn by DIY (do it yourself)

Goal: to build llama2 from scratch use PyTorch

Follow what A.Karpathy did use TinyStories

Files
- [x] model.py: vanilla llama model rewritten from scratch use PyTorch framework
- [x] export.py: export A.Karpathy's tinyllamas weights to our own naming format
- [x] tokenizer.py: string to token, token to string
- [x] generation.py: model + tokeninzer + sampling algorithm
- [x] app.py: tinystories top

# Prepare
```
conda create -n llama2 python=3.11
conda activate llama2
pip install -r requirements.txt
```

# How to use?
```
make tinystories
```