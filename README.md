<div align="center">
  
# MaRI: Material Retrieval Integration across Domains

<a href="https://jianhuiwemi.github.io/MaRI"><img src="https://img.shields.io/badge/Project_Page-Online-EA3A97"></a>
<a href="https://arxiv.org/abs/2503.08111"><img src="https://img.shields.io/badge/ArXiv-2503.01370-brightgreen"></a> 

</div>

# 📥 Dataset Construction Pipeline
We provide detailed pipeline instructions and related explanations in two files: [Synthetic Data](./dataset/synthetic/readme.md) and [Real-world Data](./dataset/real/readme.md). Please refer to each for more information.





# ⚙ Fine-tune
We provide training scripts in `train/stage1.py` and `train/stage2.py`. You can run them using:

```bash
python train/stage-1.py  # Fine-tune on large-scale synthetic data
```
We found that fine-tuning on large-scale synthetic data for just one epoch yields the best performance. Then, we perform a second-stage fine-tuning on real-world data with a lower learning rate:
```bash
python train/stage-2.py  # Fine-tune on real-world data
```

# ⭐ Inference



## Download the pretrained model

