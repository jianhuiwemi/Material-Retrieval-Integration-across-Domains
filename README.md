<div align="center">
  
# MaRI: Material Retrieval Integration across Domains

<a href="https://jianhuiwemi.github.io/MaRI"><img src="https://img.shields.io/badge/Project_Page-Online-EA3A97"></a>
<a href="https://arxiv.org/abs/2503.08111"><img src="https://img.shields.io/badge/ArXiv-2503.01370-brightgreen"></a> 

</div>

# 📥 Dataset Construction Pipeline
## Synthetic Data

We use [Blender 4.2.2](https://www.blender.org/) to render the synthetic dataset. Our rendering pipeline is provided here. To create your dataset, simply run:
  
```bash
/path/blender-4.2.2-linux-x64/blender -b --python /dataset/synthetic/render.py
```
This is the files structure:
```bash
MaRI/
├── blender-4.2.2-linux-x64/
├── objs/
│   ├── 000-xxx/
│   │   ├── modelA.glb
│   │   └── modelB.glb
│   ├── 000-xxx/
│   │   ├── modelC.glb
│   │   └── modelD.glb
│   └── ...
├── material/
│   ├── material1/
│   │   ├── material1_Color.jpg
│   │   ├── material1_NormalGL.jpg
│   │   ├── material1_Roughness.jpg
│   │   ├── material1_Displacement.jpg
│   │   └── material1_Metalness.jpg
│   ├── material2/
│   │   ├── material2_Color.jpg
│   │   ├── material2_NormalGL.jpg
│   │   └── ...
│   └── ...
├── hdri/
│   └── hdri_files/
│       ├── env1.hdr
│       ├── env2.hdr
│       └── ...
└── data/
    └── final/
        ├── 1/
        │   ├── fig.png
        │   └── mask.png
        ├── 2/
        │   ├── fig.png
        │   └── mask.png
        └── ...

```

## Real-world Data

# ⚙ Train


# ⭐ Inference



## Download the pretrained model

