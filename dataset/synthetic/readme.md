# Synthetic Dataset

## Installation Instructions
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
├── materials/
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
    └── synthetic/
        ├── 1/
        │   ├── fig.png
        │   └── mask.png
        ├── 2/
        │   ├── fig.png
        │   └── mask.png
        └── ...

```
