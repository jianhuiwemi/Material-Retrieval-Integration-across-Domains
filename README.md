<div align="center">

<h1> MaRI: Material Retrieval Integration across Domains </h1>

[**Jianhui Wang**](#)<sup>1</sup> · [**Zhifei Yang**](#)<sup>2</sup> · [**Yangfan He**](#)<sup>3</sup>· [**Huixiong Zhang**](#)<sup>1</sup> · [**Yuxuan Chen**](#)<sup>4</sup> · [**Jingwei Huang**](#)<sup>5✉</sup>

<sup>1</sup>University of Electronic Science and Technology of China · <sup>2</sup>Peking University · <sup>3</sup>University of Minnesota · <sup>4</sup>Fudan University · <sup>5</sup>Tencent Hunyuan3D

<table class="center">
  <tr>
    <td width=100% style="border: none"><img src="figures/teaser.jpg" style="width:100%"></td>
  </tr>
</table>

Accurate material retrieval is critical for creating realistic 3D assets. Existing methods rely on datasets that capture shape-invariant and lighting-varied representations of materials, which are scarce and face challenges due to limited diversity and inadequate real-world generalization. Most current approaches adopt traditional image search techniques. They fall short in capturing the unique properties of material spaces, leading to suboptimal performance in retrieval tasks. Addressing these challenges, we introduce MaRI, a framework designed to bridge the feature space gap between synthetic and real-world materials. MaRI constructs a shared embedding space that harmonizes visual and material attributes through a contrastive learning strategy by jointly training a image and a material encoder, bringing similar materials and images closer while separating dissimilar pairs within the feature space. To support this, we construct a comprehensive dataset comprising high-quality synthetic materials rendered with controlled shape variations and diverse lighting conditions, along with real-world materials processed and standardized using material transfer techniques. Extensive experiments demonstrate the superior performance, accuracy, and generalization capabilities of MaRI across diverse and complex material retrieval tasks, outperforming existing methods.

## Text-to-Image Generation
### 1. Set Environment
```bash
conda env create -f environment.yml
conda activate MaRI
```
### 2. Quick Start
