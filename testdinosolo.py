import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import shutil  # 用于复制文件
import pytest

# 定义模型类
class CLIPLikeMaterialSearchModel(nn.Module):
    def __init__(self, dinov2_model):
        super(CLIPLikeMaterialSearchModel, self).__init__()
        self.render_encoder = dinov2_model
        self.material_encoder = dinov2_model

    def encode_render(self, render_image):
        return self.render_encoder(render_image)  # 假设 DINOv2 返回特征向量

    def encode_material(self, material_image):
        return self.material_encoder(material_image)  # 假设 DINOv2 返回特征向量

def load_model(model_path, dinov2_model):
    model = CLIPLikeMaterialSearchModel(dinov2_model)
    print(f"Loading fine-tuned model from {model_path}...")
    try:
        # 加载微调后的模型权重
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Fine-tuned model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    return model

@pytest.fixture
def material_library_dir():
    return r"/mnt/large_storage/wry/wjh/figure"

@pytest.fixture
def model():
    model_path = r'/mnt/large_storage/wry/wjh/output/material_search_model.pth'
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
    return load_model(model_path, dinov2_model)

def test_model(render_image_dir, model, material_library_dir, top_k=5):
    if model is None:
        print("Model is not loaded. Exiting test.")
        return

    render_image_path = os.path.join(render_image_dir, 'fig.png')
    mask_path = os.path.join(render_image_dir, 'mask.png')

    if not os.path.exists(render_image_path) or not os.path.exists(mask_path):
        print(f"Skipping {render_image_dir} as required files are missing.")
        return

    print(f"Processing {render_image_dir}...")

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载和处理渲染图像和掩码
    render_image = Image.open(render_image_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('L')
    render_image = transform(render_image).unsqueeze(0)
    mask_image = mask_transform(mask_image).unsqueeze(0)

    if mask_image.size(1) == 1:
        mask_image = mask_image.repeat(1, 3, 1, 1)

    masked_image = render_image * mask_image

    # 推理和相似度计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    masked_image = masked_image.to(device)

    print("Encoding the render image...")
    model.eval()
    with torch.no_grad():
        render_features = model.encode_render(masked_image).cpu()

    print(f"Render features shape: {render_features.shape}")

    print("Searching for the top similar materials...")
    material_features_list = []
    material_paths = []

    # 遍历材质库
    for folder_name in os.listdir(material_library_dir):
        folder_path = os.path.join(material_library_dir, folder_name)

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.png') and "Displacement" not in file_name and "Normal" not in file_name:
                material_image_path = os.path.join(folder_path, file_name)
                material_image = Image.open(material_image_path).convert('RGB')
                material_image = transform(material_image).unsqueeze(0).to(device)
                material_paths.append(material_image_path)

                with torch.no_grad():
                    material_features = model.encode_material(material_image).cpu()
                material_features_list.append(material_features)

    # 堆叠材质特征
    material_features_stack = torch.cat(material_features_list, dim=0)

    print(f"Material features stack shape: {material_features_stack.shape}")

    # 计算余弦相似度
    similarities = torch.cosine_similarity(render_features, material_features_stack, dim=-1)

    print(f"Calculated similarities: {similarities}")

    # 获取相似度最高的前 top_k 个材质
    top_k_indices = torch.topk(similarities, k=top_k).indices
    top_k_similar_materials = [(similarities[i].item(), material_paths[i]) for i in top_k_indices]

    # 保存图片结果到指定文件夹
    results_dir = os.path.join(render_image_dir, "search_results")
    os.makedirs(results_dir, exist_ok=True)

    for i, (similarity, material_image_path) in enumerate(top_k_similar_materials):
        shutil.copy(material_image_path, os.path.join(results_dir, f"similarity_{i+1:.4f}.png"))

    print(f"Top {top_k} similar materials saved in {results_dir}")

    # 打印结果
    print(f"Top {top_k} similar materials found:")
    for similarity, material_image_path in top_k_similar_materials:
        print(f"{material_image_path} - Similarity: {similarity:.4f}")

    # 可视化结果
    plt.figure(figsize=(18, 6))
    render_img = Image.open(render_image_path)
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(render_img)
    plt.title('Rendered Image')
    plt.axis('off')

    for i, (similarity, material_image_path) in enumerate(top_k_similar_materials):
        material_image = Image.open(material_image_path)
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(material_image)
        plt.title(f'Similarity: {similarity:.4f}')
        plt.axis('off')

    plt.subplots_adjust(wspace=0.6)
    plt.show()

def main():
    render_root_dir = r"/mnt/large_storage/wry/wjh/real"
    material_library_dir = r"/mnt/large_storage/wry/wjh/figure"
    model_path = r'/mnt/large_storage/wry/wjh/output/material_search_model.pth'
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
    model = load_model(model_path, dinov2_model)

    for render_image_dir in os.listdir(render_root_dir):
        full_path = os.path.join(render_root_dir, render_image_dir)
        if os.path.isdir(full_path):
            test_model(full_path, model, material_library_dir, top_k=5)

if __name__ == "__main__":
    main()
