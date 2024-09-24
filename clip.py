import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor

# 定义数据集类
class MaterialSearchDataset(Dataset):
    def __init__(self, root_dir, log_file_path, transform=None, mask_transform=None, max_files=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = []
        self.material_files = []
        self.mask_files = []
        self.labels = []
        self.max_files = max_files  # 新增的参数，用于限制加载的文件数量

        # 创建标签字典
        self.label_dict = self.create_label_dict(log_file_path)
        self._parse_render_log(log_file_path)

        print(f"Found {len(self.image_files)} images with {len(set(self.labels))} unique labels.")

    def create_label_dict(self, log_file_path):
        materials = set()
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                parts = line.strip().split(':', 1)
                if len(parts) < 2:
                    continue
                material_path = parts[1].strip()
                materials.add(material_path)

        return {material: idx for idx, material in enumerate(sorted(materials))}

    def _parse_render_log(self, log_file_path):
        with open(log_file_path, 'r') as log_file:
            for line_num, line in enumerate(log_file, 1):
                if self.max_files and len(self.image_files) >= self.max_files:
                    break  # 如果达到了最大文件数量限制，退出循环

                parts = line.strip().split(':', 1)
                if len(parts) < 2:
                    print(
                        f"Warning: Line {line_num} in log file '{log_file_path}' does not contain exactly two values: {line.strip()}")
                    continue

                file_name_with_prefix, material_path = parts
                folder_name = file_name_with_prefix.replace('file_', '').strip()

                img_path = os.path.join(self.root_dir, folder_name, 'fig.png')
                material_path = material_path.strip()
                mask_path = os.path.join(self.root_dir, folder_name, 'mask.png')  # 假设掩码文件名为'mask.png'

                if not os.path.exists(img_path) or not os.path.exists(material_path) or not os.path.exists(mask_path):
                    print(f"Warning: Image, Mask, or Material file for '{folder_name}' does not exist.")
                    continue

                self.image_files.append(img_path)
                self.material_files.append(material_path)
                self.mask_files.append(mask_path)  # 将掩码路径添加到列表中
                self.labels.append(self.label_dict.get(material_path, -1))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        material_path = self.material_files[idx]
        mask_path = self.mask_files[idx]

        # 加载图像和掩码
        image = Image.open(img_path).convert('RGB')
        material_image = Image.open(material_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 假设掩码是单通道灰度图像

        # print(f"Loaded image: {img_path}, material image: {material_path}, mask: {mask_path}")

        # 应用 transforms
        if self.transform:
            image = self.transform(image)
            material_image = self.transform(material_image)
        if self.mask_transform:
            mask = self.mask_transform(mask)  # 将掩码转换为PyTorch张量

        # 确保掩码是一个张量，并检查其形状
        if mask.dim() == 2:  # 如果掩码是二维张量 (H, W)
            mask = mask.unsqueeze(0)  # 添加一个维度以匹配 (1, H, W)

        if mask.size(0) == 1:  # 检查掩码的通道数是否为1
            mask = mask.expand(3, -1, -1)  # 将单通道掩码扩展为三通道张量

        masked_image = image * mask  # 将掩码应用于图像

        label = self.labels[idx]

        return masked_image, material_image, label


# 定义模型类
class CLIPLikeMaterialSearchModel(nn.Module):
    def __init__(self, vision_model_name='google/vit-base-patch16-224-in21k'):
        super(CLIPLikeMaterialSearchModel, self).__init__()
        self.render_encoder = ViTModel.from_pretrained(vision_model_name)
        self.material_encoder = ViTModel.from_pretrained(vision_model_name)

    def forward(self, render_image, material_image):
        render_features = self.render_encoder(render_image).last_hidden_state[:, 0, :]  # [CLS] token
        material_features = self.material_encoder(material_image).last_hidden_state[:, 0, :]  # [CLS] token
        return render_features, material_features

# 定义训练函数
def train_model(start_epoch=0, num_epochs=20, learning_rate=1e-5, save_path='/mnt/large_storage/wry/wjh/output/material_search_model.pth',
                max_files=92000):
    # 初始化 ViT 的 image processor
    image_processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 创建数据集
    root_dir = "/mnt/large_storage/wry/wjh/output"  # 替换为你的数据集路径
    log_file_path = os.path.join(root_dir, "render_log.txt")
    dataset = MaterialSearchDataset(root_dir=root_dir, log_file_path=log_file_path, transform=transform,
                                    mask_transform=mask_transform, max_files=max_files)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=64)

    print(f"Dataset size: {len(dataset)}")

    # 定义模型和优化器
    model = CLIPLikeMaterialSearchModel()

    # 使用 DataParallel 包装模型，利用多张显卡进行训练
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将模型移动到 GPU（如果有 GPU 可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 检查是否存在已保存的模型
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])  # 使用 model.module 加载模型权重
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded model from epoch {start_epoch}")

    # 训练模型
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for i, (rendered_img, material_img, _) in enumerate(dataloader):
            rendered_img = rendered_img.to(device)
            material_img = material_img.to(device)

            # 生成负样本（打乱材质图像顺序）
            negative_img = material_img[torch.randperm(material_img.size(0))]

            # 获取正样本和负样本特征
            render_features, positive_material_features = model(rendered_img, material_img)
            _, negative_material_features = model(rendered_img, negative_img)

            # 计算正样本和负样本的余弦相似度
            positive_similarity = torch.cosine_similarity(render_features, positive_material_features)
            negative_similarity = torch.cosine_similarity(render_features, negative_material_features)

            # 创建标签：正样本相似度应为1，负样本相似度应为0
            positive_labels = torch.ones(positive_similarity.size()).to(device)
            negative_labels = torch.zeros(negative_similarity.size()).to(device)

            # 计算正样本和负样本的损失
            positive_loss = nn.MSELoss()(positive_similarity, positive_labels)
            negative_loss = nn.MSELoss()(negative_similarity, negative_labels)

            # 总损失
            loss = positive_loss + negative_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # 每个epoch结束后保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),  # 保存 model.module 的状态
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
        }, save_path)
        print(f"Model saved after epoch {epoch + 1}")


if __name__ == "__main__":
    train_model()