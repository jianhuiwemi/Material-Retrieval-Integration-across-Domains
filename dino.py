import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm, trange

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
        self.max_files = max_files

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
                    break

                parts = line.strip().split(':', 1)
                if len(parts) < 2:
                    print(f"Warning: Line {line_num} in log file '{log_file_path}' does not contain exactly two values: {line.strip()}")
                    continue

                file_name_with_prefix, material_path = parts
                folder_name = file_name_with_prefix.replace('file_', '').strip()

                img_path = os.path.join(self.root_dir, folder_name, 'fig.png')
                material_path = material_path.strip()
                mask_path = os.path.join(self.root_dir, folder_name, 'mask.png')

                if not os.path.exists(img_path) or not os.path.exists(material_path) or not os.path.exists(mask_path):
                    print(f"Warning: Image, Mask, or Material file for '{folder_name}' does not exist.")
                    continue

                self.image_files.append(img_path)
                self.material_files.append(material_path)
                self.mask_files.append(mask_path)
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
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            material_image = self.transform(material_image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image.size() != mask.size():
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=image.size()[1:], mode='bilinear', align_corners=False).squeeze(0)

        masked_image = image * mask

        label = self.labels[idx]

        return masked_image, material_image, label

# 定义模型类
class CLIPLikeMaterialSearchModel(nn.Module):
    def __init__(self, dinov2_model, temperature=0.07):
        super(CLIPLikeMaterialSearchModel, self).__init__()
        self.render_encoder = dinov2_model
        self.material_encoder = dinov2_model
        self.temperature = temperature

    def forward(self, render_image, material_image):
        # 假设 DINOv2 返回的是最后一层的特征向量
        render_features = self.render_encoder(render_image)  # [Batch, Features]
        material_features = self.material_encoder(material_image)  # [Batch, Features]

        # 归一化特征向量
        render_features = nn.functional.normalize(render_features, dim=-1)
        material_features = nn.functional.normalize(material_features, dim=-1)

        return render_features, material_features

# 定义 InfoNCE 损失函数
def contrastive_loss(render_features, material_features, temperature):
    # 计算相似度矩阵
    logits = torch.matmul(render_features, material_features.T) / temperature

    # 创建标签：正样本对应该在对角线上
    labels = torch.arange(logits.size(0)).to(logits.device)

    # 使用交叉熵损失
    return nn.CrossEntropyLoss()(logits, labels)

# 定义训练函数
def train_model(start_epoch=0, num_epochs=120, learning_rate=1e-5,
                save_path=r'/mnt/large_storage/wry/wjh/output/material_search_model.pth',
                max_files=92000):
    # 加载 DINOv2 模型
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)

    # 定义图像处理步骤
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    root_dir = "/mnt/large_storage/wry/wjh/output"
    log_file_path = os.path.join(root_dir, "render_log.txt")
    dataset = MaterialSearchDataset(root_dir=root_dir, log_file_path=log_file_path, transform=transform,
                                    mask_transform=mask_transform, max_files=max_files)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=128)

    print(f"Dataset size: {len(dataset)}")

    model = CLIPLikeMaterialSearchModel(dinov2_model=dinov2_model)

    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded model from epoch {start_epoch}")

    # 添加总体进度条
    for epoch in trange(start_epoch, num_epochs, desc="Training Progress"):
        model.train()
        total_loss = 0.0

        # 包装 dataloader 添加进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (rendered_img, material_img, _) in enumerate(progress_bar):
            rendered_img = rendered_img.to(device)
            material_img = material_img.to(device)

            render_features, material_features = model(rendered_img, material_img)

            # 使用对比学习损失函数
            loss = contrastive_loss(render_features, material_features, model.module.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条显示的当前损失值
            progress_bar.set_postfix(batch_loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            # 打印每个 batch 的损失
            print(f"Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        tqdm.write(f"Model saved after epoch {epoch + 1}")


if __name__ == "__main__":
    train_model()