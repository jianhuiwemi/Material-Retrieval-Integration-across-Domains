import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm, trange
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class MaterialSearchDataset(Dataset):
    def __init__(self, root_dir, log_file_path, transform=None, mask_transform=None, max_files=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = []
        self.material_files = []
        self.mask_files = []
        self.labels = []
        self.max_files = max_files

        self.label_dict = {}
        self._create_label_dict_and_parse_data(root_dir, log_file_path)

        print(f"Found {len(self.image_files)} images with {len(set(self.labels))} unique labels.")

    def _create_label_dict_and_parse_data(self, root_dir, log_file_path):
        materials = set()
        line_count = 0

        if root_dir and log_file_path:
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    parts = line.strip().split(':', 1)
                    if len(parts) < 2:
                        continue
                    material_path = parts[1].strip()
                    materials.add(material_path)

            material_list = sorted(materials)
            self.label_dict.update({material: idx for idx, material in enumerate(material_list)})

            with open(log_file_path, 'r') as log_file:
                for line_num, line in enumerate(log_file, 1):
                    if self.max_files and line_count >= self.max_files:
                        break

                    parts = line.strip().split(':', 1)
                    if len(parts) < 2:
                        print(f"Warning: Line {line_num} in log file '{log_file_path}' does not contain exactly two values: {line.strip()}")
                        continue

                    file_name_with_prefix, material_path = parts
                    folder_name = file_name_with_prefix.replace('file_', '').strip()

                    img_path = os.path.join(root_dir, folder_name, 'fig.png')
                    material_path = material_path.strip()
                    mask_path = os.path.join(root_dir, folder_name, 'mask.png')

                    if not os.path.exists(img_path) or not os.path.exists(material_path) or not os.path.exists(mask_path):
                        print(f"Warning: Image, Mask, or Material file for '{folder_name}' does not exist.")
                        continue

                    label = self.label_dict.get(material_path, -1)
                    if label == -1:
                        print(f"Warning: Material path '{material_path}' not found in label_dict. Skipping.")
                        continue

                    self.image_files.append(img_path)
                    self.material_files.append(material_path)
                    self.mask_files.append(mask_path)
                    self.labels.append(label)
                    line_count += 1

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        material_path = self.material_files[idx]
        mask_path = self.mask_files[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            material_image = Image.open(material_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except (OSError, IOError) as e:
            print(f"Warning: Failed to load image at index {idx}. Error: {e}")
            new_idx = (idx + 1) % len(self)
            if new_idx == idx:
                raise RuntimeError("No valid images found in dataset.")
            return self.__getitem__(new_idx)

        if self.transform:
            image = self.transform(image)
            material_image = self.transform(material_image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image.size()[1:] != mask.size()[1:]:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=image.size()[1:], mode='nearest').squeeze(0)

        masked_image = image * mask

        label = self.labels[idx]

        return masked_image, material_image, mask, label


class CLIPLikeMaterialSearchModel(nn.Module):
    def __init__(self, dinov2_model, temperature=0.07):
        super(CLIPLikeMaterialSearchModel, self).__init__()
        self.render_encoder = dinov2_model
        self.material_encoder = copy.deepcopy(dinov2_model)
        self.temperature = temperature

        for param in self.render_encoder.parameters():
            param.requires_grad = False
        for param in self.material_encoder.parameters():
            param.requires_grad = False

        if hasattr(self.render_encoder, 'blocks'):
            print("Unfreezing the last Transformer block of render_encoder.")
            for param in self.render_encoder.blocks[-1].parameters():
                param.requires_grad = True
        else:
            print("render_encoder does not have 'blocks' attribute. Please check the model architecture.")

        if hasattr(self.material_encoder, 'blocks'):
            print("Unfreezing the last Transformer block of material_encoder.")
            for param in self.material_encoder.blocks[-1].parameters():
                param.requires_grad = True
        else:
            print("material_encoder does not have 'blocks' attribute. Please check the model architecture.")

    def forward(self, render_image, material_image):
        render_features = self.render_encoder(render_image)
        material_features = self.material_encoder(material_image)

        render_features = nn.functional.normalize(render_features, dim=-1)
        material_features = nn.functional.normalize(material_features, dim=-1)

        return render_features, material_features

def contrastive_loss(render_features, material_features, temperature):
    logits = torch.matmul(render_features, material_features.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return nn.CrossEntropyLoss()(logits, labels)


def train_model(start_epoch=0, num_epochs=120, learning_rate=1e-4,
                save_dir=r'/data/yangzhifei/project/MaRi/onlysbs/pth/checkpoints',
                max_files=None, batch_size=256, early_stop_patience=1,
                resume_checkpoint=None):
    os.makedirs(save_dir, exist_ok=True)
    best_save_path = os.path.join(save_dir, 'best_model.pth')
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.NEAREST),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    root_dir = '/path_to_synthetic_data'
    log_file_path = os.path.join(root_dir, "render_log.txt")

    dataset = MaterialSearchDataset(root_dir=root_dir, log_file_path=log_file_path,
                                    transform=transform, mask_transform=mask_transform, max_files=max_files)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers = min(8, os.cpu_count() // 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Training Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")

    model = CLIPLikeMaterialSearchModel(dinov2_model=dinov2_model)

    model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print("No trainable parameters found. Exiting.")
        return

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}")


    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            checkpoint = torch.load(resume_checkpoint)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            epochs_no_improve = 0
            print(f"Loaded checkpoint from '{resume_checkpoint}' at epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint '{resume_checkpoint}': {e}")
            print("Starting training from scratch.")

    for epoch in trange(start_epoch, num_epochs, desc="Training Progress"):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (masked_img, material_img, mask, _) in enumerate(progress_bar):
            masked_img = masked_img.to(device, non_blocking=True)
            material_img = material_img.to(device, non_blocking=True)
            render_features, material_features = model(masked_img, material_img)
            loss = contrastive_loss(render_features, material_features, model.module.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(batch_loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            if i % 100 == 0:
                tqdm.write(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Avg Training Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for masked_img, material_img, mask, _ in tqdm(val_loader, desc="Validation"):
                masked_img = masked_img.to(device, non_blocking=True)
                material_img = material_img.to(device, non_blocking=True)

                render_features, material_features = model(masked_img, material_img)

                loss = contrastive_loss(render_features, material_features, model.module.temperature)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_train_loss': avg_loss,
                'avg_val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, best_save_path)
            tqdm.write(f"Validation loss improved. Best model saved at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            tqdm.write(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        current_save_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'avg_train_loss': avg_loss,
            'avg_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        }, current_save_path)
        tqdm.write(f"Checkpoint saved: {current_save_path}")

        if epochs_no_improve >= early_stop_patience:
            tqdm.write(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    print("Training completed.")

if __name__ == "__main__":
    train_model(batch_size=256, early_stop_patience=5, resume_checkpoint=None)