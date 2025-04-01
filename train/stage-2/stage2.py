import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm, trange
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class MaterialSearchDataset(Dataset):
    def __init__(self, root_dirs, log_file_paths, transform=None, mask_transform=None, max_files=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = []
        self.material_files = []
        self.mask_files = []
        self.labels = []
        self.dataset_ids = []
        self.max_files = max_files
        self.label_dict = {}
        self._create_label_dict_and_parse_data(root_dirs, log_file_paths)

        print(f"Found {len(self.image_files)} images with {len(set(self.labels))} unique labels.")

    def _create_label_dict_and_parse_data(self, root_dirs, log_file_paths):
        root_dir2 = root_dirs.get('dataset2')
        if root_dir2:
            for subdir in os.listdir(root_dir2):
                subdir_path = os.path.join(root_dir2, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                fig_path = os.path.join(subdir_path, 'fig.png')
                mask_path = os.path.join(subdir_path, 'mask.png')
                material_path = os.path.join(subdir_path, 'output.png')

                if self.max_files and len(self.image_files) >= self.max_files:
                    break

                if not os.path.exists(fig_path) or not os.path.exists(mask_path) or not os.path.exists(material_path):
                    print(f"Warning: fig.png, mask.png, or output.png missing in '{subdir_path}'. Skipping.")
                    continue

                if material_path not in self.label_dict:
                    self.label_dict[material_path] = len(self.label_dict)

                label = self.label_dict[material_path]

                self.image_files.append(fig_path)
                self.material_files.append(material_path)
                self.mask_files.append(mask_path)
                self.labels.append(label)
                self.dataset_ids.append(1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        material_path = self.material_files[idx]
        mask_path = self.mask_files[idx]
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


def train_model_finetune(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=50,
    save_dir='/save_path',
    early_stop_patience=5
):
    import torch.nn.functional as F
    from tqdm import tqdm, trange

    os.makedirs(save_dir, exist_ok=True)

    def contrastive_loss(render_features, material_features, temperature):
        logits = torch.matmul(render_features, material_features.T) / temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        return F.cross_entropy(logits, labels)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in trange(num_epochs, desc="Fine-Tuning Progress"):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (masked_img, material_img, mask, _) in enumerate(progress_bar):
            masked_img = masked_img.to(device, non_blocking=True)
            material_img = material_img.to(device, non_blocking=True)

            optimizer.zero_grad()

            render_features, material_features = model(masked_img, material_img)
            if hasattr(model, 'module'):
                temperature = model.module.temperature
            else:
                temperature = model.temperature

            loss = contrastive_loss(render_features, material_features, temperature)

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
                # 同样获取 temperature
                if hasattr(model, 'module'):
                    temperature = model.module.temperature
                else:
                    temperature = model.temperature

                loss = contrastive_loss(render_features, material_features, temperature)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        epoch_save_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),  # 如果使用 DataParallel，可以保留
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }, epoch_save_path)
        tqdm.write(f"Model saved for epoch {epoch + 1} at {epoch_save_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_save_path)
            tqdm.write(f"Validation loss improved. Best model saved at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            tqdm.write(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stop_patience:
            tqdm.write(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    print("Fine-Tuning completed.")


if __name__ == "__main__":
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

    root_dirs = {
        'dataset1': None,  # not include synthetic data
        'dataset2': '/path_to_real-world_data'
    }

    log_file_paths = {
        'dataset1': None,
        'dataset2': None  # real-world data does not include log files
    }


    small_dataset = MaterialSearchDataset(
        root_dirs=root_dirs,
        log_file_paths=log_file_paths,
        transform=transform,
        mask_transform=mask_transform,
        max_files=None
    )


    train_size = int(0.9 * len(small_dataset))
    val_size = len(small_dataset) - train_size
    train_dataset, val_dataset = random_split(small_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    print(f"Fine-Tuning Training Dataset size: {len(train_dataset)}")
    print(f"Fine-Tuning Validation Dataset size: {len(val_dataset)}")

    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    model = CLIPLikeMaterialSearchModel(dinov2_model=dinov2_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint_path = '/path/epoch_1.pth'  # weights from stage-1
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)

    def unfreeze_more_layers(model, num_layers=1):
        if hasattr(model, 'module'):
            render_encoder = model.module.render_encoder
            material_encoder = model.module.material_encoder
        else:
            render_encoder = model.render_encoder
            material_encoder = model.material_encoder

        if hasattr(render_encoder, 'blocks'):
            for block in render_encoder.blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            print("render_encoder does not have 'blocks' attribute.")

        if hasattr(material_encoder, 'blocks'):
            for block in material_encoder.blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            print("material_encoder does not have 'blocks' attribute.")

    unfreeze_more_layers(model, num_layers=1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check which parameters are frozen.")

    optimizer = optim.Adam(trainable_params, lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    train_model_finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50,
        save_dir='/data/yangzhifei/project/MaRi/mix/stage2',
        early_stop_patience=1