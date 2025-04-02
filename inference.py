import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import copy
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class CLIPLikeMaterialSearchModel(nn.Module):
    def __init__(self, base_model, temperature=0.07):
        super().__init__()
        self.render_encoder = base_model
        self.material_encoder = copy.deepcopy(base_model)
        self.temperature = temperature

    def encode_render(self, x):
        x = self.render_encoder(x)
        x = nn.functional.normalize(x, dim=-1)
        return x

    def encode_material(self, x):
        x = self.material_encoder(x)
        x = nn.functional.normalize(x, dim=-1)
        return x

def load_finetuned_model(model_path, base_model, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    model = CLIPLikeMaterialSearchModel(base_model)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model


def build_material_library_features(library_dir, model, transform, device, batch_size=32):
    material_paths = []
    for folder in os.listdir(library_dir):
        folder_path = os.path.join(library_dir, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".png"):
                    if "Displacement" in file_name or "Normal" in file_name:
                        continue
                    material_paths.append(os.path.join(folder_path, file_name))

    all_features = []
    model.eval()

    for i in range(0, len(material_paths), batch_size):
        batch_paths = material_paths[i : i + batch_size]
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = transform(img)
                images.append(img)
            except:
                continue
        if not images:
            continue
        images = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model.encode_material(images).cpu()
        all_features.append(feats)

    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)  # [N, feature_dim]
    else:
        all_features = torch.empty(0)
    return all_features, material_paths


def main():
    finetuned_model_path = "/path/epoch_25.pth"
    material_library_dir = "/path/material_library"
    input_image_path = "/path/fig.png"
    input_mask_path = "/path/mask.png"
    results_dir = "/path/search_results"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    base_model.to(device)

    model = load_finetuned_model(finetuned_model_path, base_model, device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    image = Image.open(input_image_path).convert("RGB")
    mask = Image.open(input_mask_path).convert("L")
    image = transform(image).unsqueeze(0)  
    mask = mask_transform(mask).unsqueeze(0)  
    masked_image = image * mask

    model.eval()
    with torch.no_grad():
        render_feats = model.encode_render(masked_image.to(device))

    material_feats, material_paths = build_material_library_features(material_library_dir, model, transform, device)
    material_feats = material_feats.to(device)

    if material_feats.shape[0] == 0:
        print("No valid material images found in library.")
        return

    sims = torch.matmul(render_feats, material_feats.T).squeeze(0)  # [N]
    top_k = 5
    topk_indices = torch.topk(sims, k=top_k).indices
    topk_scores = sims[topk_indices].tolist()

    for rank, idx in enumerate(topk_indices):
        score = topk_scores[rank]
        src_path = material_paths[idx]
        dst_path = os.path.join(results_dir, f"{rank+1}_{score:.4f}_{os.path.basename(src_path)}")
        shutil.copy(src_path, dst_path)
    print(f"Top-{top_k} materials copied to {results_dir}.")

if __name__ == "__main__":
    main()
