import os
import sys
import json
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision import transforms
from PIL import Image
import timm

# ----------------------------------------------------
# 1. 경로 설정 및 라이브러리 추가
# ----------------------------------------------------
OLD_FILM_DIR = Path(__file__).resolve().parent
if str(OLD_FILM_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_FILM_DIR))

# CVAE 및 데이터셋 클래스 임포트 (vanilla_cvae_no_disentanglement에서 로드)
from vanilla_cvae_no_disentanglement import (
    ImprovedCVAE,
    OASISContrastiveDataset,
    SimpleClassifier,
    transform,
    CLASSES,
    NUM_CLASSES,
    DEVICE,
    LATENT_DIM,
    IMAGE_CHANNEL,
    IMAGE_SIZE,
)

# ----------------------------------------------------
# 2. 하이퍼파라미터 및 경로 탐색
# ----------------------------------------------------
# 데이터셋 경로 후보 탐색
DATA_DIR = os.environ.get("OLD_FILM_DATA_DIR")
# 명시적인 데이터 경로 탐색 후보 (all 폴더 및 oasis_data 추가)
candidates = [
    # 1. Old_FiLM/data/all (사용자님이 전처리 완료한 41-Subject 데이터셋 경로)
    os.path.join(OLD_FILM_DIR, "data", "all"),
    # 2. oasis_data (사용자님의 메인 데이터셋 경로)
    os.path.join(OLD_FILM_DIR.parent, "oasis_data"),
    # 3. Elice GPU 상의 41-Subject 데이터셋 경로
    "/home/elicer/deep_learning_41_subject_dataset/all",
    # 4. Old_FiLM/data 경로
    os.path.join(OLD_FILM_DIR, "data")
]
if not DATA_DIR:
    for cand in candidates:
        if os.path.exists(cand):
            # 해당 경로 안에 치매 단계 클래스 폴더가 실제로 있는지 유효성 검사
            try:
                subdirs = os.listdir(cand)
                if any(c_name in subdirs for c_name in CLASSES.values()):
                    DATA_DIR = cand
                    break
            except Exception:
                continue
# 위 탐색이 실패했을 경우, 단순히 존재 여부만으로 매칭 시도
if not DATA_DIR:
    for cand in candidates:
        if os.path.exists(cand):
            DATA_DIR = cand
            break
# 최종 실패 대비 기본값 지정
if not DATA_DIR:
    DATA_DIR = candidates[0]
OUTPUT_FOLDER = os.path.join(OLD_FILM_DIR, "baseline_results", "GEN_SAMPLES_BASELINE_HYPERPARAMETER")
CHECKPOINT_PATH = os.path.join(OLD_FILM_DIR, "baseline_checkpoints", "best_cvae_baseline_ablation_hyperparameter.pth")
CLASSIFIER_PATH = os.path.join(OLD_FILM_DIR, "best_classifier_resnet18_weights_42.pth")

print("="*60)
print("🔍 설정 경로 정보")
print(f" - 데이터 경로: {DATA_DIR}")
print(f" - 결과물 저장 경로: {OUTPUT_FOLDER}")
print(f" - CVAE 가중치 경로: {CHECKPOINT_PATH}")
print(f" - 분류기 가중치 경로: {CLASSIFIER_PATH}")
print("="*60)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------------------
# 3. 모델 및 체크포인트 로드
# ----------------------------------------------------
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ 에러: {CHECKPOINT_PATH} 파일이 존재하지 않습니다.")
    sys.exit(1)

print("\n🚀 CVAE Baseline 가중치 로드 중...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = ImprovedCVAE(
    latent_dim=LATENT_DIM,
    img_size=IMAGE_SIZE,
    img_channel=IMAGE_CHANNEL,
    num_classes=NUM_CLASSES
).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

# 에포크 및 최저 손실 파악 및 저장
best_epoch = checkpoint.get("epoch", "Unknown")
best_val_loss = checkpoint.get("val_loss", "Unknown")
print(f"✅ Baseline 최적 에포크: {best_epoch}")
print(f"✅ Baseline 최저 Val Loss: {best_val_loss:.4f}" if isinstance(best_val_loss, float) else f"✅ Baseline 최저 Val Loss: {best_val_loss}")

# Validation Loss 정보를 텍스트 로그 파일로 저장
log_save_path = os.path.join(OUTPUT_FOLDER, "baseline_best_val_loss.txt")
with open(log_save_path, "w", encoding="utf-8") as f:
    f.write(f"Baseline Best Epoch: {best_epoch}\n")
    f.write(f"Baseline Best Validation Loss: {best_val_loss}\n")
print(f"Saved: {log_save_path}")

# ----------------------------------------------------
# 4. 데이터셋 로드
# ----------------------------------------------------
if not os.path.exists(DATA_DIR):
    print(f"❌ 에러: 데이터셋 경로 {DATA_DIR} 가 존재하지 않습니다.")
    sys.exit(1)

val_dataset = OASISContrastiveDataset(DATA_DIR, transform=transform, split='val')
if len(val_dataset) == 0:
    print(f"❌ 에러: 지정된 데이터 경로 {DATA_DIR} 에서 검증 데이터를 찾지 못했습니다.")
    print("현재 디렉토리 구조 및 클래스명 폴더('Non Demented', 'Very mild Dementia', 'Mild Dementia')가 있는지 확인해 주세요.")
    print("수동으로 올바른 경로를 지정하려면 아래처럼 실행하세요:")
    print("OLD_FILM_DATA_DIR=\"/실제/데이터/경로\" python Old_FiLM/evaluate_baseline_full.py")
    sys.exit(1)
print(f"✅ 검증 데이터 로드 완료 (샘플 수: {len(val_dataset)})")

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ----------------------------------------------------
# 5. 잠재 공간 정량 평가 (Intra-class Variance & Empirical Center Distance)
# ----------------------------------------------------
print("\n📊 잠재 공간 연산 중 (z_class & z_content)...")
zs = {c: [] for c in range(NUM_CLASSES)}
lat_z_content = []
lat_z_class = []
labels = []

with torch.no_grad():
    for x, y, _ in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        _, _, _, z_content, z_class_raw = model(x, y)

        lat_z_content.append(z_content.cpu())
        lat_z_class.append(z_class_raw.cpu())
        labels.append(y.cpu())

        for zi, yi in zip(z_class_raw, y):
            zs[int(yi.item())].append(zi.cpu().numpy())

lat_z_content = torch.cat(lat_z_content).numpy()
lat_z_class = torch.cat(lat_z_class).numpy()
labels = torch.cat(labels).numpy()

# Intra-class variance 계산
intra_var = {}
for c in zs:
    if len(zs[c]) > 0:
        arr = np.stack(zs[c])
        intra_var[CLASSES[c]] = float(np.mean(np.var(arr, axis=0)))
    else:
        intra_var[CLASSES[c]] = 0.0

# 경험적 클래스 중심(Empirical Center) 계산 및 L2-Norm 정규화
empirical_centers = {}
for c in range(NUM_CLASSES):
    if len(zs[c]) > 0:
        mean_vec = np.mean(np.stack(zs[c]), axis=0)
        # L2 정규화 적용
        empirical_centers[c] = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
    else:
        empirical_centers[c] = np.zeros(32)

# Center Distance 계산
dists = {}
for i in range(NUM_CLASSES):
    for j in range(i + 1, NUM_CLASSES):
        d = np.linalg.norm(empirical_centers[i] - empirical_centers[j])
        dists[f"{CLASSES[i]} vs {CLASSES[j]}"] = float(d)

avg_dist = np.mean(list(dists.values()))
margin = 2.0  # 기본 마진

# 정량 지표 JSON 파일로 저장
metric_save_path = os.path.join(OUTPUT_FOLDER, "baseline_latent_metrics.json")
with open(metric_save_path, "w", encoding="utf-8") as f:
    json.dump({
        "intra_class_variance": intra_var,
        "pairwise_center_distances": dists,
        "average_center_distance": float(avg_dist),
        "margin": margin
    }, f, indent=2)
print(f"Saved: {metric_save_path}")

# ----------------------------------------------------
# 6. PCA / t-SNE 시각화 수행
# ----------------------------------------------------
print("\n🎨 PCA & t-SNE 시각화 생성 중...")

def plot_2d(feats, labels, title, save_path):
    plt.figure(figsize=(7, 6))
    for cls_id, cls_name in CLASSES.items():
        idx = labels == cls_id
        plt.scatter(
            feats[idx, 0],
            feats[idx, 1],
            s=10, alpha=0.7,
            label=cls_name
        )
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

# PCA
pca = PCA(n_components=2)
pca_zc = pca.fit_transform(lat_z_content)
pca_zcls = pca.fit_transform(lat_z_class)

plot_2d(pca_zc, labels, "Baseline: PCA of z_content", os.path.join(OUTPUT_FOLDER, "baseline_pca_z_content.png"))
plot_2d(pca_zcls, labels, "Baseline: PCA of z_class_raw", os.path.join(OUTPUT_FOLDER, "baseline_pca_z_class.png"))

# t-SNE
tsne = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42)
tsne_zc = tsne.fit_transform(lat_z_content)
tsne_zcls = tsne.fit_transform(lat_z_class)

plot_2d(tsne_zc, labels, "Baseline: t-SNE of z_content", os.path.join(OUTPUT_FOLDER, "baseline_tsne_z_content.png"))
plot_2d(tsne_zcls, labels, "Baseline: t-SNE of z_class_raw", os.path.join(OUTPUT_FOLDER, "baseline_tsne_z_class.png"))
print("✅ PCA / t-SNE 생성 완료")

# ----------------------------------------------------
# 7. Grad-CAM 시각화 수행
# ----------------------------------------------------
print("\n🔥 Grad-CAM 시각화 분석 중...")

ANALYSIS_MEAN = torch.tensor([0.456]).view(1,1,1).to(DEVICE)
ANALYSIS_STD  = torch.tensor([0.224]).view(1,1,1).to(DEVICE)

# Hook & Grad-CAM 핵심 구현
feature_map = None
def forward_hook(module, input, output):
    global feature_map
    feature_map = output
    feature_map.retain_grad()

def get_gradcam(clf_model, img_norm, target_class):
    global feature_map
    feature_map = None
    target_layer = clf_model.model.layer4
    hook = target_layer.register_forward_hook(forward_hook)

    out = clf_model(img_norm.unsqueeze(0))
    if target_class is None:
        target_class = out.argmax(1).item()

    clf_model.zero_grad()
    out[0, target_class].backward()
    hook.remove()

    fmap = feature_map.detach().cpu().numpy()[0]
    grad = feature_map.grad.detach().cpu().numpy()[0]
    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * fmap[c]

    cam = np.maximum(cam, 0)
    # PyTorch F.interpolate를 사용하여 resize (OpenCV 대체)
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    cam_tensor = F.interpolate(cam_tensor, size=(img_norm.size(1), img_norm.size(2)), mode='bilinear', align_corners=False)
    cam = cam_tensor.squeeze().numpy()

    cam -= cam.min()
    cam /= cam.max() + 1e-7
    return cam

# 이미지 로드 및 분석 진행
image_pattern = os.path.join(OUTPUT_FOLDER, "baseline_from_*.png")
found_images = glob.glob(image_pattern)

if not found_images:
    print("⚠ 경고: 생성된 baseline MRI 파일(baseline_from_*.png)을 찾을 수 없습니다. Grad-CAM 생성을 생략합니다.")
else:
    gen_img_path = found_images[0]
    print(f" - 분석 대상 생성 이미지: {os.path.basename(gen_img_path)}")
    
    # 이미지 전처리
    img_pil = Image.open(gen_img_path).convert("L")
    img_tensor = transforms.ToTensor()(img_pil)
    img_norm = (img_tensor.to(DEVICE) - ANALYSIS_MEAN) / ANALYSIS_STD

    # 분류기 모델 로드
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"❌ 에러: 분류기 가중치 {CLASSIFIER_PATH} 가 존재하지 않아 Grad-CAM을 생성을 중단합니다.")
    else:
        clf = SimpleClassifier().to(DEVICE)
        clf.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE), strict=False)
        clf.eval()

        with torch.no_grad():
            logits = clf(img_norm.unsqueeze(0))
            pred_id = logits.argmax(1).item()
            pred_name = CLASSES[pred_id]

        cam = get_gradcam(clf, img_norm, pred_id)

        # 오버레이 이미지 시각화
        img_np = img_tensor.squeeze().numpy()
        cmap = plt.get_cmap('jet')
        heatmap = cmap(cam)[:, :, :3]  # RGB 채널만 획득 (H, W, 3), 값 범위 [0, 1]
        
        img3 = np.stack([img_np, img_np, img_np], axis=-1)  # (H, W, 3)
        overlay = 0.4 * heatmap + 0.6 * img3
        overlay = np.clip(overlay, 0.0, 1.0)


        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Baseline Generated Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Baseline Grad-CAM ({pred_name})")
        plt.axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        gradcam_save_path = os.path.join(OUTPUT_FOLDER, "baseline_gradcam_result.png")
        plt.savefig(gradcam_save_path)
        plt.close()
        print(f"Saved: {gradcam_save_path}")

print("\n✨ 모든 결측 결과 파일 생성 완료!")
print("="*60)
