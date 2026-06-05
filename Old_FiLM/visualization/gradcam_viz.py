import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANALYSIS_MEAN = torch.tensor([0.456]).view(1,1,1).to(DEVICE)
ANALYSIS_STD  = torch.tensor([0.224]).view(1,1,1).to(DEVICE)

CLASSES = {0:'Non Demented',1:'Very mild Dementia',2:'Mild Dementia'}

# ----------------------------------------------------------
# 1) Classification Model
# ----------------------------------------------------------
class PretrainedClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=3, in_chans=1):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans
        )

    def forward(self, x):
        return self.model(x)

def load_generated_image(path):
    img = Image.open(path).convert("L")
    img = transforms.ToTensor()(img)   # shape [1,224,224]
    return img

def load_classifier(model_path):
    model = PretrainedClassifier().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ----------------------------------------------------------
# 2) Grad-CAM
# ----------------------------------------------------------
feature_map = None

def forward_hook(module, input, output):
    global feature_map
    feature_map = output
    feature_map.retain_grad()

def get_gradcam(model, img_norm, target_class):
    global feature_map
    feature_map = None

    target_layer = model.model.layer4
    hook = target_layer.register_forward_hook(forward_hook)

    out = model(img_norm.unsqueeze(0))

    if target_class is None:
        target_class = out.argmax(1).item()

    model.zero_grad()
    out[0, target_class].backward()

    hook.remove()

    fmap = feature_map.detach().cpu().numpy()[0]
    grad = feature_map.grad.detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * fmap[c]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_norm.size(2), img_norm.size(1)))
    cam -= cam.min()
    cam /= cam.max() + 1e-7
    return cam


# ----------------------------------------------------------
# 3) Visualization
# ----------------------------------------------------------
def visualize_gradcam(original_img_01, cam, pred_name, save_path):
    img = original_img_01.squeeze().cpu().numpy()

    heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0

    img3 = np.stack([img, img, img], axis=-1)
    overlay = 0.4*heatmap + 0.6*img3
    overlay = np.uint8(overlay*255)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Generated Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM ({pred_name})")
    plt.axis("off")   

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Grad-CAM image: {save_path}")


# ----------------------------------------------------------
# 4) Main Runner (생성 이미지 전용)
# ----------------------------------------------------------
def run_gradcam_on_generated_image(gen_img_tensor, classifier_path, save_path):

    # load classifier
    clf = load_classifier(classifier_path)

    # normalize
    img_norm = (gen_img_tensor.to(DEVICE) - ANALYSIS_MEAN) / ANALYSIS_STD

    # classification
    with torch.no_grad():
        logits = clf(img_norm.unsqueeze(0))
        pred_id = logits.argmax(1).item()
        pred_name = CLASSES[pred_id]

    print(f"Predicted class: {pred_name}")

    # compute grad-cam
    cam = get_gradcam(clf, img_norm, pred_id)

    # save visualization
    visualize_gradcam(
        original_img_01=gen_img_tensor.cpu(),
        cam=cam,
        pred_name=pred_name,
        save_path=save_path
    )


# ----------------------------------------------------------
# 5) 🔥 실제 호출
# ----------------------------------------------------------
if __name__ == '__main__':
    import glob
    import os

    # 스크립트 파일 기준으로 경로 탐색
    VIS_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(VIS_DIR) # Old_FiLM 디렉토리

    # 생성된 실제 이미지 파일 경로 자동 검색
    image_pattern = os.path.join(BASE_DIR, "evaluation_results", "GEN_SAMPLES", "from_*.png")
    found_images = glob.glob(image_pattern)

    if not found_images:
        print("❌ 오류: Old_FiLM/evaluation_results/GEN_SAMPLES 폴더에 이미지 파일(from_*.png)이 없습니다. CVAE 모델을 먼저 실행해주세요.")
    else:
        # 첫 번째 검색된 이미지 선택
        gen_img_path = found_images[0]
        print(f"🔍 발견된 이미지로 Grad-CAM 분석 진행: {os.path.basename(gen_img_path)}")
        gen_img = load_generated_image(gen_img_path)
        
        # 저장 경로 설정 (동일 폴더 내에 gradcam_result.png 로 저장)
        save_path = os.path.join(os.path.dirname(gen_img_path), "gradcam_result.png")
        classifier_path = os.path.join(BASE_DIR, "classification_results", "best_classifier_resnet18_weights_42.pth")

        run_gradcam_on_generated_image(
            gen_img_tensor = gen_img, 
            classifier_path = classifier_path,
            save_path = save_path
        )
