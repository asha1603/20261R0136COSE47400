import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
import timm  # timm 라이브러리 추가
from sklearn.model_selection import KFold  # Cross-validation을 위한 KFold 추가 (선택적)
from sklearn.metrics import classification_report, confusion_matrix
import random

# --- 1. 설정 및 하이퍼파라미터 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'data'
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
OUTPUT_DIR = 'classification_results'

CLASSES = {
    'Non Demented': 0,
    'Very mild Dementia': 1,
    'Mild Dementia': 2,
}
NUM_CLASSES = len(CLASSES)
IMAGE_CHANNEL = 1
MODEL_NAME = 'resnet18'  # 사용할 Pretrained 모델 이름

os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f'best_classifier_{MODEL_NAME}_weights_42.pth')


# --- 2. 데이터셋 클래스 정의 ---
class OasisCVAEDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None):
        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.subject_ids = []

        def extract_subject_id(filename):
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:2])
            return parts[0]

        for class_name, label_index in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(label_index)
                    self.subject_ids.append(extract_subject_id(filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)


# --- 3. 이미지 변환 정의 ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.224]),
])


# --- 4. Pretrained 분류 모델 정의 ---
class PretrainedClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        return self.model(x)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 5. 학습 및 검증 함수 ---
def train(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(data_loader.dataset)


def validate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(data_loader.dataset)
    val_accuracy = correct_predictions / len(data_loader.dataset)
    return val_loss, val_accuracy, all_labels, all_predictions


# --- 6. 메인 실행 ---
if __name__ == '__main__':
    set_seed(42)
    full_dataset = OasisCVAEDataset(DATA_DIR, CLASSES, transform=transform)

    if len(full_dataset) > 0:
        subject_id_to_class = {}
        for i, sub_id in enumerate(full_dataset.subject_ids):
            subject_id_to_class[sub_id] = full_dataset.labels[i]

        subjects_by_class = {c: [] for c in CLASSES.values()}
        for sub_id, label in subject_id_to_class.items():
            subjects_by_class[label].append(sub_id)

        train_ratio = 0.8
        val_ratio = 0.2
        train_subjects = set()
        val_subjects = set()

        for label, subjects in subjects_by_class.items():
            np.random.shuffle(subjects)
            num_val = max(1, int(val_ratio * len(subjects)))
            val_split_subjects = subjects[:num_val]
            train_split_subjects = subjects[num_val:]

            train_subjects.update(train_split_subjects)
            val_subjects.update(val_split_subjects)

        train_indices = [i for i, sub_id in enumerate(full_dataset.subject_ids) if sub_id in train_subjects]
        val_indices = [i for i, sub_id in enumerate(full_dataset.subject_ids) if sub_id in val_subjects]

        # -----------------------------------------------------
        # ⭐️ 학습(Train) 환자 ID별 클래스 분포 확인
        # -----------------------------------------------------
        print("\n--- ⭐️ 학습(Train) 환자 ID별 클래스 분포 확인 ⭐️ ---")
        train_subject_class_map = {}
        reverse_classes = {v: k for k, v in CLASSES.items()}

        for i in train_indices:
            sub_id = full_dataset.subject_ids[i]
            true_label_index = full_dataset.labels[i]
            if sub_id not in train_subject_class_map:
                train_subject_class_map[sub_id] = reverse_classes[true_label_index]

        train_class_counts = {}
        for class_name in train_subject_class_map.values():
            train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1

        print(f"  총 환자 수: {len(train_subject_class_map)}명")
        print("\n  [요약] 학습 세트 클래스별 환자 수:")
        for class_name, count in train_class_counts.items():
            print(f"  - {class_name}: {count}명")
        print("-" * 50)

        # -----------------------------------------------------
        # ⭐️ 검증(Validation) 환자 ID별 클래스 분포 확인
        # -----------------------------------------------------
        print("\n--- ⭐️ 검증(Validation) 환자 ID별 클래스 분포 확인 ⭐️ ---")
        val_subject_class_map = {}
        reverse_classes = {v: k for k, v in CLASSES.items()}

        for i in val_indices:
            sub_id = full_dataset.subject_ids[i]
            true_label_index = full_dataset.labels[i]
            if sub_id not in val_subject_class_map:
                val_subject_class_map[sub_id] = reverse_classes[true_label_index]
            elif val_subject_class_map[sub_id] != reverse_classes[true_label_index]:
                print(f"경고: 환자 ID {sub_id}에 대해 클래스 레이블이 다릅니다. 확인 필요!")

        for sub_id, class_name in sorted(val_subject_class_map.items()):
            print(f"  환자 ID: {sub_id:<15} -> 클래스: {class_name}")

        class_counts = {}
        for class_name in val_subject_class_map.values():
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\n  [요약] 검증 세트 클래스별 환자 수:")
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count}명")
        print("-" * 50)

        # -----------------------------------------------------
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

        all_subject_ids = list(subject_id_to_class.keys())
        total_subjects = len(all_subject_ids)

        print(f"데이터 로드 완료. 총 환자 수: {total_subjects}명")
        print(f"환자 ID 기반 분할: 학습 ({len(train_subjects)}명, {len(train_dataset)} 이미지), "
              f"검증 ({len(val_subjects)}명, {len(val_dataset)} 이미지)")

        model = PretrainedClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        best_val_accuracy = 0.0
        print(f"\n--- {MODEL_NAME} 분류 모델 학습 시작 (환자 ID 분할 적용) ---")

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy, _, _ = validate(model, val_loader, criterion)

            print(f"Epoch {epoch}/{NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  ✅ Best model 저장됨. Val Acc: {best_val_accuracy:.4f} -> {BEST_MODEL_PATH}")

        print("\n--- 학습 완료 ---")
        print(f"최고 검증 정확도: {best_val_accuracy:.4f}, 가중치 저장 경로: {BEST_MODEL_PATH}")

    else:
        print("경고: 실제 데이터가 없어 모델 정의만 진행합니다. 학습 및 검증은 불가능합니다.")

        # --- 7. 학습된 모델 로드 및 최종 평가 (추가된 부분) ---
    if len(full_dataset) > 0:
        print("\n" + "="*50)
        print(f"--- 🚀 최종 평가 시작: {MODEL_NAME} ---")
        print("="*50)

        # 1. 모델 인스턴스 생성 및 가중치 로드
        final_model = PretrainedClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)
        try:
            final_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
            print(f"✅ {BEST_MODEL_PATH} 가중치 로드 성공.")
        except FileNotFoundError:
            print(f"❌ 오류: 학습된 가중치 파일 {BEST_MODEL_PATH}를 찾을 수 없습니다.")
            print("   먼저 학습을 완료하여 파일을 저장하거나, 정확한 경로를 확인해주세요.")
            # 파일이 없을 경우, 평가를 진행하지 않고 종료
            exit() 
        except Exception as e:
            print(f"❌ 가중치 로드 중 오류 발생: {e}")
            exit()

        # 2. 검증 데이터셋에 대한 평가 수행
        print("\n➡️ 검증 데이터셋(Validation Set)에 대한 분류 수행...")
        val_loss_final, val_accuracy_final, all_labels, all_predictions = validate(final_model, val_loader, criterion)
        
        # 3. 결과 출력
        print("\n" + "-"*50)
        print(f"⭐ 최종 검증 손실 (Val Loss): {val_loss_final:.4f}")
        print(f"⭐ 최종 검증 정확도 (Val Accuracy): {val_accuracy_final:.4f}")
        print("-" * 50)
        
        # 4. 상세 분류 보고서 및 혼동 행렬 출력
        reverse_classes = {v: k for k, v in CLASSES.items()}
        # NUM_CLASSES (4)를 사용하여 0, 1, 2, 3 인덱스 순서대로 클래스 이름을 가져옵니다.
        reverse_classes_list = [reverse_classes[i] for i in range(NUM_CLASSES)]

        print("\n--- 📊 분류 상세 보고서 (Classification Report) ---")
        print(classification_report(all_labels, all_predictions, target_names=reverse_classes_list, digits=4))

        print("\n--- 🔢 혼동 행렬 (Confusion Matrix) ---")
        cm = confusion_matrix(all_labels, all_predictions)
        # np.array를 문자열로 보기 좋게 변환
        cm_str = np.array2string(cm, separator=', ', prefix='[', suffix=']')
        print("    [예측된 클래스]")
        print(f"    {cm_str}")
        print("    [실제 클래스]")
        
        print("="*50)