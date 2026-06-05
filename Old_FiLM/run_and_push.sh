#!/bin/bash
# 스크립트 도중 에러가 발생하면 즉시 실행을 중단합니다.
set -e

echo "=============================================="
echo "🚀 1. CVAE 모델 학습 및 검증 시작"
echo "=============================================="
python cvae.py

echo "=============================================="
echo "📊 2. Latent Space 평가지표 계산"
echo "=============================================="
python metrics/latent_metric.py

echo "=============================================="
echo "🖼️ 3. Latent Space 시각화 (PCA & t-SNE)"
echo "=============================================="
python visualization/latent_viz.py

echo "=============================================="
echo "🔍 4. Grad-CAM 시각화 실행"
echo "=============================================="
python visualization/gradcam_viz.py

echo "=============================================="
echo "✨ 모든 결과 생성 완료! Git 업로드 시작"
echo "=============================================="
cd ..

# 1) 시각화 및 평가 수치 결과 폴더 등록
git add Old_FiLM/evaluation_results/

# 2) [선택] 새로 학습한 CVAE 모델 가중치 파일(.pth)도 LFS로 업로드하고 싶다면 아래 주석(#)을 해제하세요.
# git add -f Old_FiLM/checkpoints/

# 3) 커밋 및 푸시
git commit -m "feat: update CVAE training results and visualizations (Grad-CAM, PCA, t-SNE)"
git push origin main

echo "=============================================="
echo "✅ Git push 완료!"
echo "=============================================="
