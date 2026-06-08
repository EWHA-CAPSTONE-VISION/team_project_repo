# MOSAIC-ST: Multi-mOdal-Spatial AI for Cross-modal learning

26-1 이화여자대학교 컴퓨터공학과 졸업 프로젝트(캡스톤디자인과창업프로젝트A)를 위한 9조 SiYa의 repository입니다.

## ✔️ Project Overview

- **연구 주제**: 멀티모달 의료&바이오 데이터를 이용한 tumor 예측 XAI 모델
- **문제 정의**: 현 의료 AI는
  - 단일 모달리티(이미지·유전자) 의존,
  - 추론 속도·모델 크기·해석 가능성의 한계
    
  를 동시에 해결하지 못하는 경우가 많습니다.
  
  본 프로젝트는 WSI(조직 이미지) + Spatial Transcriptomics(ST) 유전자 발현 정보를 통합해
  “실제 임상 환경에서도 사용 가능한 경량·해석가능 모델”을 구현합니다.

## ✔️ Repository Structure
```
├── analysis              # UMAP 생성
    ├── train_UMAP_per_spot.py
    └── train_UMAP_per_spot_modality.py
├── configs/              # Experiment configuration files (YAML)
    ├── train.yaml            # train config 스크립트
    └── train_ablation.yaml   # ablation study config 스크립트
├── dataset/              # 데이터셋 로딩 및 전처리 모듈
    ├── extract_patches_st.py # ST patch 추출 스크립트
    ├── loader.py             # 데이터 로드 스크립트
    ├── preprocesing.py       # 전처리 스크립트
    ├── repreprocessing.py    # 재전처리 스크립트
    └── unify_hest_st.py      # 데이터셋 형식 통일 스크립트
├── models/               # 모델 아키텍처 정의
    ├── model.py              # 기본 모델 스크립트
    ├── model_ablation.py     # ablation 모델 스크립트
    ├── model_ver1.py         # Late fusion 방식 모델 스크립트
    ├── model_ver2.py         # Early fusion 방식 모델 스크립트
    ├── performer_pytorch.py  # scBERT 스크립트 (수정)
    └── reversible.py         # scBERT 스크립트 (수정)
├── Project-Scenario.md   # 프로젝트 시나리오 및 연구 기획
├── README.md             # 프로젝트 개요 (본 문서)
├── rquirements.txt       # 환경 설정
├── train.py              # 기본 학습 실행 스크립트
├── train_ablation.py     # Ablation study 학습 스크립트
├── train_ver1.py         # Late fusion 방식 학습 스크립트
├── train_ver2.py         # Early fusion 방식 학습 스크립트
├── test.py               # 기본 평가 실행 스크립트
└── test_ablation.py      # Ablation study 평가 스크립트
```

## ✔️ Project Status & Current Progress

| 단계 | 설명 | 진행도 |
| ---- | ---- | ---- |
| **0. 프로젝트 주제 확정** | 문제 정의 및 목적 설정 | 완료 |
| **1. 데이터셋 조사 및 전처리** | 데이터셋 후보 조사, 구조 분석 | 완료 |
| **2. Baseline 모델 설계 및 구현** | 멀티모달 dual-encoder 구성 및 성능 비교 | 완료 |
| **3. 모델 개선 / XAI 적용** | 경량화·성능 개선·downstream task 실험 | 완료 |

## ✔️ Data & Method

본 프로젝트는 다음과 같은 기술 스택을 기반으로 합니다.

**Data**
- Whole Slide Image(WSI) & Spatial Transcriptomics(ST) gene expression paired 데이터 사용
- 공개 멀티모달 의료 데이터 데이터셋 HEST-1K, STimage-1K4M 활용([HEST-1k](https://huggingface.co/datasets/MahmoodLab/hest), [STimage-1K4M](https://huggingface.co/datasets/jiawennnn/STimage-1K4M))

**Modeling**
- Image Encoder(CNN, ViT, etc.)
- Gene Encoder(scBERT, etc.)
- Multimodal Fusion(단순 concat, cosine similarity 기반, etc.)

**Explainability(설명 가능성)**
- Attention heatmap
- Patch-level feature importance
- Gene-level attribution 분석

## ✔️ how to use

**1. install**
Git을 이용해 레포지토리를 다운로드합니다.
```
git clone https://github.com/EWHA-CAPSTONE-VISION/team_project_repo.git
cd team_project_repo

```

**2. build**
가상환경을 생성한 후 필요한 패키지를 설치합니다.
```
conda create -n mosaic-st python=3.10
conda activate mosaic-st

pip install -r requirements.txt

```
데이터셋은 `dataset/` 디렉토리를 참고해 준비합니다. 

**3. test**
준비된 데이터셋을 이용하여 모델이 정상적으로 동작하는지 확인합니다.
```
python train.py

```
학습 과정에서는 training loss, validation loss, evaluation metrics 등이 출력되며, 모델 체크포인트가 저장됩니다.
실험 설정에 따라 다른 버전의 모델도 실행할 수 있습니다.

[demo](https://drive.google.com/file/d/1UIKalObZKYBMv4wdXMrD-Rd9u6l0R9ih/view?usp=sharing)

저장된 체크포인트를 활용하면 매번 재학습할 필요 없이 ablation study를 실행하거나 새로운 데이터에 대해 예측해볼 수 있습니다.
위 데모에서 활용된 체크포인트는 용량 이슈로 본 레포지토리에 업로드되지 않았으며, 필요시 메일로 컨택부탁드립니다.

## ✔️ Team Members

| 이름   | GitHub | Profile |
|--------|--------|---------|
| 류다현 | [@rlyryu](https://github.com/rlyryu) | <img src="https://avatars.githubusercontent.com/github-id1" width="50" height="50"> |
| 박은서 | [@oukl31](https://github.com/oukl31) | <img src="https://avatars.githubusercontent.com/github-id2" width="50" height="50"> |
| 유지혜 | [@jihyeyoo](https://github.com/jihyeyoo)   | <img src="https://avatars.githubusercontent.com/jihyeyoo" width="50" height="50"> |

---


