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
.
├── configs/                # Experiment configuration files (YAML)
├── dataset/                # 데이터셋 로딩 및 전처리 모듈
├── models/                 # 모델 아키텍처 정의
├── Project-Scenario.md     # 프로젝트 시나리오 및 연구 기획
├── README.md               # 프로젝트 개요 (본 문서)
├── train.py                # 기본 학습 실행 스크립트
├── train_ablation.py       # Ablation study 학습 스크립트
├── test.py                 # 기본 평가 실행 스크립트
└── test_ablation.py        # Ablation study 평가 스크립트
```

## ✔️ Project Status & Current Progress

| 단계 | 설명 | 진행도 |
| ---- | ---- | ---- |
| **0. 프로젝트 주제 확정** | 문제 정의 및 목적 설정 | 완료 |
| **1. 데이터셋 조사 및 전처리** | 데이터셋 후보 조사, 구조 분석 | 완료 |
| **2. Baseline 모델 설계 및 구현** | 멀티모달 dual-encoder 구성 및 성능 비교 | 완료 |
| **3. 모델 개선 / XAI 적용** | 경량화·성능 개선·downstream task 실험 | 진행 중 |

## ✔️ Data & Method

본 프로젝트는 다음과 같은 기술 스택을 기반으로 합니다.

**Data**
- Whole Slide Image(WSI)
- Spatial Transcriptomics(ST) gene expression
- 공개 멀티모달 의료 데이터셋(e.g., HEST-1k, STimage-1K4M)

**Modeling**
- Image Encoder(CNN, ViT, etc.)
- Gene Encoder(scBERT, etc.)
- Multimodal Fusion(단순 concat, cosine similarity 기반, etc.)

**Explainability(설명 가능성)**
- Attention heatmap
- Patch-level feature importance
- Gene-level attribution 분석

## ✔️ Team Members

| 이름   | GitHub | Profile |
|--------|--------|---------|
| 류다현 | [@unhyepnhj](https://github.com/unhyepnhj) | <img src="https://avatars.githubusercontent.com/github-id1" width="50" height="50"> |
| 박은서 | [@oukl31](https://github.com/oukl31) | <img src="https://avatars.githubusercontent.com/github-id2" width="50" height="50"> |
| 유지혜 | [@jihyeyoo](https://github.com/jihyeyoo)   | <img src="https://avatars.githubusercontent.com/jihyeyoo" width="50" height="50"> |

---


