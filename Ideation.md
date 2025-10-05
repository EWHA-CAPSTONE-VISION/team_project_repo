# Project Ideation (Version 0.9 Draft)

## 1. Team Background
- 공통 관심사: Computer Vision, biomedical AI


## 2. Keywords
- Bio-Imaging
- Robust Computer Vision
- Image segmentation, detection
- Few-shot Learning
- Spatial transcriptomics (ST)
  - 공간 정보(좌표)와 해당 위치에서 발현된 RNA 데이터가 pair된 데이터
- Multimodal AI 
- XAI (Explainable AI)
- Tumor Prediction


## 3. Project Direction
- **누구를 위해?**  
  의생명 데이터를 다루는 개인 및 단체, 임상 현장에서의 의료인


- **누구의 어떤 문제 해결을 위해?**
  - 데이터 수 부족: 희귀 질환 연구, 고비용 실험으로 인해 충분한 데이터를 확보하기 어려움
  - 환경적 변동성: 기기 차이, 조직 slicing 품질, 사용자 간 실험 편차로 인해 분석 불안정
  - 복잡한 데이터 해석: 조직 이미지와 RNA 발현 데이터를 동시에 고려해야 하는 multimodal 분석 난이도
  - 수작업 부담: 병리 이미지 기반 암 진단은 전문가의 판단으로 충분히 가능하지만, 대량의 데이터를 일일이 판독해야 하는 부담 존재
    - 이때 사람의 피로 및 편차로 인한 오류 가능성 또한 무시할 수 없음
  
  -> 높은 정확도로 자동화 가능한 도구 필요 
  -> 적은 데이터로도 안정적으로 학습하고 환경 변화에 강건하며 인간이 이해 가능한 방식으로 설명할 수 있는 AI 모델이 필요

  위와 같이 tumor 예측에서의 **바이오 이미징 데이터 처리 문제**를 해결하고, 나아가 암 진행도나 종양 subtype 예측 등 downstream task를 해결


- **어떤 기술을 사용해서?**  
  - Few-shot Learning: 데이터가 적은 상황에서도 일반화 가능한 학습 기법
  - Noise-robust CV: 조직 이미지의 변동성과 노이즈에 강건한 표현 학습
  - Representation Learning: 효율적 feature embedding 학습습
  - Multimodal Foundation Models: ST 데이터(이미지 + RNA expression의 multimodal 데이터)를 학습하는 대규모 사전학습 모델
  - Explainable AI (XAI): 모델의 예측 결과를 사용자가 이해할 수 있도록 시각화 및 해석


- **연구 파이프라인**
  1. 데이터 수집
     - 공개된 ST 데이터셋 확보(e.g., HEST-1K, SCimage-1K4M, TCGA 등)
     - 조직 이미지(H&E stain) + Gene Expression Matrix
  2. 전처리
     - ST 데이터: spot별 정규화, gene symbol을 문장으로 변
     - 이미지: patch 단위 분할 및 정규화, 색상 표준화
  4. 모델 학습
     - Contrastive learning 기반 이미지-유전자 multimodal embedding 학
     - Few-shot 학습 및 domain adaptation 적용 
     - noise-robust training으로 안정성 향
  5. 모델 경량화
     - Knowledge distillation(지식 증류), pruning, quantization 등 활용
     - 의료 현장에서 활용 가능한 경량 모델 추출
  6. Downstream Task 성능 비교
     - 종양 subtype 분류 Project Ideation (Version 0.9 Draft)

## 1. Team Background
- 공통 관심사: Computer Vision, biomedical AI


## 2. Keywords
- Bio-Imaging
- Robust Computer Vision
- Image segmentation, detection
- Few-shot Learning
- Spatial transcriptomics (ST)
  - 공간 정보(좌표)와 해당 위치에서 발현된 RNA 데이터가 pair된 데이터
- Multimodal AI 
- XAI (Explainable AI)
- Tumor Prediction


## 3. Project Direction
- **누구를 위해?**  
  의생명 데이터를 다루는 개인 및 단체, 임상 현장에서의 의료인


- **누구의 어떤 문제 해결을 위해?**
  - 데이터 수 부족: 희귀 질환 연구, 고비용 실험으로 인해 충분한 데이터를 확보하기 어려움
  - 환경적 변동성: 기기 차이, 조직 slicing 품질, 사용자 간 실험 편차로 인해 분석 불안정
  - 복잡한 데이터 해석: 조직 이미지와 RNA 발현 데이터를 동시에 고려해야 하는 multimodal 분석 난이도
  - 수작업 부담: 병리 이미지 기반 암 진단은 전문가의 판단으로 충분히 가능하지만, 대량의 데이터를 일일이 판독해야 하는 부담 존재
    - 이때 사람의 피로 및 편차로 인한 오류 가능성 또한 무시할 수 없음
  
  -> 높은 정확도로 자동화 가능한 도구 필요 
  -> 적은 데이터로도 안정적으로 학습하고 환경 변화에 강건하며 인간이 이해 가능한 방식으로 설명할 수 있는 AI 모델이 필요

  위와 같이 tumor 예측에서의 **바이오 이미징 데이터 처리 문제**를 해결하고, 나아가 암 진행도나 종양 subtype 예측 등 downstream task를 해결


- **어떤 기술을 사용해서?**  
  - Few-shot Learning: 데이터가 적은 상황에서도 일반화 가능한 학습 기법
  - Noise-robust CV: 조직 이미지의 변동성과 노이즈에 강건한 표현 학습
  - Representation Learning: 효율적 feature embedding 학습습
  - Multimodal Foundation Models: ST 데이터(이미지 + RNA expression의 multimodal 데이터)를 학습하는 대규모 사전학습 모델
  - Explainable AI (XAI): 모델의 예측 결과를 사용자가 이해할 수 있도록 시각화 및 해석


- **연구 파이프라인**
  1. 데이터 수집
     - 공개된 ST 데이터셋 확보(e.g., HEST-1K, SCimage-1K4M, TCGA 등)
     - 조직 이미지(H&E stain) + Gene Expression Matrix
  2. 전처리
     - ST 데이터: spot별 정규화, gene symbol을 문장으로 변
     - 이미지: patch 단위 분할 및 정규화, 색상 표준화
  4. 모델 학습
     - Contrastive learning 기반 이미지-유전자 multimodal embedding 학
     - Few-shot 학습 및 domain adaptation 적용 
     - noise-robust training으로 안정성 향
  5. 모델 경량화
     - Knowledge distillation(지식 증류), pruning, quantization 등 활용
     - 의료 현장에서 활용 가능한 경량 모델 추출
  6. Downstream Task 성능 비교
     - 종양 subtype 분류
     - 암 진행도 예측


## 4. Expected Outcomes  
  적은 데이터로도 안정적으로 학습·분석 가능한 **ST 데이터 기반 multimodal 경량 XAI 모델**의 개발을 기본 목표로 하여, 구체적으로는:
  - 데이터 수집-전처리-모델 학습-모델 경량화-downstream task에서의 성능 비교로 이어지는 파이프라인
  - 기기, 측정 환경, 사용자에 따른 영향을 적게 받는 범용성과 신뢰성을 갖춘 모델 제안


