# Project Ideation (Version 0.9 Draft)

## 1. Team Background
- 공통 관심사: Computer Vision, biomedical AI


## 2. Keywords
- Bio-Imaging
- Robust Computer Vision
- Image segmentation, detection
- Few-shot Learning
- Spatial transcriptomics (ST)
  - 조직 이미지와 해당 조직에서 발현된 RNA 데이터가 pair된 이미지-텍스트 데이터
- Multimodal AI 
- XAI (Explainable AI)


## 3. Project Direction
- **누구를 위해?**  
  바이오/의학/AI 연구자와 의료진, 의료인공지능 기업 등 바이오 데이터를 다루는 개인 및 단체


- **누구의 어떤 문제 해결을 위해?**
  - 데이터 수 부족: 희귀 질환 연구, 고비용 실험으로 인해 충분한 데이터를 확보하기 어려움
  - 환경적 변동성: 기기 차이, 조직 slicing 품질, 사용자 간 실험 편차로 인해 분석 불안정
  - 복잡한 데이터 해석: 조직 이미지와 RNA 발현 데이터를 동시에 고려해야 하는 multimodal 분석 난이도
      
  -> 따라서, 적은 데이터로도 안정적으로 학습하고 환경 변화에 강건하며 인간이 이해 가능한 방식으로 설명할 수 있는 AI 모델이 필요함

  위와 같은 **바이오 이미징 데이터 처리 문제**를 해결하고, 나아가 tissue 이미지나 RNA 데이터 분석 등 수동으로 수행하기 어려운 downstream task를 해결


- **어떤 기술을 사용해서?**  
  - Few-shot Learning: 데이터가 적은 상황에서도 일반화 가능한 학습 기법
  - Noise-robust CV: 조직 이미지의 변동성과 노이즈에 강건한 표현 학습
  - Generative Models (Diffusion, GAN, VAE, etc.): 데이터 증강 및 representation learning 강화
  - Multimodal Foundation Models: ST 데이터(이미지 + RNA expression의 multimodal 데이터)를 학습하는 대규모 사전학습 모델
  - Explainable AI (XAI): 모델의 예측 결과를 사용자가 이해할 수 있도록 시각화 및 해석


- **연구 파이프라인**
  1. 데이터 수집
     - 공개된 ST 데이터셋 확보(e.g., Visium 등)
     - 조직 이미지(H&E stain) + RNA 행렬 
  2. 전처리
     - 이미지 정규화 및 품질 보정
     - RNA 데이터 정규화 및 특징 추출
     - 이미지-유전자 좌표 매칭
  3. 모델 학습
     - Contrastive learning 기반 multimodel embedding(이미지 <-> RNA)
     - Few-shot 학습을 위한 meta-learning 기법 도입
     - Domain adaptation 및 noise-robust training 적용
  4. 모델 경량화
     - Knowledge distillation(지식 증류), pruning, quantization 등 활용
     - 의료 현장에서 활용 가능한 경량 모델 추출
  5. Downstream Task 성능 비교
     - 세포/조직 segmentation
     - 제포 유형 annotation
     - 질병 진단 및 예후 예측


## 4. Expected Outcomes  
  적은 데이터로도 안정적으로 학습·분석 가능한 **바이오 이미징 처리 모델/시스템**의 개발을 기본 목표로 하여, 구체적으로는:
  - ST 기반 multimodal foundation model
  - 데이터 수집-전처리-모델 학습-모델 경량화-downstream task에서의 성능 비교로 이어지는 파이프라인
  - 기기, 측정 환경, 사용자에 따른 영향을 적게 받는 범용성과 신뢰성을 갖춘 모델 제안


