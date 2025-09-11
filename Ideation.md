# Project Ideation (Version 0.9 Draft)

## 1. Team Background
- 공통 관심사: Computer Vision, biomedical AI

## 2. Keywords
- Bio-Imaging
- Robust Computer Vision
- Image segmentation, detection
- Few-shot Learning
- Spatial transcriptomics
- multimodal AI
- XAI

## 3. Project Direction
- **누구를 위해?**  
  연구자와 의료진, 의료인공지능 기업 등 바이오 데이터를 다루는 개인 및 단체

- **누구의 어떤 문제 해결을 위해?**  
  데이터 수가 적거나(e.g., 희귀 질환, 고비용 실험) 환경 영향(e.g., 기기 차이, 슬라이싱 품질)에 민감하여 분석이 어려운 **바이오 이미징 데이터 처리 문제**
  나아가, tissue 이미지나 RNA 데이터 분석 등 수동으로 수행하기 어려운 downstream task를 해결

- **어떤 기술을 사용해서?**  
  multimodal 데이터인 ST 데이터를 Few-shot Learning, Noise-robust CV, Generative Models 등 컴퓨터 비전 기법으로 처리  
  *Spatial transcriptomics(ST, 공간전사체): 조직 이미지와 해당 조직에서 발현된 RNA 데이터가 pair된 이미지-텍스트 데이터

- **무얼 만들려고 하는가?**  
  적은 데이터로도 안정적으로 학습·분석 가능한 **바이오 이미징 처리 모델/시스템**의 개발을 기본 목표로 하여, 구체적으로는:
  1. ST 기반 multimodal foundation model
  2. 데이터 수집-전처리-모델 학습-모델 경량화-downstream task에서의 성능 비교로 이어지는 파이프라인
  3. 기기, 측정 환경, 사용자에 따른 영향을 적게 받는 범용성 있고 설명 가능한 의료 AI


