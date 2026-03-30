# Team-09
| (1) 과제명 |  Multimodal Integration of Spatial Transcriptomics and Whole Slide Images for Tumor Prediction |
|:---  |---  |
| (2) 팀 번호/이름 | 09-SiYa |
| (3) 구성원 | 박은서(2303050): 리더, 외부 데이터셋(STImage) 전처리 파이프라인 구축(STImage 데이터 수집 및 Raw Data 정제, HEST 데이터셋 포맷과 호환성을 위해 구조 표준화), 프로젝트 성과 발표 <br> 류다현(2376087): 팀원,외부 데이터셋(HEST) 전처리 파이프라인 구축(HEST 데이터 수집 및 Raw Data 정제, 데이터 로딩을 위한 초기 로더 구현, 베이스라인 모델 파일 구조 설계) <br> 유지혜(2271040): 팀원,초기 기술 타당성 검증(전체 구조도 기반 핵심 학습 파이프라인 개발, 초기 프로토타입 코드 작성), 기술 장표 제작 |
| (4) 지도교수 | 황의원 교수 |
| (5) 트랙  | 연구 |
| (6) 과제 키워드 | Computer Vision, Multimodal, Medical AI |
| (7) 과제 내용 요약 | 본 연구는 spatial transcriptomics(ST)와 whole slide image(WSI)를 결합한 멀티모달 학습을 통해 종양 예측 성능을 향상시키고, 각 모달리티 간 상호작용을 분석하는 것을 목표로 한다. 기존 병리 AI 연구는 주로 단일 모달리티 정보에 의존하거나, 멀티모달을 활용하더라도 단순 성능 향상에 집중하여 모달리티 간 상호작용 및 설명 가능성에 대한 분석이 부족하다는 한계를 지닌다. 이를 해결하기 위해, 본 연구에서는 ST와 WSI를 동시에 입력으로 사용하는 종양 예측 task를 설정하고, 2가지 encoder 구성(early-fusion, late-fusion)과 4가지 fusion 구조(attention, concatenation, gating, similarity 기반)를 비교 분석한다. 각 구조는 서로 다른 방식으로 두 모달리티 간의 상호작용을 모델링하며, 이를 통해 어떤 결합 방식이 성능 향상에 효과적인지를 평가한다. 또한 ST의 spatial coordinate 정보를 활용하여, 모델이 예측에 활용한 영역과 유전자를 시각화함으로써 설명 가능성을 확보하며, 이를 통해 기존 WSI 기반 모델 대비 공간적 정보와 유전자 발현 정보를 함께 고려한 해석이 가능함을 보인다. |
| (8) 주요 Link | https://github.com/EWHA-CAPSTONE-VISION/team_project_repo |
