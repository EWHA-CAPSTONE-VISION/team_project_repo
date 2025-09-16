# Project-Senario (Version 0.1 Draft)

|항목|내용|
|---|---|
|프로젝트명|멀티모달 의료&바이오 이미지 데이터를 이용한 경량화된 설명 가능한 tumor 예측 AI 연구|
|프로젝트 키워드|Computer Vision, Multimodal, Medical AI|
|트랙|연구|
|프로젝트 멤버|류다현, 박은서, 유지혜|
|팀지도교수|황의원 교수님|
|무엇을 만들고자 하는가|주어진 데이터를 기반으로 tumor 여부를 정확히 진단할 뿐만 아니라 임상적 이점(설명 가능성, 범용성 등)을 갖는 설명 가능한 의료 AI|
|고객|multimodal data를 활용해 환자의 병명을 정확히 진단하고, 진단 과정을 설명할 수 있어야 하는 병원과 보건소 등의 의료기관|
|Pain Point|의료 AI는 높은 정확성과 더불어 설명가능성 확보가 중요하나, 기존 의료 AI 연구는 단순히 예측 성능 향상에만 주목하고 있다. 더불어 기존의 모델들은 image data에 관한 예측 성능이 떨어지며 범용성 또한 떨어진다는 문제점을 안고 있으며 모델이 무거워 성능을 끌어올리는 데에 하드웨어적 한계도 존재한다. 이러한 점들은 의료 AI의 성능을 끌어올리는 데에 있어 핵심적인 과제로 주목받고 있다.|
|사용할 소프트웨어 패키지의 명칭과 핵심기능/용도|- MuSeGNN(scRNAseq+transcriptomics 통합 활용, gene-gene relationship 학습) <br> - scBERT(scRNAseq 기반 LLM을 활용해 gene-gene interaction 학습 및 cell type annotation 수행) <br> - Gene2vec(co-expression 패턴 기반 gene embedding 제공) <br> - CS-CORE(scRNAseq 데잍 기반 co-expression 예측하는 R 패키지) <br> - scTransform(scRNAseq data normalization, variance stabilization하는 R 패키지) <br> - Celcomen(인과적 식별성을 보장하는 공간전사체(ST) 데이터 기반 모델) <br> - Scanpy(single-cell data 분석하는 python 패키지)|
|사용할 소프트웨어 패키지의 명칭과 URL|- MuSeGNN(https://github.com/HelloWorldLTY/MuSe-GNN) <br> - scBERT(https://github.com/TencentAILabHealthcare/scBERT) <br> - Gene2vec(https://github.com/jingcheng-du/Gene2vec) <br> - CS-CORE(https://github.com/ChangSuBiostats/CS-CORE) <br> - scTransform(https://github.com/satijalab/sctransform) <br> - Celcomen(https://github.com/Teichlab/celcomen) <br> - Scanpy(https://github.com/scverse/scanpy)|
|팀그라운드룰|https://github.com/EWHA-CAPSTONE-VISION/team_project_repo/blob/main/GroudRule.md|
|최종수정일|2025.09.16|
