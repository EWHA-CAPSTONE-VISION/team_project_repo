import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from hest import STHESTData

# =======================================================
# 1. STimage-1K4M Meta data
# =======================================================

# meta data 다운로드
local_path = hf_hub_download(
    repo_id="jiawennnn/STimage-1K4M",      # repo ID
    filename="meta/meta_all_gene02122025.csv",  # repo 내 path
    repo_type="dataset",
    local_dir="./stimage",            # 저장할 위치
    local_dir_use_symlinks=False,
)

print(local_path)

# meta data processing
meta_raw = pd.read_csv("stimage/meta/meta_all_gene02122025.csv")
meta = meta_raw[meta_raw["species"]=="human"][["slide", "tissue", "pmid", "involve_cancer", "tech"]]
# 필요시 여기서 tissue 종류 필터링

# =======================================================
# 2. STimage-1K4M data
# =======================================================

# 전체 데이터셋 다운로드 - 용량 확인 후 진행
snapshot_download(
    repo_id="jiawennnn/STimage-1K4M",
    repo_type="dataset",
    local_dir="./stimage",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("다운로드 완료")

# 일부 데이터셋 다운로드
# 확인용, 실제 이용시 slide와 tech는 meta의 티슈를 기준으로 정리해 for 문으로 아래 과정 반봅
slide = meta.loc[0, "slide"]    # slide 이름
tech = meta.loc[0, "tech"]      # tech 종류

snapshot_download(
    repo_id="jiawennnn/STimage-1K4M",
    repo_type="dataset",
    local_dir="./stimage",          # 로컬 경로 설정
    local_dir_use_symlinks=False,
    allow_patterns=[
        f"{tech}/coord/{slide}_coord.csv",    # slide 내의 spot 위치 정보
        f"{tech}/gene_exp/{slide}_count.csv", # spot 별 유전자 발현량 raw data
        f"{tech}/image/{slide}.png"           # slide H&E image
    ],
)

# =======================================================
# 3. STimage-1K4M to HEST-1K
# =======================================================
coord_path = Path(f"./stimage/{tech}/coord/{slide}_coord.csv")
count_path = Path(f"./stimage/{tech}/gene_exp/{slide}_count.csv")
img_path = Path(f"./stimage/{tech}/image/{slide}.png")

# 1) 바코드 형식으로 인덱스 수정 (coord, count)
# coord -- column 이름 수정, 바코드 형식 인덱스 수정
coord = pd.read_csv(coord_path, index_col=0)
coord = coord.rename(columns={'yaxis': 'Y', 'xaxis': 'X'}) # HEST-1K 요구 맞춰 수정
spot_diameter = coord.iloc[0]["r"]*2 # coord에 반지름 존재

new_idx_crd = []
for idx in coord.index:
    parts = idx.rsplit('_', 1)
    if len(parts) == 2 and 'x' in parts[1]:
        row, col = parts[1].split('x')
        new_idx = f"{int(row):03d}x{int(col):03d}"
        new_idx_crd.append(new_idx)
    else:
        new_idx_crd.append(idx)       
coord.index = new_idx_crd

# count -- 바코드 형식 인덱스 수정
count = pd.read_csv(count_path, index_col=0)
new_idx_cnt = []
for idx in count.index:
    parts = idx.rsplit('_', 1)
    if len(parts) == 2 and 'x' in parts[1]:
        row, col = parts[1].split('x')
        new_idx = f"{int(row):03d}x{int(col):03d}"
        new_idx_cnt.append(new_idx)
    else:
        new_idx_cnt.append(idx)       
count.index = new_idx_cnt

# imgae -- resize, array로 변경
img = Image.open(img_path)
img_down = img.resize((max(1, img.width//10), max(1, img.height//10)))
img_array = np.array(img_down)

# 2) 공통 spot merge
common_spots = count.index.intersection(coord.index)

count = count.loc[common_spots]
coord = coord.loc[common_spots, ['X', 'Y']].values

# 3) AnnData
# 기본 anndata 생성
adata = sc.AnnData(count)
adata.obsm['spatial'] = coord

# obs column
spatial = pd.DataFrame(
        adata.obsm['spatial'], 
        index=pd.Index(adata.obs.index, name='spot'),
        columns=['pxl_col_in_fullres', 'pxl_row_in_fullres']
)

# spatial 생성 및
array_rows = []
array_cols = []
for idx in spatial.index:
    try:
        row, col = str(idx).split('x')
        array_rows.append(int(row))
        array_cols.append(int(col))
    except:
        array_rows.append(0)
        array_cols.append(0)
    
spatial['array_row'] = array_rows
spatial['array_col'] = array_cols
    
# obs에 추가
adata.obs = adata.obs.join(spatial)
adata.obs['in_tissue'] = True

# uns에 추가
adata.uns['spatial'] = {
    'ST': {
        'images': {
            'downscaled_fullres': {'imgdata': img_array}
        }
    }
}

# 4) STHESTData로 변환
# 메타데이터 설정
pixel_size = 0.5 # WSI 픽셀 크기
spot_dist = 150.0 # ST는 일반적으로 150-200 픽셀

meta = {
    'pixel_size_um_estimated': pixel_size,
    'pixel_size_um_embedded': pixel_size,
    'fullres_height': img.height,
    'fullres_width': img.width,
    'spots_under_tissue': len(adata.obs),
    'spot_diameter': spot_diameter,   
    'inter_spot_dist': spot_dist                 
}

# HEST 객체 생성
st_data = STHESTData(adata, img_array, pixel_size, meta)