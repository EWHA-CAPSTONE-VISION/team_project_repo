import pandas as pd
import datasets
import os
import zipfile
from huggingface_hub import snapshot_download, login
import scanpy as sc
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

def download_hest(patterns, local_dir, ids_to_query, download_all, folders):
    repo_id = 'MahmoodLab/hest'

    if not download_all:
      allow_patterns = []
      for fid in ids_to_query:
          for folder in folders:
              allow_patterns.append(f"{folder}/{fid}[._]*")
      snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, repo_type="dataset", local_dir=local_dir)
    else:
      snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping cell vit segmentation...')
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
            path_zip = os.path.join(seg_dir, filename)
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)

def download_organs(dir, organs):
    """
    - dir: str, local dir path where you want to download hest file
    - organs: list, list of organs witch you want to download
    """
    
    print("1. Download meta data...")
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

    DOWNLOAD_ALL = False        # 전체 폴더 다운받을지 or 일부 폴더만 다운받을지
    FOLDERS = ['metadata', 'st', 'patches'] # 일부 폴더만 다운받을 경우 받을 폴더 목록 설정(현재 /metadata, /st, /patches 폴더만)

    # Filter the dataframe by organ, oncotree code...
    meta_organs = meta_df[meta_df['organ'].isin(organs)]

    print("2. get sample ids...")
    ids_to_query = meta_organs['id'].values
    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    print(f"sample ids: {ids_to_query}")

    print("3. Download Hest-1k data...")
    download_hest(list_patterns, dir, DOWNLOAD_ALL, FOLDERS)

    return ids_to_query
   
def preprocess_hest(ids):
    """
    - ids: list, downloaded sample ids (for pre-processing)
    """
    for id in tqdm(ids):
        # AnnData
        adata = sc.read_h5ad(f"{local_dir}/st/{id}.h5ad")
        adata.layers["raw"] = adata.X.copy() # raw data adata.raw_counts에 백업

        adata.var_names_make_unique()
        if adata.raw is not None:
            adata.raw.var_names_make_unique()

        # metadata
        with open(f"{local_dir}/metadata/{id}.json") as f:
            meta = json.load(f)

        # metadata -> adata.obs에 추가
        organ = meta.get('organ')

        disease_state = meta.get('disease_state')
        if disease_state in ['Tumor', 'Cancer']:
            disease_state = 1
        elif disease_state == 'Healthy':
            disease_state = 0
        else:
            disease_state = ""

        oncotree_code = meta.get('oncotree_code')
        species = meta.get('species')

        adata.obs['sample_id'] = id
        adata.obs['organ'] = organ
        adata.obs['disease_state'] = disease_state
        adata.obs['oncotree_code'] = oncotree_code
        adata.obs['species'] = species

        # filtering spots & genes
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)

        # normalization
        sc.pp.normalize_total(adata, inplace=True)  # Normalizing to median total counts
        sc.pp.log1p(adata)  # Logarithmize the data

        # HVG
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']].copy()

        adata.write_h5ad(f"{local_dir}/processed/{id}.h5ad")

def preprocess_stimage(slides):
    for slide in tqdm(slides):
        coord_path = Path(f"{local_dir}/coord/{slide}_coord.csv")
        count_path = Path(f"{local_dir}/gene_exp/{slide}_count.csv")
        img_path = Path(f"{local_dir}/image/{slide}.png")

        # 1. coord 파일 전처리
        coord = pd.read_csv(coord_path, index_col=0)
        coord = coord.rename(columns={'yaxis': 'Y', 'xaxis': 'X'}) # HEST-1K 요구 맞춰 수정
        spot_diameter = coord.iloc[0]["r"]*2 # coord에 반지름 존재

        # 바코드 형식으로 인덱스 수정
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

        # 2. count 파일 전처리
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

        # 3. 공통 spot merge
        common_spots = count.index.intersection(coord.index)

        count = count.loc[common_spots]
        coord = coord.loc[common_spots, ['X', 'Y']].values

        # 4. AnnData 생성
        adata = sc.AnnData(count)
        adata.obsm['spatial'] = coord

        # obs column
        spatial = pd.DataFrame(
            adata.obsm['spatial'], 
            index=pd.Index(adata.obs.index, name='spot'),
            columns=['pxl_col_in_fullres', 'pxl_row_in_fullres']
        )

        # spatial 생성
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

        # 6. 이미지 처리
        img = Image.open(img_path)
        img_down = img.resize((max(1, img.width//10), max(1, img.height//10)))
        img_array = np.array(img_down)

        # uns에 추가
        adata.uns['spatial'] = {
            'ST': {
               'images': {
                    'downscaled_fullres': {'imgdata': img_array}
               }
            }
        }

        # 7. adata 파일로 저장
        adata.write_h5ad(f"{local_dir}/st/{slide}.h5ad")

    for slide in tqdm(slides):
        # AnnData
        adata = sc.read_h5ad(f"{local_dir}/st/{slide}.h5ad")
        adata.layers["raw"] = adata.X.copy() # raw data adata.raw_counts에 백업

        adata.var_names_make_unique()
        if adata.raw is not None:
            adata.raw.var_names_make_unique()

        # metadata -> adata.obs에 추가
        organ = meta[meta["slide"]==slide]["tissue"].iloc[0]
        disease_state = meta[meta["slide"]==slide]["involve_cancer"].iloc[0]
        species = "human" # 전처리시 human만 남김
    
        if disease_state == True:
            disease_state = 1
        elif disease_state == False:
            disease_state = 0
        else:
            disease_state = ""
    
        adata.obs['sample_id'] = slide
        adata.obs['organ'] = organ
        adata.obs['disease_state'] = disease_state
        adata.obs['species'] = species

        # zero spot/gene filtering
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)

        # normalization
        sc.pp.normalize_total(adata, inplace=True)  # Normalizing to median total counts
        sc.pp.log1p(adata)  # Logarithmize the data

        # HVG
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']].copy()

        adata.write_h5ad(f"{local_dir}/processed/{slide}.h5ad")
        

if __name__=="__main__":
    login(token="YOUR TOKEN")

    local_dir='../../hest_data' # 데이터 root 폴더
    organs = ["bowel"]

    ids = download_organs(local_dir, organs)

    preprocess_hest(ids)
