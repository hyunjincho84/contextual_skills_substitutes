Python의 마지막 BERT layer에 해당하는 SAE activation을 사용해, O*NET 직업별
연도 간 Top-128 overlap heatmap을 만듭니다. 기본값은 2010–2025년 모든 연도에
고유 공고가 10개 이상인 8개 직업입니다. 미분류 `99-9999.00`은 제외합니다.

실행:

```bash
bash tmp/run_python_onet_overlap.sh
```

환경변수 `BASE_DATA_DIR`, `PYTHON_BIN` 또는 CLI로 경로를 바꿀 수 있습니다.

```bash
bash tmp/run_python_onet_overlap.sh \
  --years 2010-2025 --min-postings-per-year 10 --expected-occupations 8 \
  --layer 12 --topk 128 --max-iters 100 \
  --output-dir /home/jovyan/LEM_data3/hyunjincho/LEM/tmp/python_sae_onet_min10
```

계산 순서는 각 Python masking instance의 SAE 벡터 → 공고 ID별 평균 →
O*NET 코드·게시 연도별 공고 평균입니다. 따라서 같은 공고에 Python instance가
여러 개 있어도 공고 하나의 총 비중은 1입니다. 연도는 원본 `posted`에서 가져온
감사(audit) metadata의 `posted_year`를 사용합니다. 공고별로 평균하는 것은 SAE를
통과한 벡터이며, BERT 벡터를 먼저 평균한 뒤 SAE에 넣는 방식이 아닙니다.

Overlap은 `sparse_auto_encoder/yearly/overlap_by_year.py`의 함수를 직접 사용합니다.
직업별로 각 연도 평균 벡터의 Top-128을 구하고, 모든 분석 연도에 공통인 feature를
반복 제거하고 보충합니다. 공통 feature가 없어지면 연도 쌍의 집합 교집합 크기를
계산합니다. 기존과 같은 `RdBu_r` 색상과 0–128 척도를 사용합니다. 제거는 각 직업
안에서 수행하므로 직업 간 제외 feature 집합은 다를 수 있습니다.

기본 입력:

- `sparse_auto_encoder/python_yearly/overlaps/features_by_layer/`의 전체 Python 캐시
- `predictions_clean/bert_pred/pred/`의 해당 캐시 생성 당시 prediction 파일
- `tmp/python_onet_scale/python_posting_metadata_references.csv.gz`의 원본 공고 metadata

GPU를 사용하거나 모델을 재학습하지 않습니다. Legacy 캐시에 문장 hash가 없으므로
기존 추출 순서를 재현해 인덱스의 연속 sample ID, 파일·행·연도 순서, 전체 instance
수와 공고별 instance 수를 대조합니다. 캐시 이후 수정된 prediction 파일은 거부합니다.
`row_idx` 단독으로 연결하지 않고 `(원본 file_path, row_idx)`를 거쳐 원본 공고 ID에
연결합니다. 검증 실패 시 중단하며, 조용히 instance를 누락하지 않습니다.

출력 기본 경로는 `BASE_DATA_DIR/tmp/python_sae_onet_min10/`입니다.

```text
selected_occupations.csv
posting_counts_by_onet_year.csv
validated_instance_index.parquet
posting_representations/year_2010.parquet ... year_2025.parquet
group_means.npz
overlap_longform.csv
top_feature_ids_by_onet_year.csv
feature_removal_summary.csv
run_manifest.json
15-1252.00/
  overlap_heatmap_large.png
  overlap_matrix_layer_12.csv
  excluded_feature_ids.json
... (8개 O*NET 코드)
```

각 그림의 y축에 연도별 고유 공고 수가 표시됩니다. `feature_removal_summary.csv`에는
직업별 제외 feature 수와 최종 Top-K에 포함된 0-activation feature 수도 기록됩니다.
기존 overlap 기준을 유지하므로 0-activation을 임의로 제거하거나 다른 지표로 바꾸지
않습니다. 각 연도의 공고 수 10개라는 기준은 탐색용 표본 기준이며 통계적 안정성을
보장하지는 않습니다. 2025년 원본 데이터는 6월 5일까지입니다.

숫자 집계는 float64로 누적하고, 공고별 벡터는 float32로 저장합니다. 연도별 평균은
저장된 공고별 벡터에서 float64로 계산합니다. 경로, checkpoint 출처, seed(기존 캐시),
파일 signature, 코드 hash, 라이브러리 버전과 누락·실패 건수는 manifest에 기록됩니다.

글꼴·해상도 등을 바꾸어 그림만 다시 만들려면:

```bash
bash tmp/run_python_onet_overlap.sh --stage plot --dpi 300 --annotation-font-size 9
```

전체 재계산에는 새 `--output-dir`을 지정합니다. 기존 결과를 자동으로 삭제하지 않습니다.
작은 결정적 샘플 검증은 다음과 같이 실행합니다.

```bash
/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python \
  -m unittest discover -s tmp -p 'test_python_onet_yearly_overlap.py' -v
```
