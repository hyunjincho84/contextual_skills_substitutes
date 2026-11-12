Python의 연도별 SAE 표현 퍼짐을 공고 단위로 계산합니다.

평균 pairwise cosine similarity는 응집도이며, 엄밀한 직경(maximum pairwise distance)이
아닙니다. 그림에는 `mean_cosine_distance = 1 - mean_pairwise_cosine`을 사용합니다.
값이 높으면 같은 연도에 속하는 공고들의 표현이 더 퍼져 있다는 뜻입니다.

계산 순서는 Python SAE instance 벡터의 공고별 평균 → 각 공고 벡터 L2 정규화 →
동일 연도 내 서로 다른 공고 쌍의 cosine 평균입니다. 연도 전체를 먼저 평균하면
연도 안의 분포가 사라지므로 이 계산에서는 공고별 벡터를 유지합니다. 기존 overlap의
Top-128 선택이나 공통 feature 제거는 적용하지 않고 고정된 전체 4,096차원을 씁니다.

정규화 벡터를 u_i, 유효 공고 수를 n이라 하면:

```text
mean_pairwise_cosine = (||sum_i u_i||^2 - n) / (n * (n - 1))
mean_cosine_distance = 1 - mean_pairwise_cosine
```

이는 모든 i<j 쌍을 직접 계산한 평균과 같으며, 자기 자신과의 cosine은 제외됩니다.
N×N 행렬이나 쌍 표본 추출을 사용하지 않습니다. 계산량은 O(n × feature_dimension)입니다.
norm이 0인 공고 벡터는 cosine이 정의되지 않으므로 별도로 기록합니다.

실행:

```bash
/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python -B tmp/python_yearly_dispersion.py
```

`BASE_DATA_DIR` 환경변수와 `--base-data-dir`, `--source-run`, `--output-dir`, `--years`,
`--batch-size`, `--dpi`를 사용할 수 있습니다. 재계산에는 새 output directory를 지정합니다.
저장된 CSV에서 그림만 다시 생성하려면 `--stage plot`을 지정합니다.
기본 입력은 `tmp/python_sae_onet_min10/`의 검증 인덱스 및 원래 layer-12 SAE 캐시이며,
기본 출력은 `/home/jovyan/LEM_data3/hyunjincho/LEM/tmp/python_sae_yearly_dispersion/`입니다.

- `yearly_dispersion.csv`: 전체 Python 및 기존 8개 직업별 연도 지표, 공고·instance·쌍 수,
  cosine 합과 분모. 합·분모는 ordered pair 기준이고 평균은 unordered pair 기준과 같습니다.
- `overall_dispersion.png`: 전체 Python 공고의 연도별 평균 cosine 거리와 공고 수.
- `overall_dispersion_only.png`: 평균 cosine 거리 그래프만 별도로 저장한 이미지.
- `onet_dispersion.png`: 기존 8개 O*NET 직업 각각의 연도별 평균 cosine 거리.
- `zero_norm_postings.csv`: cosine 계산에서 제외된 norm=0 공고 ID.
- `selected_occupations.csv`, `run_manifest.json`: 대상 직업, 출처, 설정, 검증·처리 건수, 버전.

이 지표는 연도 안의 분포를 요약합니다. 연도별 중심이 이동해도 퍼짐은 일정할 수 있고,
중심이 유지되어도 퍼짐은 변할 수 있습니다. 전체 Python 결과에는 직업 구성 변화가
영향을 줄 수 있어 직업별 결과를 함께 저장합니다. 공고 쌍들은 서로 독립이지 않으므로,
쌍 개수를 독립 표본 수로 해석하면 안 됩니다. 현재 그림에는 불확실성 구간을 추정하지
않았습니다. 2025년 데이터는 6월 5일까지입니다.

검증:

```bash
/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python -B \
  -m unittest discover -s tmp -p 'test_python_yearly_dispersion.py' -v
```
