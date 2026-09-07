# ptz_calib_dy — PTZ-Calib Python 재구현

[PTZ-Calib (ICRA 2025)](https://arxiv.org/pdf/2502.09075) 공개 C++ 코드(`../PTZ-Calib/src/`)를
구조·기능 그대로 Python으로 옮긴 구현이다. 결과 JSON 스키마가 동일하므로 논문 평가
스크립트(`../PTZ-Calib/scripts/eval_*.py`)를 수정 없이 재사용한다.

## 설치

```bash
cd ptz_calib_dy
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy opencv-python-headless==4.10.0.84 shapely
```

`opencv-python-headless`는 C++ 원본이 사용한 OpenCV 4.5.3과 동작을 맞추기 위해 4.x로 고정한다.

## 실행

C++ 쪽 `run_*.sh`와 인자·순서가 동일하다.

```bash
bash run_ptzba_synthetic.sh      # 3. PTZ-BA (offline)  + 5. 평가
bash run_reloc_synthetic.sh      # 4. PTZ-Reloc (online) + 5. 평가
bash run_ptzba_worldcup14.sh
bash run_reloc_worldcup14.sh
```

개별 실행:

```bash
.venv/bin/python scripts/run_ptz_ba.py \
    -i ../PTZ-Calib/data/synthetic/offline/scene_01 \
    -f ../PTZ-Calib/data/synthetic/offline_matches/scene_01 \
    -a ../PTZ-Calib/data/synthetic/offline/scene_01/scene_01.json \
    -o output-synthetic-offline [--dist] [--verbose] [--seed 0]

.venv/bin/python scripts/run_ptz_reloc.py \
    --ref_images  ../PTZ-Calib/data/synthetic/offline/scene_01 \
    --ref_features ../PTZ-Calib/data/synthetic/offline_matches/scene_01 \
    --ref_params  output-synthetic-offline/scene_01.json \
    --test_images ../PTZ-Calib/data/synthetic/online/scene_01 \
    --test_features ../PTZ-Calib/data/synthetic/online_matches/scene_01 \
    --output output-synthetic-online
```

C++ 결과와 수치 비교:

```bash
.venv/bin/python scripts/compare_with_cpp.py \
    --pred_dir output-synthetic-offline --ref_dir ../PTZ-Calib/output-synthetic-offline
```

검증:

```bash
.venv/bin/python tests/test_basic.py      # 카메라 모델 / JSON / 트랙 통계
.venv/bin/python tests/test_jacobian.py   # 그룹 수치미분 == 완전 수치미분
```

## 모듈 대응표

| Python | C++ |
|---|---|
| `ptzcalib/types.py` | `src/core/types.{h,cc}` |
| `ptzcalib/data_io.py` | `src/core/data_io.{h,cc}` |
| `ptzcalib/union_find.py` | `src/core/union_find.h` |
| `ptzcalib/tracks.py` | `src/core/tracks.{h,cc}`, `flat_pair_map.h` |
| `ptzcalib/krt_optimizer.py` | `src/core/krt_optimizer.{h,cc}` |
| `ptzcalib/ptzray_optimizer.py` | `src/core/ptzray_optimizer.{h,cc}` |
| `ptzcalib/ptz_incremental_optimizer.py` | `src/core/ptz_incremental_optimizer.{h,cc}` |
| `ptzcalib/lm.py` | 위 두 optimizer가 사용하는 Ceres LM/trust-region 부분 |
| `ptzcalib/os_path.py`, `logging_utils.py` | `src/utils/` |
| `scripts/run_ptz_ba.py` | `src/app/run_ptz_ba.{h,cc}` |
| `scripts/run_ptz_reloc.py` | `src/app/run_ptz_reloc.{h,cc}` |

## 최적화 백엔드 (`lm.py`)

Ceres 대신 `lm.py`에 Ceres 1.14의 동작을 그대로 옮겼다. `scipy.optimize.least_squares`를
쓰지 않은 이유는 accept/reject·종료 조건·damping 규칙이 달라 "수렴 여부"(= 등록 성공 판정)가
C++과 어긋나기 때문이다.

- `minimize()` = `TrustRegionMinimizer::Minimize` + `LevenbergMarquardtStrategy`
  - damping `D = clamp(diag(JᵀJ), 1e-6, 1e32) / radius`, 초기 radius `1e4`
  - jacobi scaling `s = 1/(1+‖J_col‖)` (iteration 0에서 1회 계산)
  - 종료: function/parameter tolerance는 **step 채택 여부와 무관하게** 검사 (Ceres와 동일)
  - `min_trust_region_radius` 도달도 `CONVERGENCE`로 처리 (Ceres와 동일)
- `central_difference_step()` = `NumericDiffCostFunction<CENTRAL>`
  (`delta = max(sqrt(eps), |x|·1e-6)`)
- `DenseNumericSystem` = `DENSE_QR` (KRT 최적화, 자유변수 4~6개)
- `SchurSystem` = `SPARSE_SCHUR` (전역 BA, ray 블록 소거)

### 수치미분 비용

Ceres는 잔차 블록마다 numeric diff를 돌리지만, 여기서는 **서로 다른 잔차에 영향을 주는
파라미터를 한 번에 섭동**해 벡터화한다. 예를 들어 "모든 이미지의 rvec[0]"을 동시에 흔들어도
각 관측은 자기 이미지에만 의존하므로 한 번의 잔차 평가로 전체 열을 얻는다. 30장·2216 트랙
문제에서 자코비안 1회가 약 30–50 ms이고, 결과는 완전 수치미분과 **비트 단위로 일치**한다
(`tests/test_jacobian.py`).
