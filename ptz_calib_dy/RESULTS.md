# 재현 검증 결과

Python 포팅본으로 사전 검증 5단계를 그대로 수행한 결과. 기준선은 동일 머신에서 돌린
C++ 공개 코드 출력(`../PTZ-Calib/output-*`)이다.

- 실행일: 2026-08-05
- 환경: Python 3.11.15 / numpy 2.4.6 / scipy 1.17.1 / opencv-python-headless 4.10.0
- 기준선: C++ (OpenCV 4.5.3 + Ceres 1.14.0)

## 1. C++ 기준선 대비 수치 일치

`scripts/compare_with_cpp.py` — 카메라별 `|Δf|`(px), `ΔR`(deg), `Δt`, `Δdist`.

| 실행 | 카메라 수 | Δf mean / max | ΔR mean / max (deg) | Δt mean / max | Δdist max |
|---|---|---|---|---|---|
| Synthetic offline (10 씬) | 300 / 300 일치 | 0.0004 / 0.0074 | 0.00010 / 0.00130 | 0.00009 / 0.00093 | 0 |
| Synthetic online (10 씬) | 1500 / 1500 일치 | 0.0004 / 0.0075 | 0.00010 / 0.00132 | 0.00009 / 0.00093 | 0 |
| WorldCup14 offline (4 캠) | 67 / 67 일치 | 0.0010 / 0.0078 | 0.00001 / 0.00004 | 0.00003 / 0.00017 | 0 |
| WorldCup14 online (7 캠) | 108 / 108 일치 | 0.0005 / 0.0049 | 0.00000 / 0.00005 | 0.00002 / 0.00017 | 0 |

- 등록에 성공한 이미지 **집합**이 모든 실행에서 C++과 완전히 동일 (`+p`/`+r` = 0).
- 잔차는 LM 종료 허용오차 수준(초점거리 1e-2 px 미만). 10개 씬 중 8개는 `Δf < 1e-5`,
  즉 사실상 비트 단위로 동일하다. scene_08 / GER_POR / SUI_FRA 만 1e-3 대 편차가 남는데,
  이는 gauge freedom(ray 스케일·전역 회전)이 있는 문제에서 LM 경로가 미세하게 갈린 결과다.

## 2. 정확도 (논문 대비)

### Synthetic — offline (Table IV)

| Scene | N | FLE mean/med (px) | APE_rot mean/med (deg) | APE_trans mean/med (m) |
|---|---|---|---|---|
| scene_01 | 30 | 0.56 / 0.48 | 0.19 / 0.19 | 0.09 / 0.09 |
| scene_02 | 30 | 0.97 / 0.85 | 0.07 / 0.06 | 0.06 / 0.06 |
| scene_03 | 30 | 1.14 / 1.18 | 0.08 / 0.07 | 0.05 / 0.05 |
| scene_04 | 30 | 0.67 / 0.47 | 0.09 / 0.11 | 0.04 / 0.04 |
| scene_05 | 30 | 4.56 / 4.36 | 0.12 / 0.12 | 0.09 / 0.09 |
| scene_06 | 30 | 0.70 / 0.64 | 0.14 / 0.14 | 0.04 / 0.04 |
| scene_07 | 30 | 0.69 / 0.48 | 0.08 / 0.08 | 0.03 / 0.03 |
| scene_08 | 30 | 0.57 / 0.48 | 0.09 / 0.09 | 0.06 / 0.06 |
| scene_09 | 30 | 0.77 / 0.66 | 0.15 / 0.13 | 0.06 / 0.06 |
| scene_10 | 30 | 0.73 / 0.67 | 0.06 / 0.07 | 0.03 / 0.03 |
| **전체(pool)** | **300** | **1.14 / 0.75** | **0.11 / 0.10** | **0.06 / 0.05** |

### 종합 비교

| 실험 | 지표 | 논문 | C++ 기준선 | Python | 판정 |
|---|---|---|---|---|---|
| Synthetic offline | FLE (px) | 1.52 / 0.97 | 1.14 / 0.75 | **1.14 / 0.75** | 일치 |
| | APE_rot (deg) | 0.12 / 0.11 | 0.11 / 0.10 | **0.11 / 0.10** | 일치 |
| | APE_trans (m) | 0.08 / 0.08 | 0.06 / 0.05 | **0.06 / 0.05** | 일치 |
| Synthetic online | FLE (px) | 1.65 / 1.02 | 1.22 / 0.76 | **1.22 / 0.76** | 일치 |
| | APE_rot (deg) | 0.13 / 0.12 | 0.11 / 0.10 | **0.11 / 0.10** | 일치 |
| | APE_trans (m) | 0.08 / 0.08 | 0.06 / 0.05 | **0.06 / 0.05** | 일치 |
| WorldCup14 online | IoU_part (%) | 96.8 / 97.8 | 96.8 / 97.6 | **96.8 / 97.6** | 일치 |
| | IoU_whole (%) | 93.1 / 94.4 | 93.1 / 94.4 | **93.1 / 94.4** | 일치 |

- Python = C++ 기준선. 따라서 논문값과의 차이(Synthetic FLE 1.14 vs 1.52 등)는 포팅 문제가
  아니라 공개 코드/데이터 자체의 특성이다. Synthetic은 공개 코드가 논문보다 다소 좋게 나오고,
  WorldCup14는 IoU_part median만 0.2%p 낮다.
- 등록 성공률은 전 구간 100% (offline 300/300 · 67/67, online 1500/1500 · 108/108).

## 3. 처리 시간

단일 스레드 기준 (C++은 Ceres `num_threads=32`).

| 단계 | Python | 비고 |
|---|---|---|
| Synthetic offline | 5.0 – 9.3 s / 씬 (30장) | 전역 BA 8~10회 포함 |
| WorldCup14 offline | 1.6 – 2.8 s / 캠 (15~22장) | |
| Synthetic online | 0.04 – 0.13 s / 장 | 논문 Table V: 0.68 s/장 |
| WorldCup14 online | 0.04 – 0.09 s / 장 | |

온라인 단계는 논문 수치보다 빠른데, 논문 값에는 특징 추출·매칭이 포함되고 여기서는 데이터셋에
동봉된 사전 계산 매칭을 쓰기 때문이다(공개 코드와 동일 조건).

## 4. PLAN.md 기재 내용 중 정정 사항

구현 과정에서 C++ 코드를 다시 확인해 아래 두 항목은 계획서 기술과 실제 코드가 달랐다.
**코드 쪽을 따랐다.**

1. **전역 BA의 intrinsics는 이미지 간 공유되지 않는다.**
   PLAN 1-B-4는 "intrinsics 9개를 전체 이미지가 공유"라고 적었으나,
   `PTZRayOptimizer`의 `shared_ic_ids_`는 생성자에서 `iota(0..n)`로 초기화되고
   `SetSharedIntrinsics()`는 어디서도 호출되지 않는다. 즉 **이미지마다 독립 intrinsics 블록**을
   갖는다(PTZ 줌이 프레임마다 다르므로 물리적으로도 이쪽이 맞다).
   실제로 C++ 출력 `scene_01.json`의 30개 카메라 초점거리는 533/799/1167 등 서로 다르다.

2. **왜곡 모델의 p1/p2는 OpenCV와 교차되어 있지 않다.**
   PLAN 0-4는 "코드가 OpenCV와 p1/p2 역할이 교차됨"이라 적었으나,
   `x_dist = x·radial + 2p1·xy + p2(r²+2x²)`, `y_dist = y·radial + 2p2·xy + p1(r²+2y²)` 는
   `cv::projectPoints`의 정의와 **동일**하다.
   다만 별개의 불일치가 하나 있다: `Camera::dist_` 5-vector를 손으로 쓴 왜곡 식은
   `[k1,k2,k3,p1,p2]` 순서로 읽는 반면, 같은 벡터를 그대로 넘기는
   `cv::projectPoints` / `cv::undistortPoints` / 평가 스크립트의 `cv2.undistort`는
   OpenCV 규약인 `[k1,k2,p1,p2,k3]`로 해석한다. 현재는 `k1` 외 전부 0으로 고정되어 결과가
   같지만, `k2`/`p1`/`p2`를 풀면 드러날 잠재 버그다.

## 5. 확인된 논문 ↔ 공개 코드 차이 (포팅 시 코드를 따른 지점)

- 논문 식 (3)은 `k1,k2,p1,p2` 추정을 기술하나 코드는 `k1`만 자유 변수 (`SubsetParameterization`).
- 논문은 robust loss `ρ(·)`를 명시하나 코드는 squared loss + 트랙 길이 `ScaledLoss`
  (Huber는 주석 처리). 트랙 길이 가중치는 **비후보 이미지를 포함한 전체 트랙 길이**를 쓴다.
- 논문 IV-C의 frustum-overlap 기반 레퍼런스 재선택은 코드에 없다 — 매칭 수 최대인 레퍼런스 선택.
- `PTZRayFactor`(PTZ-IBA 전역 BA)는 intrinsics 블록에 왜곡 계수를 갖고 있으면서도 **왜곡을 전혀
  적용하지 않는다**. 따라서 `--dist`는 georeferencing과 온라인 재측위에만 영향을 준다.
- `Reproj2d3dFactor`만 `intrinsics[1]`(fy)을 사용한다. ray factor들은 fx를 fy로도 쓰므로,
  georeferencing에서 fy는 2D-3D 항에만 작용하는 자유 변수가 되고
  `ObtainRefinedCameraParams`가 출력 시 `fy = fx`로 덮어써 폐기된다.
- `PtzIncrementalOptimizer::Solve`의 루프 종료 경로에 return 누락(C++ UB) →
  Python에서는 명시적 실패 반환.
- `RegisterNextImage`의 "이미 등록된 이미지 아님" assert는 release 빌드에서 비활성화되고,
  `FindNextImages()` 결과 리스트가 stale해지면 이미 등록된 이미지를 다시 등록할 수 있다.
  포팅본도 동일하게 동작하도록 두었다.
- `LoadMatchesInfo`가 만드는 `n×n` 테이블의 빈 엔트리는 `src=dst=0, confidence=0`이며,
  모든 순회 루프가 이를 그대로 훑는다(결과에는 영향 없음). 순회 순서(row-major)가
  `RegisterNextImage`의 "첫 매칭 선택"에 영향을 주므로 포팅본도 같은 순서를 유지한다.
- WorldCup14 어노테이션 JSON은 `K`/`R`/`t`/`dist`가 빈 배열이다. C++은 `memcpy` 0바이트로
  기본값(단위행렬)을 유지하므로, 포팅본도 예외 대신 동일하게 처리한다(`_memcpy_into`).

## 6. 재현 절차

```bash
cd ptz_calib_dy
bash run_ptzba_synthetic.sh     # 3 + 5
bash run_reloc_synthetic.sh     # 4 + 5
bash run_ptzba_worldcup14.sh    # 3
bash run_reloc_worldcup14.sh    # 4 + 5

.venv/bin/python scripts/compare_with_cpp.py \
    --pred_dir output-synthetic-offline --ref_dir ../PTZ-Calib/output-synthetic-offline
```
