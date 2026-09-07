# PTZ-Calib Python 재구현 — 전체 절차 (v3)

- **대상**: PTZ-Calib(ICRA 2025) 알고리즘의 Python 재구현본 `ptz_calib_dy/`
- **이 문서의 성격**: **절차서**다. "무엇을 어떤 순서로 하고, 무엇을 보고 통과/실패를 판정하는가"만 적는다.
  실측 수치는 담지 않고 결과 문서로 링크한다 — 데이터셋이 바뀌어도 이 문서는 그대로 유효해야 한다.
- **각 절의 형식**: `절차 → 합격 기준 → 현 상태` (미해결 사안이 있으면 해당 검증 절 안에 **미결** 블록)

| 문서 | 역할 |
| --- | --- |
| `PLAN.md` (이 문서) | 절차와 판정 기준 |
| [README.md](README.md) | 설치·실행법·모듈 대응표·`lm.py` 설계 |
| [RESULTS.md](RESULTS.md) | 0-1 결과 — C++ 기준선 대비 수치 일치, 논문 지표 재현 (2026-08-05) |
| [EVALUATION.md](EVALUATION.md) | **검증 방법의 정본** — 지표 정의, 실행 명령, 판정 체크리스트 |
| [RESULTS-own.md](RESULTS-own.md) | **자체 데이터 실측값의 정본** — 1·2 단계 결과, 남은 잔차의 구조 분석 (2026-08-26 갱신) |
| [worklog/](worklog/) | 세션별 진행 기록·판단 근거 (08-06 / 08-07 / 08-11 / 08-12) |

## 전체 흐름

```
0-1 C++ → Python 포팅 ─────────────── 공개 데이터셋으로 포팅 정확성 확정
                                       └ 이후 ptzcalib/ 는 건드리지 않는다
0-2 Reference Frame 수집 ──────────── 자체 데이터 입력 준비 + 촬영 품질 게이트

1 Offline Stage   1-1 매칭 → 1-2 PTZ-IBA(로컬) → 1-3 지오참조(월드) → 1-4 검증
                   산출: offline.json (프레임별 K, R, t, dist)

2 Online Stage    2-1 테스트 프레임 → 2-2 매칭 → 2-3 재측위 → 2-4 검증
                   산출: online.json (테스트 프레임별 K, R, t, dist)
```

**핵심 원칙 — `ptzcalib/` 는 수정하지 않는다.** 0-1에서 C++ 수치 동등성이 확정되므로, 자체 데이터를
위해 알고리즘 코드를 고치면 그 검증이 무효가 된다. 자체 데이터는 **공개 데이터셋과 동일한 입력 포맷으로
변환**해서 같은 스크립트에 넣는다. 신규 코드는 `scripts/` 아래로 한정한다.

---

## 0. 준비사항

### 0-1. C++ → Python 포팅

**절차**

1. C++ 원본 빌드 후 공개 데이터셋으로 실행해 **기준선 출력**을 만든다
   ```bash
   cd ../PTZ-Calib && bash install_deps.sh && bash build.sh
   bash run_ptzba_synthetic.sh && bash run_reloc_synthetic.sh
   bash run_ptzba_worldcup14.sh && bash run_reloc_worldcup14.sh
   ```
2. 모듈 단위로 포팅한다. 대응 관계는 [README.md](README.md) "모듈 대응표"
3. 최적화 백엔드는 `scipy.optimize`가 아니라 **Ceres 1.14 동작을 옮긴 `ptzcalib/lm.py`**를 쓴다.
   accept/reject·종료 조건·damping이 다르면 "수렴 여부" = **등록 성공 판정**이 C++과 어긋난다
4. 같은 데이터셋을 Python으로 돌린다 (`run_ptzba_*.sh` / `run_reloc_*.sh`)
5. `scripts/compare_with_cpp.py`로 카메라별 `Δf` / `ΔR` / `Δt` / `Δdist` 를 대조한다
6. 논문 평가 스크립트를 **수정 없이** 재사용해 논문 Table 값을 재현한다
   (`../PTZ-Calib/scripts/eval_synthetic.py`, `agg_synthetic.py`, `eval_worldcup.py`) —
   결과 JSON 스키마를 C++과 동일하게 유지했기 때문에 가능하다

**합격 기준**

| 항목 | 기준 |
| --- | --- |
| 등록 이미지 **집합** | C++과 완전 일치 (`compare_with_cpp.py`의 `+p`/`+r` = 0) |
| `Δf` | max < 1e-2 px (LM 종료 허용오차 수준) |
| `ΔR` / `Δt` | max < 1e-2° / 1e-3 |
| 논문 지표 | FLE·APE·IoU가 **C++ 기준선과 동일**할 것. 논문값 자체와의 차이는 공개 코드/데이터 특성이므로 판정 대상이 아니다 |
| 자코비안 | `tests/test_jacobian.py` — 그룹 수치미분 == 완전 수치미분 |
| 모델·IO | `tests/test_basic.py` — 카메라 모델 / JSON / 트랙 통계 |

**현 상태: 완료** (2026-08-05). 실측값과 정정 사항은 [RESULTS.md](RESULTS.md).

**포팅 시 코드를 따른 지점** (논문 ↔ 공개 코드 불일치, 상세는 [RESULTS.md](RESULTS.md) §4–5)

- 논문 식 (3)은 `k1,k2,p1,p2`를 추정하나 코드는 **`k1`만 자유 변수**
- 논문의 robust loss `ρ(·)`는 코드에 없다 — squared loss + 트랙 길이 `ScaledLoss`
- 논문 IV-C의 frustum-overlap 기반 온라인 레퍼런스 재선택은 코드에 없다 — **매칭 수 최대** 선택
- 전역 BA의 intrinsics는 **이미지 간 공유되지 않는다** (프레임마다 줌이 다르므로 물리적으로도 맞다)
- `Camera::dist_` 5-vector 순서 불일치는 **잠재 버그**로 남아 있다 — 손으로 쓴 왜곡 식은
  `[k1,k2,k3,p1,p2]`, 같은 벡터를 받는 OpenCV 함수는 `[k1,k2,p1,p2,k3]`로 읽는다.
  현재 `k1` 외 전부 0이라 결과는 같지만, `k2`/`p1`/`p2`를 풀면 드러난다

**알아둘 제약** (이후 단계의 설계 근거가 되는 항목만)

- `data_io.load_matches_info`는 `findHomography`의 **RANSAC inlier mask를 버리고** 모든 매칭을
  inlier로 취급한다. 손실함수도 squared loss다 → **아웃라이어 제거는 전적으로 매칭 생성 단계 책임**
- 전역 BA는 `TracksBuilder.filter(4)` — 한 점이 **최소 4장에서 관측**되어야 트랙으로 살아남는다.
  0-2의 중첩 설계 기준이 여기서 나온다
- 초기 페어의 두 번째 시드는 **평균 픽셀 시차 ≥ 50px** 조건을 만족해야 한다
- 이미지별 intrinsics 블록이 독립이다 → 프레임마다 줌이 달라도 무방
- `PTZRayFactor`(전역 BA)는 **왜곡을 적용하지 않는다** → `--dist`는 지오참조와 온라인 재측위에만 영향
- `Reproj2d3dFactor`만 `intrinsics[1]`(fy)을 쓴다. 지오참조에서 fy는 2D-3D 항에만 작용하는 자유
  변수이고 출력 시 `fy = fx`로 덮어써 폐기된다 → **어노테이션이 fy를 구속하지 못하면 붕괴한다** (1-3)

---

### 0-2. Reference Frame 수집

레퍼런스 프레임은 offline stage의 **유일한 입력**이다. 여기서 잘못되면 이후 단계에서 복구할 수 없다.

#### (1) 카메라 · 설치

- **강체 고정** — 삼각대 또는 벽 브래킷. 세션 중 마운트가 흔들리면 전 프레임이 오염된다
- 카메라 설정에서 **반드시 끌 것**

  | 설정 | 끄는 이유 |
  | --- | --- |
  | **OSD 오버레이** (타임스탬프/채널명/P·T·Z 리드아웃) | 모든 프레임의 **같은 픽셀**에 찍히므로 자기 자신과 완벽히 매칭되고, 그 매칭들이 항등 호모그래피에 일관돼 **RANSAC이 걸러내지 못한다**. "카메라가 회전하지 않았다"는 제약을 BA에 직접 주입한다 |
  | 렌즈 왜곡 자동 보정 (LDC / dewarp) | 추정 왜곡 모델과 불일치 |
  | 전자식 흔들림 보정 (EIS / DIS) | 프레임마다 미세 이동 → 순수 회전(`C` 공유) 가정을 깬다 |
  | 디지털 줌 | 광학 줌만 사용 |
  | 스마트 코덱 / 가변 크롭 | 프레임마다 해상도가 바뀌면 안 된다 |

  OSD를 끄지 못하면 `extract_match.py --mask` 로 가린다 (`<images>/mask.png` 자동 인식).
- **전 프레임 동일 해상도** — 코드가 `cx, cy = W/2, H/2`를 가정한다
- **스텝-스테어 캡처** — 이동 정지 → 줌/포커스 정착 대기(1~2초) → 스틸 캡처.
  이동 중 캡처(모션블러·롤링셔터) 금지
- 관심 장면이 카메라에서 **충분히 멀 것(수십 m 이상)**. 실제 PTZ는 광학 중심이 아니라 마운트 축을
  중심으로 회전하므로 수 cm의 평행이동 오프셋이 있고, 근거리에서만 유의미한 오차가 된다

#### (2) 촬영 설계

목표 규모 **16~24장** (공개 데이터셋: Synthetic 30장, WorldCup14 15~22장).

- **줌 계층 구조**로 촬영한다 — 트랙 길이 확보에 가장 효과적이다
  - 광각 링: 최광각으로 감시 대상 각도 범위를 팬으로 훑는다 (서로 크게 겹침)
  - 중간 줌: 각 프레임이 광각 프레임 2~3장에 걸치도록
  - 망원: 각 프레임이 광각/중간 프레임 **3장 이상**과 겹치도록
- **중첩률 인접 프레임 간 60% 이상.** 근거 — 전역 BA가 트랙 길이 ≥ 4를 요구하므로 한 점이 4장
  이상에 들어와야 한다. **팬 스텝을 수평 FOV의 25~30% 이하**로 잡으면 충족된다.
  줌 계층 내 인접 프레임에도 같은 기준을 적용한다 (망원일수록 FOV가 좁아 스텝도 좁아야 한다)
- 각 프레임 위치를 **PTZ 프리셋으로 저장** → 재촬영·검증 시 동일 자세 재현
- 프레임마다 **pan/tilt/zoom 값을 CSV로 기록** (`fname,pan,tilt,zoom`) — 1-4·2-4의 quasi-GT.
  가능하면 ONVIF/SDK **리드아웃 실측값**을 쓴다. GOTO **지정값**으로 대신할 수 있지만,
  그 경우 "지정 대비 실제 도달 각도 오차"가 지표에 섞여 들어간다
- **동일 세션에서 한 번에** 촬영 (조명·장면 변화 최소화)

#### (3) 장면 조건

- **정적 장면** — 코드에 이동물체 마스킹이 없다. 차량·보행자가 적은 시간대를 고른다
- **텍스처가 충분한 구조물 위주.** 하늘·수면·균일 벽면만 잡히는 프레임은 배제한다 —
  이런 프레임은 매칭 페어가 0이 되어 **등록 자체가 불가능**하고, 트랙 통계 게이트로는 걸리지 않는다
- **반복 패턴**(동일 창문 배열, 타일, 펜스) 위주 프레임 주의 — 매칭 아웃라이어의 주 원인
- **조명 급변**(역광 ↔ 순광)이 섞이면 매칭 수가 급감한다. 흐린 날/균일 조명 시간대 권장

#### (4) 입력 규격 — 코드가 요구하는 형식 (어기면 조용히 실패한다)

```
data/<dataset>/
  offline/                    # 레퍼런스 이미지. 디렉터리명이 결과 JSON 파일명이 된다
      ptz_frame_001.jpg ...
      mask.png                # (선택) 비-0 = 사용. 이미지 목록에서는 자동 제외된다
      offline.json            # 지오참조 어노테이션 = fit 2D-3D
  offline_matches/
      ptz_frame_001.jpg.txt   # 파일명 = "이미지 파일명 + .txt" (확장자 포함)
      pairs_matches.txt
  online/                     # 테스트 이미지
  online_matches/
  survey/                     # 측량 CSV. make_annotation.py 입력 → offline.json / gt/*.json
      points.csv              # 3D (point_id, X, Y, Z, split=fit|holdout)
      observations.csv        # 2D 클릭 (set, fname, point_id, u, v) — 원시 픽셀
  gt/offline.json, gt/online.json   # 홀드아웃 2D-3D (어노테이션과 동일 스키마)
  ptz_log/offline.csv, ptz_log/online.csv
```

- 이미지 인덱스는 `sorted(listdir)` 순서로 정해진다 → **제로패딩 파일명**(`-001`)을 쓴다
- features 파일: 1행 `num_kpts desc_dim`, 이후 `x y scale orientation desc...`.
  `desc_dim = 0`도 허용되나 **online stage가 레퍼런스 디스크립터를 재사용**하므로 offline은 기록해 둔다
- `pairs_matches.txt`: 블록 사이 빈 줄, 블록 첫 줄 `imgA.jpg imgB.jpg`, 이후 `queryIdx trainIdx`
  - **offline은 양방향 필수** — `matches_info` 테이블이 `[i*n+j]` 한 칸만 채우므로 한 방향만 쓰면
    등록 순서에 따라 페어가 보이지 않는다
  - **online은 `(레퍼런스, 테스트)` 방향** — `find_best_match`가 `pairs[i][1] == 테스트 파일명`으로
    필터링한다. 뒤집으면 전 프레임 실패한다
  - 매칭 **40쌍 미만 페어는 블록을 쓰지 않는다** (공개 데이터셋과 동일 기준)
- 어노테이션 JSON 스키마
  ```json
  {"cameras": {"<확장자 없는 이미지명>": {
      "name": "...", "res": [W, H], "pos": [], "K": [], "R": [], "t": [], "dist": [],
      "marker": {"pix": [[u/W, v/H], ...], "pos": [[X, Y, Z], ...]}}}}
  ```
  - **`pix`는 해상도로 나눈 정규화 좌표다.** 픽셀 좌표를 그대로 넣으면 안 된다
  - `K/R/t/dist`는 빈 배열 허용. 단 **키 자체가 없으면 예외가 아니라 침묵이다** —
    `data_io.py`가 `KeyError`를 삼키고 `return False, [], ...` 하므로 **어노테이션 전체가 빈 것으로
    전달되고 지오참조가 조용히 no-georef로 퇴화한다.** 로그의 `Exception caught: 'K'` 한 줄이 유일한 단서다

#### (5) 촬영 품질 게이트

1-1의 매칭까지 만든 뒤 `<출력>/match_report.json` 으로 **즉시** 판정한다.
여기서 미달이면 이후 단계에 비용을 쓰지 말고 **재촬영**한다.

| 항목 | 기준 | 미달 시 조치 |
| --- | --- | --- |
| 길이 ≥ 4 트랙 수 | **1000개 이상** (공개 `scene_01` = 2216) | 중첩 부족 → 팬 스텝을 줄이거나 광각 프레임 추가 |
| 페어 0인 고립 프레임 | **0장** | 해당 프레임이 무텍스처 → 그 자세를 텍스처 있는 구조물로 다시 잡는다 |
| `near_identity_matches.ratio` | **2% 미만** | OSD 오버레이 잔존 의심 → 카메라 설정 또는 `--mask` |
| 페어당 매칭 수 분포 | median 수십 쌍 이상 | 조명·반복패턴·모션블러 점검 |

**현 상태: 1차 시촬영(22장, 실내) 기준으로 게이트는 통과했으나 (1) 카메라 설정이 닫히지 않았다.**
실측은 [worklog/2026-08-06.md](worklog/2026-08-06.md), 판정 근거는 [EVALUATION.md](EVALUATION.md) §5.

**미결**

- **카메라 설정 미확인** — OSD가 켜진 채 촬영해 `mask.png`로 대응했고(화면의 13.3%),
  LDC/EIS/디지털 줌/스마트 코덱은 켜짐 여부를 확인하지 못했다. 본 촬영 전에 닫아야 한다
- **PTZ 프리셋 미저장** — 동일 자세 재현은 GOTO 지정값 기록으로 대신하고 있다
- **무텍스처 프레임 2장** — 페어 0으로 등록 불가. 트랙 게이트로는 걸리지 않는 결함이라
  위 표에 "고립 프레임 0장" 항목을 별도로 둔다
- **줌 계층 내 중첩 미달** — 중간·망원 tier의 인접 프레임 중첩이 60%에 못 미친다.
  전 프레임이 다른 tier와 겹쳐 트랙 게이트는 통과했으나, 본 촬영에서는 중간/망원 팬 스텝을 절반으로 줄인다
- **PTZ 로그가 리드아웃 실측이 아니라 GOTO 지정값** — quasi-GT로 쓰기에는 충분함을 확인했으나
  (OSD 리드아웃이 남은 프레임과 값 일치), 지정값과 실제 도달 자세의 차이는 분리되지 않는다.
  분리에는 ONVIF/SDK 리드아웃이 필요하다

---

## 1. Offline Stage

레퍼런스 프레임에서 **프레임별 카메라 파라미터**를 추정한다.
1-2까지는 측량 없이 로컬 좌표계로, 1-3에서 2D-3D 측량을 넣어 월드 좌표계로 옮긴다.

### 1-1. 매칭 생성 (`scripts/extract_match.py`)

**절차**

```bash
.venv-match/bin/python scripts/extract_match.py --mode offline \
    -i data/infiniq/offline -o data/infiniq/offline_matches --viz 3
```

- extractor / matcher는 교체 가능한 인터페이스다 —
  `--extractor {superpoint,disk,aliked} --matcher {superglue,lightglue}`.
  **기본은 SuperPoint + SuperGlue(outdoor)** 로, 논문·공개 데이터셋과 동일 조건이어야 실패 원인이
  매칭 때문인지 알고리즘 때문인지 분리된다
- **페어 선정**: offline은 전수 페어 (N ≤ 24면 276페어, GPU로 수 분)
- **아웃라이어 필터는 전부 이 단계에서** 건다 (알고리즘은 RANSAC mask를 버린다 — 0-1 제약)

  | 필터 | 플래그 | 기본값 |
  | --- | --- | --- |
  | matcher confidence 게이트 | `--conf_thresh` | 0.5 |
  | `findHomography(RANSAC)` inlier만 기록 | — | thresh 4.0 |
  | 페어 최소 매칭 수 | `--min_inliers` | 40 |
  | 정적 OSD 마스킹 | `--mask` / `--exclude_rect` | `<images>/mask.png` 자동 |
  | SuperPoint 검출 임계 | `--keypoint_threshold` | **0.01** |

- **`--keypoint_threshold` 기본값을 0.01로 고정한 이유**: SuperPoint 자체 기본값 0.005로 `scene_01`을
  돌리면 **등록 2/30으로 붕괴**한다. 트랙 수는 오히려 늘어나므로 매칭 품질 문제가 아니라,
  **PTZ-IBA 초기 페어 선정이 "매칭 수 최대 페어가 무엇이냐"에 민감**해 시드 2-view 해가 발산하는 것이다
- features 파일의 `scale` 열(코드가 버리는 열)에 SuperPoint 점수를 실어 둔다 →
  online stage가 레퍼런스 특징점을 **재추출 없이** 읽어 인덱스 불일치를 원천 차단한다

**합격 기준**

- 공개 데이터셋 왕복 테스트 — `offline/scene_01`에 이 스크립트를 돌려 만든 매칭으로
  `run_ptz_ba.py`를 실행했을 때 **동봉 매칭 사용 시와 등록률·FLE·APE가 동급**일 것.
  자체 데이터로 넘어가기 전에 **변환기 자체의 버그를 여기서 잡는다**
- 자체 데이터에서는 0-2 (5) 촬영 품질 게이트

**현 상태: 완료.** 왕복 테스트 통과(등록 30/30, FLE·APE 동봉 매칭 이상) — [worklog/2026-08-06.md](worklog/2026-08-06.md).

### 1-2. PTZ-IBA — 로컬 좌표계 (`--no_georef`)

측량 데이터 확보 전에 **알고리즘이 우리 이미지에서 도는지**부터 확인한다.

```bash
.venv/bin/python scripts/run_ptz_ba.py \
    -i data/infiniq/offline -f data/infiniq/offline_matches \
    -o output-infiniq-offline --no_georef 2>&1 | tee /tmp/ptz_ba.log
```

- `--no_georef`는 지오참조를 건너뛰고 **`K, R`만** 로컬 좌표계로 쓴다 (`t = 0`, `dist = 0`)
- `--dist`는 이 경로에서 **무영향**이다 — `PTZRayFactor`가 왜곡을 적용하지 않기 때문 (0-1 제약)
- 이 출력이 2-3 온라인 재측위의 레퍼런스가 될 수 있다 (월드 좌표가 필요 없다면)

### 1-3. 지오참조 — 월드 좌표계 (`-a`)

**절차**

1. 로컬 미터 좌표계(ENU 등)를 정의하고 **원점·축 방향·단위·취득 수단·측량 1σ를 문서로 고정**한다.
   최종 2D-3D 오차를 해석할 때 측량 오차와 알고리즘 오차를 구분해야 하므로 **1σ 기록은 필수**다
2. 측량점 **20~30점**을 어노테이션 프레임 **2~3장 이상**에 분산해 클릭한다
   - 어노테이션 프레임은 **offline 레퍼런스 세트에 포함되고 등록에 성공한 프레임**이어야 한다
   - **프레임 3장 이상**을 권장한다. 2장이면 단일 강체 `R_l_w`/`t_l_w`가 fy 자유도를 흡수해
     **fy가 붕괴한다**(0-1 제약의 `Reproj2d3dFactor` 항목). 실제로 2 → 3장으로 늘린 것만으로
     `fy/fx`가 0.42 → 0.96~1.06으로 회복됐다
   - 점이 이미지 전체에 골고루 퍼질 것. 높이(Z) 변화가 있는 점을 섞으면 안정적이지만,
     단일 평면만으로도 동작한다 (WorldCup14 = 축구장 평면)
3. **홀드아웃 분할** — 측량점을 지오참조용(`offline/offline.json`)과 평가용(`gt/*.json`)으로
   **미리, 겹치지 않게** 나눈다. 권장 70:30. 이것이 1-4 절대 정확도 지표의 전제다
4. 실행 (`-a` 를 주면 IBA 직후 지오참조 BA가 1회 더 돈다)
   ```bash
   .venv/bin/python scripts/run_ptz_ba.py \
       -i data/infiniq/offline -f data/infiniq/offline_matches \
       -a data/infiniq/offline/offline.json \
       -o output-infiniq-offline-georef 2>&1 | tee /tmp/ptz_ba.log

   # 왜곡 추정 버전 — 채택 여부는 1-4의 홀드아웃으로 판정한다
   .venv/bin/python scripts/run_ptz_ba.py ... --dist -o output-infiniq-offline-georef-dist
   ```

**주의** — `_set_init_trans_local_to_world`는 **어노테이션 프레임 중 첫 성공 프레임 하나로만**
초기화하고 즉시 return한다. 나머지 프레임은 초기값이 아니라 이후 BA에서만 반영된다.

### 1-4. 검증

세 층으로 본다. **(a)가 하나라도 어긋나면 (b)의 수치는 볼 의미가 없다** (맞아 보여도 우연이다).
지표 정의·실행 명령·일괄 확인 스니펫은 [EVALUATION.md](EVALUATION.md) §1–3, 판정 체크리스트는 §5.

#### (a) 실행 중 판정 — 이론적으로 맞게 진행됐는가

| 확인 | 합격 기준 |
| --- | --- |
| `PTZ-IBA End: success` | 초기 페어 등록 + incremental 루프 완료. `failed`면 초기 페어를 50회 시도해도 등록 불가 |
| `Registered/Total` | 목표 **90% 이상**. `success`는 전 프레임 등록을 뜻하지 않는다 |
| `unregistered` 목록 | **설명 가능**할 것 (겹침 없는 프레임 등). 무작위로 빠지면 매칭을 의심 |
| GBA `final reprojection error all` | 발산하지 않을 것 (유한값, 수 px 수준) |
| 지오참조 초기화 게이트 | 전방(`z>0`) · `det(R) > 0` · EPNP 초기 재투영 < 300 px |
| `Error 2d-2d` / `Error 2d-3d` | `--no_georef`면 `-1`. 지오참조를 했으면 `-1`이 아닐 것 (실제로 돌았다는 증거) |

**출력 파라미터의 불변식** — 여기서 어긋나면 모델이 깨진 것이다.

| 확인 | 합격 기준 |
| --- | --- |
| JSON 카메라 수 | `== Registered` |
| 모든 `R` | `det(R) = 1`, `R Rᵀ = I` (1e-9 이내), 모든 `f > 0` |
| **모든 카메라의 `pos`가 동일** | PTZ는 순수 회전 모델이라 광학 중심이 공유돼야 한다 |
| `focal_vs_zoom.monotonic_increasing` | `true` — 줌 단계가 오르면 focal도 올라야 한다 |
| FOV | (0°, 170°) 안. 광각 프레임 FOV가 카메라 스펙과 대체로 맞을 것 |
| 실행 옵션 ↔ 저장값 | `--no_georef` → `t = 0`·`dist = 0`·holdout `skipped` / 지오참조 → `t ≠ 0` / `--dist` → `dist[0] ≠ 0`이고 나머지 4개는 정확히 0 (코드는 `k1`만 자유) |

> `distType` 필드는 판정에 쓰지 않는다. `data_io.py`가 `dist[0] < 1e-5`로 **부호 없이** 비교해
> 배럴 왜곡(`k1 < 0`)이면 `distType`이 `""`로 남는다. **`dist[0]`을 직접 본다.**

#### (b) 사후 정량 — 결과가 쓸 만한 정확도인가 (`scripts/eval_own.py --mode offline`)

```bash
.venv/bin/python scripts/eval_own.py --mode offline \
    -i data/infiniq/offline -f data/infiniq/offline_matches \
    --params output-infiniq-offline-georef-dist/offline.json \
    --ptz_log data/infiniq/ptz_log/offline.csv \
    --holdout data/infiniq/gt/offline.json \
    -o output-infiniq-offline-georef-dist/eval_report.json
```

| 지표 | 성격 | 합격 기준 |
| --- | --- | --- |
| `holdout_2d3d.errors_px` / `errors_m` | **절대 정확도 — 주 지표.** 지오참조에 쓰지 않은 측량점을 저장된 `K,R,t,dist`로 투영 | **측량 1σ의 픽셀 환산치 수준**까지 내려갈 것. 프레임별 `per_frame` 편차가 크지 않을 것 |
| `holdout` vs Summary `Error 2d-3d` | 과적합 검사 | **같은 자릿수**일 것. 홀드아웃만 크면 지오참조가 fit 점에 과적합 |
| `cross_frame_consistency_px` | 프레임 간 **상대** 일관성 (측량 불필요) | mean/median이 낮고, `worst_pairs`가 전체 평균에서 크게 튀지 않을 것 |
| `ptz_rotation_check.abs_diff_deg` | PTZ 리드아웃 대비 상대 회전 — **quasi-GT** | mean **1° 미만** |
| `ptz_rotation_check.best_fit_scale` | 리드아웃 대 focal 스케일 정합 | **1에 가까울 것.** 벗어나면 각도 단위 스케일 또는 전역 focal 편향 |
| `residual_after_scale_deg` | 스케일 제거 후 잔차 | 스케일 보정 전보다 뚜렷이 작을 것 (= 잔차가 스케일 성분이고 회전 자체는 정상) |
| `focal_vs_zoom.f_over_zoom_spread_pct` | 줌 배율 대비 focal 선형성 | 작을 것 |

**평가 자체가 유효했는지도 확인한다** — `pairs_evaluated > 0`, holdout의
`skipped_frames == []` · `n_behind_camera == 0` · `n_no_plane_intersect == 0`.

**`--dist` 채택 판정은 홀드아웃 하나로만 한다.** `cross_frame_consistency`는 `transfer_errors()`가
`PTZRayFactor`와 같은 모델을 쓰려고 **왜곡을 일부러 무시**하기 때문에 `--dist` 결과가 오히려 나빠
보인다 — 단독으로는 회귀 신호가 아니다. 홀드아웃이 개선되지 않으면 `--dist`를 쓸 근거가 없다.

#### (c) 정성 — 파노라마 스티칭

```bash
.venv/bin/python scripts/render_panorama.py \
    -i data/infiniq/offline --params output-infiniq-offline/offline.json \
    -o output-infiniq-offline/panorama.png
```

바닥 타일선·걸레받이·벽 모서리 같은 **직선 구조가 프레임 경계를 넘어 끊김 없이 이어지면** 통과.
화면 고정 OSD의 고스팅은 장면에 붙은 요소가 아니므로 정합 결함이 아니다.

#### (d) 실패 신호

- `PTZ-IBA End: failed` / 등록률이 현저히 낮음 / GBA 재투영이 수십~수백 px
- 카메라 `pos`가 프레임마다 다름 — **순수 회전 모델이 깨졌다**
- `focal_vs_zoom.monotonic_increasing == false` — 줌과 focal이 역전
- 홀드아웃 2D-3D가 fit 점의 `Error 2d-3d`보다 훨씬 큼 — 과적합
- `best_fit_scale`이 1에서 크게 벗어남

**현 상태: 완료.** (a) 필수 항목 전원 통과, (b) 지표도 `--dist` 설정에서 전부 양호,
(c) 통과, (d) 실패 신호 0건. **채택 설정은 지오참조 + `--dist`**이고, 절대 정확도는 측량 노이즈
바닥의 1.5~2배까지 와 있다. 실측값과 판정 근거는 [RESULTS-own.md](RESULTS-own.md) §2–3·§5.
같은 입력·기본 `--seed`로 재실행 시 출력 JSON이 **md5 단위로 동일**함도 확인했다.

**남은 잔차의 원인 — 렌즈 왜곡으로 판정 (해결).**

한때 "측량 좌표계의 전단"을 주 원인으로 봤으나 **철회한다.** 근거는 두 가지다.

- **왜곡을 모델에 넣으면 전단 요구가 사라진다.** 월드 좌표에 보정을 넣고 `R_l_w`/`t_l_w`만
  재적합하면, 왜곡 없는 지오참조에서는 전단 1개(직각에서 3.54°)가 적합에 쓰지 않은 홀드아웃을
  2.8배 개선하지만, `--dist` 파라미터에서는 **같은 전단 요구가 0.72°로 붕괴**하고 강체만으로도
  홀드아웃이 이미 측량 노이즈 바닥이다 → **`k1`이 같은 계통 성분을 설명한다**
- **전단 가설은 물리적으로 성립하지 않는다.** 3.54°는 측량 범위(Y 8.46 m)에 걸쳐
  **0.52 m = 타일 1.17칸**의 어긋남이다. 그런데 측량점 43개가 전부 타일 간격 0.445 m의
  **정확한 정수배**(잔차 0.000000칸)다. 즉 전단이 실재하려면 타일 격자 자체가 방 한쪽 끝에서
  한 칸 넘게 밀린 평행사변형이어야 하는데, 그 정도면 육안으로 드러나고 칸 세기가 일관될 수도 없다.
  `--dist` 후 남는 0.72°는 0.107 m = **0.24칸**으로 타일 이하이며 코너 지정 오차 범위다

→ **측량 좌표는 정상이고, 잔차의 계통 성분은 렌즈 왜곡이다.** 따라서 줄자 재실측은 불필요하고,
`--dist` 채택은 자유도 증가가 아니라 **누락된 물리 모델을 넣은 것**으로 본다.

**미결** — 상세 분석은 [RESULTS-own.md](RESULTS-own.md) §6, 판별자 정리는 §7.

- **`k1`의 절대값은 아직 검증되지 않았다.** 원인이 왜곡이라는 것과 추정된 `k1`이 옳다는 것은 다르다.
  `k1`은 프레임별로 독립 추정되는데 줌 단계 **내부 산포가 단계 간 차이와 맞먹고** 광각에서는
  부호까지 뒤집힌다 — 물리 왜곡이라면 같은 줌에서 같은 값이어야 한다. 남은 잔차를 프레임별로
  흡수하고 있을 여지가 있다. **광각 체커보드로 `k1`을 독립 측정**하면 닫힌다
- **f 절대값과 각도 스케일이 분리되지 않는다.** 세 설정에서 `f 비율 × best_fit_scale`이
  **0.3% 이내로 일정**하다 — 순수 회전 문제에서 f와 각도는 같은 픽셀 이동을 만들므로 세 해가
  **하나의 축퇴 능선 위 서로 다른 점**이라는 뜻이다. 회전 데이터만으로는 "카메라 각도 단위가 7%
  틀렸다"와 "f를 7% 과소추정했다"를 가를 수 없다.
  - 2D-3D 측량점도 이를 가르지 못한다 — 전역 focal 스케일 자유도를 줘도 홀드아웃이 거의 개선되지
    않는다(f×1.008). 측량점은 **사용된 반경에서의 유효 배율**만 구속하고, paraxial f와 `k1`은
    서로 교환되기 때문이다
  - **비용 0의 판별자가 하나 있다**: `--dist` 해의 광각 hFOV가 "로컬 해의 f가 7.36% 과소추정된
    경우의 참값"과 일치한다. **카메라 스펙시트의 최광각 수평 FOV**를 보면 갈린다 — 모델명 확인 필요
- **체커보드 독립 검증 (미실시)** — 최광각 고정 줌에서 `f`, `k1`의 독립 측정값을 얻는다.
  망원에서는 사실상 불가하므로 **광각 1점 검증**에 한정한다. **위 두 미결을 동시에 닫는 단일 조치**다
- **어노테이션 점 분포** — fit 점이 전부 화면 하단(바닥 타일 코너)이고 Z = 0 단일 평면이다.
  화면 상단·높이 변화 점을 추가하면 프레임별 편차(현재 약 4배)가 줄고, 등방 3D 스케일 축퇴
  (`t`가 자유로워 `(s,t) → (ks,kt)`가 투영을 보존 — 현재 데이터로는 판정 불가)도 풀린다
- **홀드아웃 비율·규모** — 현재 fit : holdout ≈ 50:50(권장 70:30 미달)이고 홀드아웃 관측이 한 자릿수라
  지표 자체의 분산이 크다. 측량점 총량을 늘리는 것이 우선이다
- **재투영 오차 목표(1~2 px) 미달** — 로컬 좌표계 GBA 재투영이 목표를 넘는다. 아웃라이어 잔존
  신호는 없다. 근거리 실내 장면의 회전 중심 오프셋이 후보이나, **레버암 가설은 철회했다** —
  순수 회전 가정의 2D-2D BA가 수 px로 맞는다는 사실이 유효 베이스라인을 1 cm 이하로 묶기 때문이다.
  가장 싼 검증은 **원거리 장면 재촬영**이다

---

## 2. Online Stage

임의 자세로 찍은 테스트 프레임을, 레퍼런스 파라미터를 기준으로 **한 장씩 재측위**한다.

### 2-1. 테스트 프레임 요건

- **50~100장.** 레퍼런스와 **다른 pan/tilt/zoom** 조합 (프리셋 사이 값, 임의 자세).
  레퍼런스 격자의 **절반 스텝** 사이값이 좋은 기본 설계다
- 각 테스트 프레임이 **레퍼런스 최소 1장과 40쌍 이상 매칭**될 것 → 레퍼런스 커버리지 범위 안에서 촬영
- **의도적으로 어려운 케이스를 포함**하고, CSV에 **난이도 라벨 열**을 넣는다 (2-4에서 분리 집계)

  | 군 | 내용 |
  | --- | --- |
  | normal | 정상 커버리지 |
  | boundary | 레퍼런스 팬/틸트 격자의 경계 및 그 밖 |
  | zoom_extrap | 레퍼런스 최대 줌보다 더 망원 (외삽) |
  | mixed | 위 조합 |

- 레퍼런스와 **다른 시간대/조명**을 섞는다
- 테스트 프레임도 pan/tilt/zoom을 기록한다 (`ptz_log/online.csv`) — 난이도 열을 여기에 둔다

### 2-2. 매칭 생성

```bash
.venv-match/bin/python scripts/extract_match.py --mode online \
    --ref_images data/infiniq/offline --ref_features data/infiniq/offline_matches \
    --test_images data/infiniq/online -o data/infiniq/online_matches \
    --ptz_log data/infiniq/ptz_log/online.csv
```

- **레퍼런스 특징점은 재추출하지 않고 `offline_matches`에서 읽는다** — 인덱스 불일치 방지
- `--topk 3` — 테스트 프레임당 상위 3개 레퍼런스를 기록한다. 공개 데이터셋은 k=1이지만,
  자체 데이터는 커버리지가 불확실하므로 여유를 두고 `find_best_match`가 최대 매칭을 고르게 한다
- `--prefilter 8` + `--ptz_log` — 레퍼런스 후보를 **PTZ 리드아웃 각도 차이**로 사전 축소한다
  (전역 검색 불필요). 로그가 없으면 생략 가능
- 출력 블록 방향은 반드시 `(레퍼런스, 테스트)` — 0-2 (4)

### 2-3. 재측위 (`scripts/run_ptz_reloc.py`)

**함정 먼저** — `read_cam_from_json`은 **레퍼런스 폴더의 모든 이미지가 파라미터 JSON에 있을 것**을
요구한다. 미등록 프레임이 남아 있으면 전체가 중단된다. 등록된 프레임만 담은 디렉터리를 만들어 넘긴다.

```bash
REG=data/infiniq/offline_registered; rm -rf $REG; mkdir -p $REG
# offline.json 에 있는 이미지 + mask.png 만 심볼릭 링크

.venv/bin/python scripts/run_ptz_reloc.py \
    --ref_images $REG --ref_features data/infiniq/offline_matches \
    --ref_params output-infiniq-offline-georef-dist/offline.json \
    --test_images data/infiniq/online --test_features data/infiniq/online_matches \
    --output output-infiniq-online-georef-dist --dist
```

- `--ref_params`에 **로컬 좌표계 결과(`--no_georef`)를 주면 결과도 로컬**, 월드 좌표계 결과를 주면
  월드다. 홀드아웃 2D-3D로 평가하려면 **월드 파라미터**를 넣어야 한다
- `--dist`는 레퍼런스를 만든 설정과 **맞춰서** 준다

### 2-4. 검증

#### (a) 성공률 · 처리 시간 · 실패 분류

| 확인 | 합격 기준 |
| --- | --- |
| 재측위 성공률 | 기록한다. **매칭이 성립한 프레임 기준으로는 100%에 가까울 것** |
| 이미지당 처리 시간 | 기록한다 (참고: 공개 데이터셋 0.04–0.13 s/장) |
| 실패 프레임 분류 | **매칭 부족 / 미수렴 / 재투영 > 100px / FOV 범위 이탈** 로 나눈다. 매칭 부족 외의 실패는 알고리즘 실패이므로 원인을 규명한다 |
| `unregistered` 설명 가능성 | 실패가 **촬영 품질**(무텍스처·커버리지 이탈)로 설명될 것 |

#### (b) 난이도군별 분리 집계

`eval_own.py --mode online --ptz_log ...` 가 CSV의 난이도 열로 자동 집계한다
(`per_difficulty`: 군별 프레임 수·성공·성공률·일관성).

**합격 기준: 난이도가 정확도를 떨어뜨리지 않을 것.** 성공률 차이가 있다면 그것이 난이도 자체가 아니라
**촬영 품질 차이**로 설명되는지 확인한다. 망원(zoom_extrap)은 같은 각오차가 더 작은 픽셀오차로
나타나므로 일관성 지표가 오히려 좋게 나오는 것이 정상이다.

#### (c) 정량 지표

```bash
.venv/bin/python scripts/eval_own.py --mode online \
    --ref_images data/infiniq/offline_registered --ref_features data/infiniq/offline_matches \
    --ref_params output-infiniq-offline-georef-dist/offline.json \
    --test_images data/infiniq/online --test_features data/infiniq/online_matches \
    --test_params output-infiniq-online-georef-dist/online.json \
    --ptz_log data/infiniq/ptz_log/online.csv \
    --holdout data/infiniq/gt/online.json \
    -o output-infiniq-online-georef-dist/eval_report.json
```

지표 정의와 판정 기준은 **1-4 (b)와 동일**하다. online에서 `--holdout`은 `--test_params`에 적용된다.
추가로 볼 것:

- **홀드아웃 프레임이 전부 재측위에 성공했는지** (`skipped_frames == []`) — 실패 프레임의 점은
  집계에서 빠지므로, 남은 관측이 몇 개인지 항상 같이 본다
- **offline 홀드아웃과 같은 자릿수인지** — online이 크게 나쁘면 재측위 단계 문제이고,
  비슷하면 두 단계가 공유하는 원인(측량 좌표계 등)이다

#### (d) 실패 신호

- 매칭이 성립했는데 실패하는 프레임이 있다 (미수렴 / 재투영 > 100px / FOV 이탈)
- 특정 난이도군만 성공률이 무너진다 — 촬영 품질로 설명되지 않으면 알고리즘 한계다
- 레퍼런스 **오선택**에서 기인한 실패 — 관측되면 frustum overlap 기반 재선택을 검토한다
  (공개 코드에는 없는 기능이다 — 0-1 "코드를 따른 지점")
- 홀드아웃이 offline 대비 크게 나쁘다

**현 상태: 완료.** 월드 좌표계 파라미터로 재실행해 online 홀드아웃까지 산출했고,
(b) 난이도군별 집계·(c) 지표 모두 확보했다. **홀드아웃이 offline과 같은 자릿수**여서 재측위 단계가
추가 오차를 거의 넣지 않음을 확인했고, 매칭이 성립한 프레임은 **전부** 성공했다.
실측값은 [RESULTS-own.md](RESULTS-own.md) §4.

**미결**

- **`eval_own.py`의 online 실패 사유 분류가 자동화돼 있지 않다** — 현재 리포트는 `failed_frames`
  목록만 준다. (a)의 4분류는 `pairs_matches.txt`의 블록 수를 직접 세어 판정하고 있다
  ([RESULTS-own.md](RESULTS-own.md) §8에 확인 명령). 리포트가 이 판정을 채우면 닫힌다
- **옥외·조명 변화 미검증** — 시간차(레퍼런스 대비 +23시간) 조건에서 열화는 없었으나 실내
  인공조명이었다. 옥외에서는 재확인이 필요하고, 현재 라벨은 시간대를 나누지 않아 시간차는
  전체 단위 판정에 그친다
- **홀드아웃 관측 수 부족** — online 홀드아웃 11관측 중 **한 프레임이 rmse를 지배한다**.
  1-4의 미결(`k1`·f 절대값 검증, 점 분포)을 그대로 물려받으므로 offline 측량을 보강하면 함께 해소된다
