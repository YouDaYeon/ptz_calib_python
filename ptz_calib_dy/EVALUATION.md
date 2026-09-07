# PTZ-IBA 평가 가이드

PTZ-IBA(offline stage) 결과가 **제대로 나왔는지** 확인하는 방법을 정리한다.
평가는 **실행 중 로그 판정**, **사후 정량 지표**, **정성 확인** 세 층으로 나뉜다.

> GT 카메라 파라미터가 없는 자체 데이터(infiniq 등)는 논문의 FLE/APE/IoU를 **쓸 수 없다** (§4).
> 이 경우 `scripts/eval_own.py`의 **교차 프레임 일관성**과, georef를 했다면
> **홀드아웃 2D–3D 오차**가 주 정량 지표다 (§2).
> 자세한 실측 수치 예시는 `RESULTS-own.md`를 참고한다.
>
> 과제 성능 지표(**다중 카메라 위치 정합 오차 ≤ 0.8 m**) 판정은 리포트의
> `calibration_quality`가 자동으로 낸다 (§2-6). 플래그를 주지 않아도 항상 나온다.

---

## 1. 실행 중 판정 (`run_ptz_ba.py`)

### 1-1. 파이프라인 성공 / 실패

`run_ptz_ba.py`는 IBA가 끝나면 로그에 아래 중 하나를 남긴다.


| 로그                     | 의미                              | exit code         |
| ---------------------- | ------------------------------- | ----------------- |
| `PTZ-IBA End: success` | 초기 페어 등록 성공 + incremental 루프 완료 | 0 (georef까지 성공 시) |
| `PTZ-IBA End: failed`  | 초기 페어를 50회 시도해도 등록 불가           | -1                |


**주의**: `PTZ-IBA End: success`는 **전체 프레임이 다 등록됐다**는 뜻이 아니다.
일부 프레임만 등록돼도 success일 수 있다. 등록 수는 `Registered/Total`로 본다.

또한 `PtzIncrementalOptimizer.solve()`는 **마지막 global BA가 실패해도** incremental이 끝나면
`True`를 반환한다. 품질은 아래 재투영 오차와 사후 평가로 확인한다.

### 1-2. 로그에서 볼 지표


| 로그 문자열                                                       | 지표                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| `Currently registered: N, total: M`                          | 등록 진행 상황                                                |
| `Global bundle adjustment success. Rreprojecion error: X.XX` | IBA 내부 **2D–2D ray 재투영 오차 (px)**                        |
| `Registered/Total: N/M`                                      | 최종 등록률                                                  |
| `Error 2d-2d` / `Error 2d-3d`                                | georeferencing 후 2D–2D / 2D–3D 오차 (`--no_georef`면 `-1`) |




### 1-3. 실행 예시 (infiniq)

annotation이 있으면 IBA 직후 georeferencing이 **1회** 더 돌아간다
(`PTZRayOptimizer`에서 2D–2D + 2D–3D joint BA).

```bash
cd ptz_calib_dy

# IBA만 (georef 생략) — annotation 없을 때
.venv/bin/python scripts/run_ptz_ba.py \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    -o output-infiniq-offline \
    --no_georef \
    2>&1 | tee /tmp/ptz_ba.log

# IBA + georef — annotation 있을 때
.venv/bin/python scripts/run_ptz_ba.py \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    -a data/infiniq/offline/offline.json \
    -o output-infiniq-offline-georef \
    2>&1 | tee /tmp/ptz_ba.log

# IBA + georef + 왜곡 추정
.venv/bin/python scripts/run_ptz_ba.py \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    -a data/infiniq/offline/offline.json \
    -o output-infiniq-offline-georef-dist \
    --dist \
    2>&1 | tee /tmp/ptz_ba.log

# 핵심 로그만 추출 (georef 시 Error 2d-2d / Error 2d-3d 도 포함)
grep -E "Currently registered:|final reprojection error all|IBA End|Registered/Total|Error 2d-2d|Error 2d-3d" /tmp/ptz_ba.log
```

기대 예시 (worklog 2026-08-06): `Registered/Total: 20/22`, GBA 재투영 `~2.9 px`.

Summary의 `Error 2d-2d`, `Error 2d-3d`가 georef 품질 지표다.
IBA만 평가할 때는 `--no_georef`를 쓰고, GBA 재투영 오차 + §2를 본다.

### 1-4. 단계별 내부 성공 조건 (참고)


| 단계                | 옵티마이저             | 성공 조건                             |
| ----------------- | ----------------- | --------------------------------- |
| 초기 페어 / global BA | `PTZRayOptimizer` | LM `CONVERGENCE`                  |
| 다음 이미지 등록         | `KRTOptimizer`    | 수렴 + 재투영 < 100 px + FOV 유효        |
| GBA 실패 시          | —                 | 방금 등록한 이미지를 `reg_image_ids_`에서 제거 |


---



## 2. 사후 정량 평가 — GT 없음 (`eval_own.py`)

`scripts/eval_own.py`는 **offline(PTZ-IBA)** / **online(PTZ-Reloc)** 모드를 지원한다.
여기서는 offline(IBA)만 다룬다.

평가는 `run_ptz_ba.py`를 어떤 옵션으로 돌렸는지에 따라 **볼 수 있는 지표가 달라진다**.


| 실행 설정             | 저장된 파라미터                     | 사후 평가로 볼 수 있는 것                      |
| ----------------- | ---------------------------- | ------------------------------------ |
| `--no_georef`     | `K, R` (`t = 0`, `dist = 0`) | cross-frame consistency, PTZ 로그 대조만  |
| georef (`-a`)     | `K, R, t` (월드 좌표계)           | 위 + **holdout 2D–3D 절대 오차 (px / m)** |
| georef + `--dist` | `K, R, t, dist`              | 위 + 왜곡 보정 효과 (holdout px/m 로 판정)     |




### 2-1. 지표


| 지표                           | 설명                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `registration_rate`          | JSON에 저장된 카메라 수 / 전체 이미지 수                                                                     |
| `unregistered`               | 등록 실패 프레임 이름 목록                                                                                |
| `cross_frame_consistency_px` | 매칭 키포인트를 infinite homography `K_j R_j R_i^T K_i^{-1}`로 i→j 전달했을 때 픽셀 오차 (mean/median/p90/rmse) |
| `holdout_2d3d.errors_px`     | georef에 **쓰지 않은** 측량점을 저장된 `K,R,t,dist`로 투영한 픽셀 오차 (`--holdout` 필요)                            |
| `holdout_2d3d.errors_m`      | 같은 홀드아웃 점을 클릭 광선–수평면 교차로 역산한 **미터 단위** 오차                                                      |
| `ptz_rotation_check`         | PTZ pan/tilt 로그와 추정 R 비교 (`--ptz_log` 필요, quasi-GT)                                            |
| `focal_vs_zoom`              | zoom 단계별 focal 일관성 (`--ptz_log` 필요)                                                            |
| `calibration_quality`        | 과제 지표(0.8 m) 판정 등급 — `valid` / `degraded` / `invalid` / `unknown` (§2-6)                       |
| `verdict`                    | 판정 상세 — 임계값, 판정에 쓴 rmse, 이유, 그리고 **등급에 영향을 주지 않는** 병기값 (§2-6)                                 |


cross-frame consistency는 **프레임 간 상대 일관성**만 측정한다.
모든 프레임에 공통으로 틀린 focal bias는 잘 잡히지 않는다.
또한 `transfer_errors()`는 `PTZRayFactor`와 동일한 모델을 재현하려고 **왜곡을 일부러 무시**한다.
그래서 `--dist`로 얻은 파라미터는 이 지표에서 오히려 나빠 보일 수 있다 (§2-4).
`--dist` 효과 판정은 cross-frame이 아니라 holdout 2D–3D로 한다.

`holdout_2d3d`는 절대 정확도 지표이며 **georef된 파라미터에서만** 의미가 있다.
`--no_georef` 결과(`t = 0`)를 주면 `status: skipped`로 빠진다.

`calibration_quality`는 그 `errors_m.rmse` **하나만으로** 결정된다.
등록률이나 관측 수는 함께 기록되지만 등급을 바꾸지 않는다 (§2-6).

### 2-2. PTZ 로그 / 측량 데이터 경로

PTZ pan/tilt/zoom CSV는 디렉터리별로 나뉜다.

```
data/infiniq/ptz_log/
├── offline.csv   # offline(IBA) 22장
└── online.csv    # online(reloc) 70장
```

`--ptz_log`에는 **모드에 맞는 CSV 파일**을 넘긴다 (디렉터리가 아님).

측량 2D–3D annotation은 georef용과 홀드아웃용이 **프레임 단위로 분리**돼 있다.

```
data/infiniq/offline/offline.json   # georef 입력 (-a):    ptz_frame_001 / 007 / 012
data/infiniq/gt/offline.json        # 홀드아웃 (--holdout): ptz_frame_002 / 006 / 008
```

두 집합은 겹치지 않으므로 `--holdout` 결과는 in-sample fit이 아니다.
반대로 `run_ptz_ba.py` Summary의 `Error 2d-3d`는 georef에 **쓴** 점 기준이라 과적합될 수 있다.
절대 정확도 판정은 `Error 2d-3d`가 아니라 `holdout_2d3d`로 한다.

### 2-3. 실행 예시 (offline / IBA)

**(a)** `--no_georef` **결과 — cross-frame + PTZ 로그만**

```bash
cd ptz_calib_dy

.venv/bin/python scripts/eval_own.py --mode offline \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    --params output/output-infiniq-offline-georef-dist/offline.json \
    --ptz_log data/infiniq/ptz_log/offline.csv \
    -o output/output-infiniq-offline-georef-dist/eval_report.json
```

PTZ 로그가 없으면 `--ptz_log`를 빼면 된다.

**(b) georef 결과 —** `--holdout` **추가**

```bash
.venv/bin/python scripts/eval_own.py --mode offline \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    --params output/output-infiniq-offline-georef-dist/offline.json \
    --ptz_log data/infiniq/ptz_log/offline.csv \
    --holdout data/infiniq/gt/offline.json \
    -o output/output-infiniq-offline-georef-dist/eval_report.json
```

`--params`만 georef 출력으로 바꾸고 `--holdout`을 붙이면 된다.
`-i` / `-f`는 (a)와 동일하다 — cross-frame은 같은 매칭으로 계산돼야 비교가 된다.

**(c) georef +** `--dist` **결과 — (b)와 동일,** `--params`**만 교체**

```bash
.venv/bin/python scripts/eval_own.py --mode offline \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    --params output-infiniq-offline-georef-dist/offline.json \
    --ptz_log data/infiniq/ptz_log/offline.csv \
    --holdout data/infiniq/gt/offline.json \
    -o output-infiniq-offline-georef-dist/eval_report.json
```

세 설정을 §1-3의 명령으로 각각 실행해 두면, 위 세 평가를 그대로 돌려 비교할 수 있다.

### 2-4. 세 설정 비교 / 해석

```bash
python -c "
import json
runs = [
    ('no_georef  ', 'output-infiniq-offline/eval_report.json'),
    ('georef     ', 'output-infiniq-offline-georef/eval_report.json'),
    ('georef+dist', 'output-infiniq-offline-georef-dist/eval_report.json'),
]
for label, path in runs:
    try:
        r = json.load(open(path))
    except FileNotFoundError:
        print(f'{label}: (없음)'); continue
    c = r['cross_frame_consistency_px']
    line = f\"{label}: reg {r['registered']}/{r['total_images']}  cross-frame mean={c['mean']:.2f} p90={c['p90']:.2f}\"
    h = r.get('holdout_2d3d', {})
    e, m = h.get('errors_px') or {}, h.get('errors_m') or {}
    if h.get('status') == 'ok' and e.get('count'):
        line += f\"  holdout {e['rmse']:.2f} px\"
        if m.get('count'):
            line += f\" / {100 * m['rmse']:.1f} cm (rmse, n={e['count']})\"
    else:
        line += f\"  holdout {h.get('status', '-')}\"
    line += f\"  [{r.get('calibration_quality', '-')}]\"
    print(line)
"
```

리포트가 저장된 실제 경로는 `output/output-infiniq-offline{,-georef,-georef-dist}/eval_report.json`이다
(§1-3 예시의 `-o`를 그대로 쓰면 현재 디렉터리에 생기므로, 위 스니펫의 경로는 실행 위치에 맞춰 고친다).

**실측 예시 (infiniq offline 22장, 홀드아웃 3프레임 × 3점 = 9 obs)**


| 설정                    | 등록    | cross-frame mean / p90 (px) | holdout rmse (px)   | holdout rmse (cm) |
| --------------------- | ----- | --------------------------- | ------------------- | ----------------- |
| (a) `--no_georef`     | 20/22 | 1.54 / 2.88                 | `skipped` (`t = 0`) | —                 |
| (b) georef            | 20/22 | 1.54 / 2.88                 | 39.5                | 13.2              |
| (c) georef + `--dist` | 20/22 | 3.31 / 7.37                 | **15.9**            | **7.8**           |


판정 기준:


| 비교                             | 기대                             | 어긋나면                                                        |
| ------------------------------ | ------------------------------ | ----------------------------------------------------------- |
| (a) vs (b) `registration_rate` | 동일                             | georef는 등록에 관여하지 않는다. 달라지면 실행 조건이 다른 것                      |
| (a) vs (b) cross-frame         | 거의 동일                          | georef BA가 IBA 해의 `K,R`을 망가뜨렸다는 신호 (실측: 1.54 → 1.54, 변화 없음) |
| (b) `holdout_2d3d`             | px rmse가 `Error 2d-3d`와 같은 자릿수 | 홀드아웃만 크면 georef가 3점에 과적합                                    |
| (b) vs (c) `holdout_2d3d`      | (c)가 같거나 더 낮다                  | (c)가 나빠지면 왜곡 파라미터 과적합 — `--dist` 없이 간다                      |
| (b) vs (c) cross-frame         | (c)가 다소 커질 수 있다                | 아래 주석 참고 — 단독으로는 회귀 신호가 아니다                                 |


(c)에서 cross-frame이 커지는 것(1.54 → 3.31)은 **측정 방식의 부작용**이다.
`transfer_errors()`는 왜곡을 무시하는데 (c)의 `K,R`은 `dist ≠ 0`을 전제로 맞춰진 값이라,
왜곡 없는 모델로 재보면 불일치가 그대로 잔차로 나온다.
따라서 (b) vs (c) 판정은 **holdout 2D–3D 하나로** 한다.
infiniq 실측에서는 (c)가 39.5 → 15.9 px (13.2 → 7.8 cm)로 절반 이하이므로 `--dist`**가 유효**하다.

반대로 `--dist`는 측량점 수가 적을 때 쉽게 과적합된다.
(c)가 (b)보다 홀드아웃에서 나아지지 않으면 `--dist`**를 쓸 근거가 없다**.

### 2-5. JSON 리포트만 다시 보기

```bash
python -c "
import json
r = json.load(open('output-infiniq-offline-georef/eval_report.json'))
print('registered:', r['registered'], '/', r['total_images'])
v = r.get('verdict') or {}
print('quality   :', r.get('calibration_quality'), '|', v.get('reason'))
c = r['cross_frame_consistency_px']
print(f\"cross-frame: mean={c['mean']:.2f} median={c['median']:.2f} p90={c['p90']:.2f} n={c['count']}\")
h = r.get('holdout_2d3d', {})
if h.get('status') == 'ok' and (h.get('errors_px') or {}).get('count'):
    e, m = h['errors_px'], h.get('errors_m') or {}
    print(f\"holdout px : rmse={e['rmse']:.2f} mean={e['mean']:.2f} p90={e['p90']:.2f} n={e['count']}\")
    if m.get('count'):
        print(f\"holdout m  : rmse={m['rmse']:.4f} mean={m['mean']:.4f}\")
    for fr in h['per_frame']:
        pe = fr['errors_px']
        if pe.get('count'):
            print(f\"  {fr['frame']}: n={pe['count']} rms={pe['rmse']:.2f} px\")
else:
    print('holdout:', h.get('status'), h.get('reason', ''))
if r.get('unregistered'):
    print('unregistered:', r['unregistered'])
"
```

`quality`가 `None`으로 나오면 그 리포트는 §2-6 판정이 추가되기 전에 만들어진 것이다.
§2-3 명령으로 다시 돌리면 채워진다.

online(reloc) 평가 시에는 `--mode online`과 `data/infiniq/ptz_log/online.csv`를 쓰고,
`--holdout`은 `--test_params`에 대해 적용된다.

### 2-6. 과제 지표 판정 (`calibration_quality`)

과제 성능 지표인 **다중 카메라 위치 정합 오차 ≤ 0.8 m**를 리포트가 스스로 판정한다.
판정 대상은 `holdout_2d3d.errors_m.rmse` **하나**다 — 미터 단위 절대 오차가 곧 지표이기 때문이다.

등급 정의는 `ptzcalib/quality.py`의 `CalibrationQuality`에 있다.
**여섯 상태 중 네 개만** `eval_own.py`가 산출한다.


| 등급         | 조건                    | 의미                             | `eval_own.py` |
| ---------- | --------------------- | ------------------------------ | ------------- |
| `valid`    | `rmse ≤ 0.8 m`        | 허용 기준 이내의 정상 calibration       | 산출            |
| `degraded` | `0.8 m < rmse ≤ 1.0 m`| 기준 초과, 제한적으로 사용 가능             | 산출            |
| `invalid`  | `rmse > 1.0 m`        | 오차가 커서 사용할 수 없음                | 산출            |
| `unknown`  | rmse를 구할 수 없음         | 판정에 필요한 정보가 없어 아직 판단 불가        | 산출            |
| `fallback` | —                     | 이전 또는 대체 calibration을 사용하는 상태  | **산출 안 함**    |
| `moving`   | —                     | PTZ 이동 등으로 현재 해를 신뢰할 수 없는 상태   | **산출 안 함**    |


두 임계값은 **상한 포함**이다. `rmse == 0.8`은 `valid`, `rmse == 1.0`은 `degraded`다.

`fallback` / `moving`은 "얼마나 정확한가"가 아니라 **"지금 어느 calibration을 쓰는가"** 를
말하는 **런타임 상태**다. 저장된 JSON을 채점하는 배치 평가에는 그 정보가 아예 없으므로
`classify()`는 이 둘을 절대 반환하지 않는다. 관측할 수 있는 곳은 live_web뿐이다
(`app.py::_view_drift`가 화면 이동을, 재캘리브레이션 실패가 대체 해 사용을 안다).
enum을 `ptzcalib/quality.py`에 둔 이유가 이것이다 — 상태 정의가 두 곳으로 갈리지 않게 한다.

#### rmse가 없으면 전부 `unknown`

`invalid`는 **실제로 측정된 rmse가 1.0 m를 넘을 때만** 부여한다.
아래 네 경우는 모두 `unknown`이다.


| 상황                                | `holdout_2d3d.status` |
| --------------------------------- | --------------------- |
| `--holdout` 미지정                   | `skipped`             |
| `--no_georef` 결과 (`t = 0`)        | `skipped`             |
| 홀드아웃 프레임이 등록 카메라와 하나도 겹치지 않음       | `error`               |
| georef인데 유효 관측 0개 (카메라 뒤 / 평면 미교차) | `ok`, `count = 0`     |


측정되지 않은 calibration은 **실패한 것이 아니라 판정되지 않은 것**이다.
어느 경우였는지는 `verdict.reason`에 원문이 그대로 들어간다.

#### 등급에 영향을 주지 않는 병기값

`verdict`에는 아래 값도 함께 기록되지만 **등급을 바꾸지 않는다.**
`calibration_quality`가 항상 정확히 한 가지만 뜻하게 하기 위해서다 —
표본이 적어서 좋게 나온 것과 정확도가 좋아서 좋게 나온 것을 숫자 하나로 뭉개지 않는다.


| 필드                                        | 값 (offline georef+dist) | 왜 병기하는가                          |
| ----------------------------------------- | ----------------------- | -------------------------------- |
| `n_obs` / `n_frames_measured`             | 9 / 3                   | 22프레임 중 3프레임만 절대 정확도가 측정됐다        |
| `registered` / `total` / `rate`           | 20 / 22 / 0.909         | 등록 실패 프레임은 오차 통계에 흔적을 남기지 않는다     |
| `n_behind_camera`, `n_no_plane_intersect` | 0, 0                    | 통계에서 조용히 빠진 관측 수                  |
| `skipped_frames`                          | `[]`                    | 미등록이라 채점되지 못한 홀드아웃 프레임            |


**`valid` 하나만 보고 판단하면 안 된다.** `n_obs 9 / 3프레임`으로 받은 `valid`와
`n_obs 30 / 10프레임`으로 받은 `valid`는 신뢰도가 다르다.
online의 `rate`(58/70)는 특히 중요하다 — 실패한 12프레임에는 측량점이 하나도 없어서
`errors_m`에 어떤 흔적도 남기지 않는다.

#### 프레임별 등급

`holdout_2d3d.per_frame[*].quality`에 프레임 단위 등급도 들어간다.
PTZ 자세 하나가 곧 가상 카메라 하나이므로, **다중 카메라 중 어느 카메라가 발목인지**를
여기서 본다. 전역 등급은 전 관측을 합친 rmse로 따로 계산되며 프레임별 최악값을 따르지 않는다.

#### 실행

```bash
cd ptz_calib_dy

.venv/bin/python scripts/eval_own.py --mode offline \
    -i data/infiniq/offline \
    -f data/infiniq/offline_matches \
    --params output/output-infiniq-offline-georef-dist/offline.json \
    --holdout data/infiniq/gt/offline.json \
    -o output/output-infiniq-offline-georef-dist/eval_report.json
```

`--pass_threshold_m`은 기본 `0.8`, `--degraded_threshold_m`은 기본 `1.0`이므로
**플래그를 주지 않아도 판정이 항상 나온다.** `--degraded_threshold_m`이
`--pass_threshold_m`보다 작으면 degraded 구간이 비어버리므로 실행 자체를 거부한다.

판정 결과는 **exit code에 반영하지 않는다** — 어느 등급이든 `0`이다.
`0`이 아닌 값은 인자 오류 같은 실행 실패를 뜻한다. 기존 배치 스크립트가
`invalid` 때문에 중단되지 않게 하기 위해서다.

출력 마지막 두 줄:

```
calib quality     : VALID  rmse 0.0779 m  (valid <= 0.8, degraded <= 1.0 m)
    not graded    : n_obs 9 over 3 frames  |  registered 20/22 (90.9%)  |  excluded 0 behind / 0 off-plane
```

프레임별 등급은 홀드아웃 상세 줄 **끝에 대괄호로** 붙는다.

```
    ptz_frame_002: n=3  rms 15.21 px  ...  |  rms 0.1045 m (10.5 cm)  ...  [valid]
    ptz_frame_006: n=3  rms  5.77 px  ...  |  rms 0.0295 m ( 3.0 cm)  ...  [valid]
    ptz_frame_008: n=3  rms 22.24 px  ...  |  rms 0.0800 m ( 8.0 cm)  ...  [valid]
```

#### 실측 판정 (2026-09-07)


| 설정                        | rmse (m) | 전역 등급         | 프레임별       | 병기값                  |
| ------------------------- | -------- | ------------- | ---------- | -------------------- |
| offline georef + `--dist` | 0.0779   | **`valid`**   | 3/3 valid  | n=9, 20/22 (90.9%)   |
| offline georef            | 0.1320   | **`valid`**   | 3/3 valid  | n=9, 20/22 (90.9%)   |
| offline `--no_georef`     | —        | **`unknown`** | —          | `t = 0`으로 채점 불가      |
| online georef + `--dist`  | 0.0630   | **`valid`**   | 4/4 valid  | n=11, 58/70 (82.9%)  |


지표 0.8 m 대비 여유가 약 **10배**다. 따라서 현재 리스크는 정확도가 아니라
**표본 수와 커버리지**다 — 절대 정확도가 측정된 프레임이 offline 3/22, online 4/70뿐이다.

임계값을 조여보면 프레임별 등급이 갈리는 것을 확인할 수 있다.

```bash
# rmse 0.0779 m 를 degraded 구간에 넣어 프레임별 분화를 확인
.venv/bin/python scripts/eval_own.py --mode offline \
    -i data/infiniq/offline -f data/infiniq/offline_matches \
    --params output/output-infiniq-offline-georef-dist/offline.json \
    --holdout data/infiniq/gt/offline.json \
    --pass_threshold_m 0.05 --degraded_threshold_m 0.10
# → 전역 degraded, 002 invalid (0.1045) / 006 valid (0.0295) / 008 degraded (0.0800)
```

`002`가 가장 나쁜 것은 우연이 아니다 — 그 프레임 세 점의 오차가 **전부 같은 방향**
(`|dY| 0.0988 m`, 추정 Y가 GT보다 일관되게 작음)인 계통 편향이다.
산포가 아니라 그 프레임의 포즈 자체가 치우쳐 있다는 신호다.

---



## 3. 정성 평가 (`render_panorama.py`)

등록된 프레임을 equirectangular 파노라마로 스티칭해 **이음선·뒤집힘**을 눈으로 확인한다.
IBA 월드 좌표계는 첫 등록 카메라 기준 임의 프레임이라, 스크립트가 자동으로 뒤집힘을 보정한다.

```bash
cd ptz_calib_dy

.venv/bin/python scripts/render_panorama.py \
    -i data/infiniq/offline \
    --params output-infiniq-offline/offline.json \
    -o output-infiniq-offline/panorama.png
```

등록되지 않은 프레임은 JSON에 없으므로 파노라마에도 포함되지 않는다.

---



## 4. 논문 지표 — 자체 데이터로는 비교 불가

**결론: infiniq 자체 데이터에는 논문 지표와 동일한 GT가 없어 논문값과 직접 비교할 수 없다.**
논문(PTZ-Calib) Table의 FLE / APE / IoU는 모두 **GT 카메라 파라미터 또는 GT 필드 호모그래피**를
전제로 하는데, 자체 데이터에는 그 어느 것도 존재하지 않는다.

### 4-1. 논문 지표가 요구하는 GT vs 자체 데이터가 가진 것


| 논문 지표                | 계산에 필요한 GT                      | infiniq 자체 데이터 | 가능 여부               |
| -------------------- | ------------------------------- | -------------- | ------------------- |
| FLE `                | f - f'                          | `              | 프레임별 GT focal (`K`) |
| APE_rot / APE_trans  | 프레임별 GT pose (`R`, `t`)         | 없음             | **불가**              |
| IoU_part / IoU_whole | 스포츠 필드 GT 호모그래피 (WorldCup14 전용) | 필드 자체가 없음      | **불가**              |


`data/infiniq/gt/` 는 이름이 `gt`지만 **GT 카메라 파라미터가 아니다**.
스키마상 `K`, `R`, `t`, `dist` 필드가 있긴 하나 **전부 빈 배열**이고,
실제로 들어 있는 값은 `marker`(2D 클릭 픽셀 + 3D 측량 좌표)뿐이다.

```bash
python -c "
import json
j = json.load(open('data/infiniq/gt/offline.json'))['cameras']
for name, v in j.items():
    m = v.get('marker') or {}
    print(name, 'K', v['K'], 'R', v['R'], 't', v['t'], '| marker pos:', len(m.get('pos', [])))
"
# ptz_frame_002 K [] R [] t [] | marker pos: 3   ← K/R/t 비어 있음
```

즉 `gt/` 디렉터리는 **홀드아웃 2D–3D 측량점**(§2-2)이지 GT pose가 아니다.
따라서 `../PTZ-Calib/scripts/eval_synthetic.py` / `eval_worldcup.py`는
자체 데이터에 **적용할 수 없다** (두 스크립트 모두 GT JSON에서 카메라 파라미터를 읽는다).

### 4-2. 자체 데이터에서 논문 지표를 대체하는 것

GT가 없으므로 논문 지표 대신 §2의 지표로 판정한다.


| 논문 지표   | 자체 데이터 대체 지표                                 | 성격                                        |
| ------- | -------------------------------------------- | ----------------------------------------- |
| APE_rot | `holdout_2d3d.errors_px` / `errors_m` (§2-1) | 절대 오차이지만 **측량점 기준**이라 pose 오차와 1:1 대응은 아님 |
| APE_rot | `ptz_rotation_check` (`--ptz_log`)           | PTZ 리드아웃 기반 **quasi-GT**, 상대 회전만          |
| FLE     | `focal_vs_zoom` (`--ptz_log`)                | zoom 단계별 **일관성**만, 절대 focal 오차 아님         |
| —       | `cross_frame_consistency_px`                 | 프레임 간 상대 일관성 (공통 bias는 못 잡음)              |
| IoU     | 없음                                           | 대체 불가                                     |


이들 중 어느 것도 논문 Table의 수치와 **같은 단위·같은 정의가 아니므로**,
"논문 대비 몇 % 성능" 식의 서술은 자체 데이터에서는 성립하지 않는다.
자체 데이터 결과는 §2 지표의 **절대값과 설정 간 상대 비교**로만 보고한다 (`RESULTS-own.md`).

### 4-3. 논문 지표 비교가 가능한 유일한 경로 — 공개 데이터셋 (참고)

논문값과의 비교는 GT가 함께 배포되는 **공개 데이터셋에서만** 성립하며,
이 저장소에서는 이미 수행해 `RESULTS.md`에 기록돼 있다
(Synthetic FLE/APE, WorldCup14 IoU 모두 C++ 기준선과 일치).

```bash
cd ptz_calib_dy

# synthetic 전 scene 일괄 (BA → FLE/APE 집계)
bash run_ptzba_synthetic.sh

# 단일 scene 확인
.venv/bin/python ../PTZ-Calib/scripts/eval_synthetic.py \
    --pred output-synthetic-offline/scene_01.json \
    --gt ../PTZ-Calib/data/synthetic/gt/scene_01.json

# 10 scene 집계
.venv/bin/python ../PTZ-Calib/scripts/agg_synthetic.py \
    --pred_dir output-synthetic-offline \
    --gt_dir ../PTZ-Calib/data/synthetic/gt \
    --label "synthetic offline (python)"
```

이 경로는 **포팅 정확성 검증용**이다.
자체 데이터 품질과는 무관하며, 자체 데이터 판정은 §2로 되돌아간다.

---



## 5. 실무 판정 체크리스트

두 층으로 나눠 본다.

- **최소 (필수)** — *이론적으로 맞게 진행됐는가.* 파이프라인이 정의대로 돌았고 출력이
모델의 불변식을 만족하는지 보는 **구조 검사**다. 여기서 하나라도 어긋나면 정확도 수치는
볼 의미가 없다 (맞아 보여도 우연이다).
- **권장 (품질)** — *결과가 쓸 만한 정확도인가.* 최소 항목이 전부 통과한 뒤에만 의미가 있는
**수치 검사**다.

> **아래 체크박스는 2026-08-25 infiniq offline 22장 실행 결과다.**
> 세 설정을 모두 돌려 채웠다 — **(a)** `--no_georef` / **(b)** georef / **(c)** georef + `--dist`.
> 다른 데이터로 재실행할 때는 체크를 지우고 다시 채운다.
>
> **한 줄 요약: 최소 14개 항목 전원 통과, 권장 지표도 (c)에서 전부 양호 —
> 홀드아웃 15.9 px / 7.8 cm, PTZ 회전 오차 0.32°, 파노라마 이음선 정상. 결과를 신뢰해도 된다.**
> 단 (c)의 프레임별 `k1` 산포와 홀드아웃 대 `Error 2d-3d` 3배 차이는 과적합 경향으로 계속 지켜본다.
>
> 재실행 결과는 `output/output-infiniq-offline{,-georef,-georef-dist}/offline.json`과
> **바이트 단위로 동일**했다 (`--seed` 기본값 기준 재현 확인).



### 최소 (필수) — 이론적으로 맞게 진행됐는가

**실행 로그**

- [x] `PTZ-IBA End: success` (초기 페어 등록 + incremental 루프 완료)
  ```
  → (a)(b)(c) 모두 `success`, exit 0
  ```
- [x] `Registered/Total`이 기대에 가깝다 (예: 22장 중 20장 이상)
  ```
  → 세 설정 모두 `20/22` (90.9%)로 동일
  ```
- [x] `unregistered` 목록이 설명 가능하다 (겹침이 없는 프레임 등) — 무작위로 빠지면 매칭을 의심
  ```
  → `ptz_frame_014`, `ptz_frame_019`. 두 프레임은 `pairs_matches.txt`에 **페어가 0건**
    (나머지 프레임 중앙값 24건)이라 등록이 불가능하다 — BA가 아니라 매칭 단계 입력 문제
  ```
- [x] GBA `final reprojection error all`이 발산하지 않았다 (유한값, 수 px~十数 px)
  ```
  → (a) 2.92 / (b) 3.15 / (c) 2.55 px
  ```

**출력 파라미터의 불변식**

- [x] JSON 카메라 수 == `Registered` 수
  ```
  → 세 설정 모두 20 == 20
  ```
- [x] 모든 `R`이 유효 회전행렬 (`det(R) = 1`, `R Rᵀ = I`), 모든 `f > 0`
  ```
  → `|det(R) - 1| < 1e-9`, `|R Rᵀ - I|max < 1e-9`, `f` 2449–5008 px
  ```
- [x] **모든 카메라의** `pos`**가 동일** — PTZ는 순수 회전 모델이라 광학 중심이 공유돼야 한다
  ```
  → 세 설정 모두 `pos` 성분별 spread = 0.000000 m
  ```
- [x] `focal_vs_zoom.monotonic_increasing == true` — zoom 단계가 오르면 focal도 올라야 한다
  ```
  → 세 설정 모두 `true` (zoom 1.5 → 2.2 → 2.8에서 f 단조 증가)
  ```

**실행 옵션과 저장된 값의 일치** (§2 도입 표)

- [x] `--no_georef` → `t = 0`, `dist = 0`, Summary `Error 2d-2d`/`Error 2d-3d`가 `-1`, holdout `skipped`
  ```
  → (a) `|t| = 0`, `dist` 전부 0, `Error 2d-2d/2d-3d = -1.0/-1.0`, holdout `skipped` — 전부 일치
  ```
- [x] georef → `t ≠ 0`이고 `Error 2d-2d`/`Error 2d-3d`가 `-1`이 아니다 (georef가 실제로 돌았다)
  ```
  → (b) `|t| = 2.689 m`, `1.226 / 15.060` · (c) `|t| = 2.925 m`, `1.041 / 5.387`
  ```
- [x] `--dist` → `dist[0] ≠ 0`이고 나머지 4개는 `0` (코드는 `k1`만 자유 변수 — `RESULTS.md` §5)
  ```
  → (c) `k1 ∈ [-0.0471, +0.3243]` 전 프레임 non-zero, `k2/k3/p1/p2` 전부 정확히 0
  ```

> `distType` 필드는 판정에 쓰지 않는다. `[data_io.py:149](ptzcalib/data_io.py#L149)`가
> `dist[0] < 1e-5`로 부호 없이 비교해서, 배럴 왜곡(`k1 < 0`)이면 `distType`이 `""`로 남는다.
> **(c) 실측이 그 경우다**: `k1 < 0`인 3프레임(`001`/`004`/`012`)만 `distType = ""`,
> 나머지 17프레임은 `"k1"`. 같은 실행 안에서 값이 갈리므로 `dist[0]`**을 직접 본다.**

**평가 자체가 유효했는가**

- [x] `eval_own.py`가 에러 없이 완주하고 `pairs_evaluated > 0`
  ```
  → 세 설정 모두 exit 0, `pairs_evaluated = 102`, 관측 21,237개
  ```
- [x] holdout을 썼다면 `skipped_frames == []` (홀드아웃 프레임이 전부 등록돼 있다)
  ```
  → (b)(c) 모두 `[]` — `ptz_frame_002/006/008` 3프레임 전부 등록됨
  ```
- [x] holdout `n_behind_camera == 0`, `n_no_plane_intersect == 0`
  ```
  (측량점이 카메라 뒤로 가거나 수평면과 안 만나면 그 점은 통계에서 빠진 것)
  → (b)(c) 모두 0 / 0 — 측량점 9개가 전부 통계에 반영됨
  ```

**일괄 확인 스니펫**

```bash
python -c "
import json, numpy as np
P = 'output/output-infiniq-offline-georef-dist'
cams = json.load(open(f'{P}/offline.json'))['cameras']
rep  = json.load(open(f'{P}/eval_report.json'))

R = [np.array(v['R']).reshape(3, 3) for v in cams.values()]
K = [np.array(v['K']).reshape(3, 3) for v in cams.values()]
pos = np.array([v['pos'] for v in cams.values()])
d0 = [v['dist'][0] for v in cams.values()]
rest = [max(abs(x) for x in v['dist'][1:]) for v in cams.values()]

print('cams == registered :', len(cams) == rep['registered'])
print('R valid            :', all(abs(np.linalg.det(r) - 1) < 1e-9 and
                                  np.abs(r @ r.T - np.eye(3)).max() < 1e-9 for r in R))
print('f > 0              :', all(k[0, 0] > 0 for k in K))
print('shared centre      :', float(np.abs(pos.max(0) - pos.min(0)).max()) < 1e-6)
print('georef (t != 0)    :', all(np.linalg.norm(v['t']) > 1e-6 for v in cams.values()))
print('dist: k1 only      :', all(abs(x) > 0 for x in d0) and max(rest) == 0)
fz = rep.get('focal_vs_zoom') or {}
print('focal monotonic    :', fz.get('monotonic_increasing'))
print('pairs_evaluated    :', rep['pairs_evaluated'])
h = rep.get('holdout_2d3d', {})
if h.get('status') == 'ok':
    print('holdout clean      :', not h['skipped_frames'] and
          h['n_behind_camera'] == 0 and h['n_no_plane_intersect'] == 0)
else:
    print('holdout            :', h.get('status'), h.get('reason', ''))
"
# 2026-08-25 (c) 실행 결과: 9개 항목 전부 True / pairs_evaluated 102
```



### 권장 (품질) — 결과가 쓸 만한 정확도인가

아래 괄호 안은 **2026-08-25 infiniq offline 22장 실측값**이다. 합격선이 아니라 참고 기준이며,
데이터셋이 바뀌면 자기 데이터의 기준선을 새로 잡아야 한다.

**상대 일관성**

- [x] `cross_frame_consistency_px` mean/median이 낮다
  ```
  → (a) 1.54 / 1.34 · (b) 1.54 / 1.32 · (c) 3.31 / 2.41 px (n=21,237).
    georef가 IBA 해를 건드리지 않았음이 (a)=(b)로 확인된다. (c)가 큰 이유는 §2-4 참고
  ```
- [x] `worst_pairs`가 전체 평균에서 크게 튀지 않는다
  ```
  — 특정 페어만 크면 전체 해가 아니라 그 프레임 매칭 문제다
  → (a) 2.95 px (`009`–`013`) · (b) 3.48 px · (c) 13.82 px (`017`–`020`).
    (a)(b)는 평균의 약 2배로 정상 범위
  ```

**절대 정확도** (georef를 했을 때만)

- [x] `calibration_quality`가 `valid` — 과제 지표 0.8 m 이내 (판정 로직은 §2-6)
  ```
  → (b) `valid` (0.132 m) · **(c)** `valid` **(0.078 m)** · (a) `unknown` (`t = 0`으로 채점 불가).
    프레임별도 (b)(c) 모두 3/3 `valid`. 단 `n_obs 9 / 3프레임`이라 표본이 얇다 —
    등급만 보지 말고 `verdict`의 병기값을 함께 읽는다
  ```
- [x] `holdout_2d3d.errors_px` / `errors_m` 확인 (§2-4) — 절대 정확도는 이 지표로만 판정
  ```
  → (b) 39.52 px / 13.2 cm · **(c) 15.91 px / 7.8 cm** (rmse, n=9)
  ```
- [x] `--dist` 유/무를 **홀드아웃 rmse로** 비교해 채택 여부 결정
  ```
  → 39.5 → 15.9 px, 13.2 → 7.8 cm로 절반 이하 → **`--dist` 채택**
  ```
- [x] 홀드아웃 px rmse가 Summary `Error 2d-3d`와 같은 자릿수 (홀드아웃만 크면 georef 과적합)
  ```
  → (b) 15.06 → 39.5 px (2.6배) · (c) 5.39 → 15.9 px (3.0배).
    자릿수는 같아 통과지만 **3배 차이는 과적합 경향** — 측량점 3프레임×3점이 적은 탓
  ```
- [x] 프레임별 `per_frame` 편차가 크지 않다
  ```
  → (c) 5.77 / 15.21 / 22.24 px (`006`/`002`/`008`). 최대·최소 약 4배 차 — 측량점 수를 늘릴 것
  ```

> **주의 — (c)의** `k1` **산포.** `k1`은 프레임마다 독립 추정된다.
> zoom 단계별 평균은 +0.023 → +0.190 → +0.247로 물리적으로 그럴듯하게 증가하지만,
> **단계 내 산포(0.135)가 단계 간 차이와 맞먹고** zoom 1.5에서는 부호까지 뒤집힌다
> (−0.047 … +0.088). 홀드아웃이 개선됐으므로 지금은 채택하되,
> 측량점이나 프레임이 늘면 재검토 대상이다.

**PTZ 로그 대조** (`--ptz_log`가 있을 때)

- [x] `ptz_rotation_check.abs_diff_deg` mean이 1° 미만
  ```
  → (a) 1.80° · (b) 1.09° · **(c) 0.32°** (max 0.88°, 115 페어). (c)만 기준 통과
  ```
- [x] `best_fit_scale`이 1에 가깝다 — 1에서 벗어나면 리드아웃 대 focal 스케일 불일치
  ```
  → (a) 1.075 · (b) 1.045 · **(c) 1.009**. `--dist`가 focal 스케일까지 바로잡았다
  ```
- [x] `residual_after_scale_deg`가 스케일 보정 전보다 뚜렷이 작다
  ```
  → (a) 1.80 → 0.23° · (b) 1.09 → 0.22° · (c) 0.32 → 0.23°.
    세 설정이 같은 0.23°로 수렴 = 잔차는 스케일 성분이었고 회전 자체는 정상
  ```
- [x] `focal_vs_zoom.f_over_zoom_spread_pct`가 작다 — zoom 배율 대비 focal 선형성
  ```
  → (a) 8.17% · (b) 8.06% · **(c) 7.40%**
  ```

**정성**

- [x] `render_panorama.py` 이음선이 자연스럽다 (§3)
  ```
  → (c) 기준 캔버스 4000×1386, 커버리지 75.3%, 최대 중첩 13프레임.
    바닥 타일선·걸레받이·벽 모서리가 프레임 경계를 넘어 끊김 없이 이어진다.
    고스팅은 화면 고정 OSD 타임스탬프뿐 — 장면에 붙은 요소가 아니라 정합 결함이 아니다
  ```



### 실패 신호

- `PTZ-IBA End: failed`
- 등록률이 현저히 낮음 (예: 2/30)
- GBA 재투영 오차가 수십~수백 px
- 카메라 `pos`가 프레임마다 다름 — 순수 회전 모델이 깨졌다는 뜻
- `focal_vs_zoom.monotonic_increasing == false` — zoom과 focal이 역전
- cross-frame consistency가 크거나 파노라마가 심하게 어긋남
- georef 후 홀드아웃 2D–3D 오차가 georef에 쓴 점의 `Error 2d-3d`보다 훨씬 큼 (과적합)
- `ptz_rotation_check`의 `best_fit_scale`이 1에서 크게 벗어남

*2026-08-25 실행에서는 위 실패 신호가 하나도 관측되지 않았다.*

---



## 6. 관련 파일


| 파일                                       | 역할                                 |
| ---------------------------------------- | ---------------------------------- |
| `scripts/run_ptz_ba.py`                  | IBA + (선택) georef 실행, Summary 출력   |
| `scripts/eval_own.py`                    | PTZ-IBA 정량 평가                      |
| `scripts/render_panorama.py`             | 파노라마 정성 확인                         |
| `data/infiniq/gt/*.json`                 | 홀드아웃 2D–3D 측량점 (GT pose 아님 — §4-1) |
| `../PTZ-Calib/scripts/eval_synthetic.py` | **공개 데이터셋 전용** 합성 GT 대비 FLE/APE    |
| `RESULTS.md`                             | 공개 데이터셋 논문 지표 재현 결과                |
| `RESULTS-own.md`                         | infiniq 실측 평가 결과 예시                |


