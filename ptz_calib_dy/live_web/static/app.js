'use strict';

// ---------------------------------------------------------------- state
const S = {
  status: null,
  calib: null,          // last successful calibration (server payload)
  overlay: null,        // survey / horizon / grid / keypoints, in frame pixels
  frame: [1920, 1080],  // calibrated-frame size in original pixels
  points: [],           // clicked points with their solved world coordinates
  hold: [],             // holdout measurements: user pixels scored against the survey GT
  holdSummary: null,
  holdSel: -1,
  frozen: false,
  busy: false,
  snapshot: null,       // Image of the calibrated frame
  hover: null,          // [u, v] under the cursor, for the loupe
  selected: -1,
  autoTimer: null,
  ptz: null,            // last /api/ptz payload: position + preset list + unit ranges
  ptzBusy: false,
  ptzChaseMin: 0,       // poll the pose faster while the camera is on its way
  ptzChaseUntil: 0,
};

const $ = (id) => document.getElementById(id);

// A calibration answers clicks on the frozen snapshot unconditionally - that
// frame IS the one it was solved on.  On the live view it stays valid only
// while the camera has not moved, which the server measures by phase
// correlation against the calibrated frame.
function calibValid() {
  if (!S.calib) return false;
  if (S.frozen) return true;
  const d = S.status && S.status.view_drift;
  return !(d && d.stale);
}
const drift = () => (S.status && S.status.view_drift) || null;
const stage = $('stage'), overlayC = $('overlay'), frozenC = $('frozen'), loupe = $('loupe');
const fmt = (x, n = 3) => (x === null || x === undefined || !isFinite(x) ? '—' : x.toFixed(n));

// ---------------------------------------------------------------- status
async function refreshStatus() {
  try {
    S.status = await (await fetch('/api/status')).json();
  } catch (e) {
    S.status = null;
  }
  renderStatus();
  if (S.status && S.status.ptz && S.status.ptz.ready) refreshPtz();
}

function pill(el, text, cls) {
  el.textContent = text;
  el.className = 'pill' + (cls ? ' ' + cls : '');
}

function renderStatus() {
  const s = S.status;
  if (!s) { pill($('pill-stream'), '서버 응답 없음', 'bad'); return; }

  const st = s.stream;
  pill($('pill-stream'),
    st.connected ? `RTSP ${st.fps.toFixed(1)} fps · ${st.frame_size ? st.frame_size.join('×') : '—'}`
                 : `RTSP 끊김 — ${st.last_error || '재연결 중'}`,
    st.connected ? 'ok' : 'bad');

  const en = s.engine;
  pill($('pill-engine'),
    en.ready ? `${en.device} · ref ${en.num_refs} · ${en.factor}` : '엔진 로딩 중',
    en.ready ? 'ok' : 'warn');

  if (S.calib) {
    const age = (Date.now() - Date.parse(S.calib.calibrated_at)) / 1000;
    pill($('pill-calib'),
      `캘리브레이션 ${age < 1 ? '방금' : Math.round(age) + '초 전'} · ${fmt(S.calib.reproj_error_px, 2)} px`,
      'ok');
  } else {
    pill($('pill-calib'), '미캘리브레이션', 'warn');
  }

  const d = drift();
  const dp = $('pill-drift');
  dp.hidden = !(S.calib && d);
  if (S.calib && d) {
    pill(dp,
      d.stale ? `화면 이동 ${fmt(d.shift_px, 1)} px — 재캘리브레이션 필요`
              : `화면 정지 ${fmt(d.shift_px, 2)} px (허용 ${fmt(d.tolerance_px, 1)})`,
      d.stale ? 'bad' : 'ok');
    // re-solve as soon as the camera settles somewhere new
    if (d.stale && !S.frozen && $('auto-move').checked && !S.busy) calibrate({ freeze: false });
    updateHint();
  }

  $('cfg-body').innerHTML = kv([
    ['참조 파라미터', short(en.ref_params)],
    ...offlineQualityRows(en.offline_quality),
    ['참조 프레임 수', en.num_refs],
    ['서베이 점', en.num_survey],
    ['팩터', en.factor + (en.factor === 'FDist' ? ' (dist 적용)' : '')],
    ['평면 Z 기본값', en.plane_z + ' m'],
    ['RTSP', st.url],
  ]);
}

const short = (p) => (p || '').split('/').slice(-2).join('/');

// The offline grade of the reference bundle (EVALUATION.md 2-6).  It is a
// property of `ref_params`, not of the live solve: the surveyed points it was
// measured against have no physical marker, so no live frame can re-observe
// them.  Shown in the settings card so the inherited accuracy is never implicit.
const Q_CLASS = { valid: 'ok', degraded: 'warn', invalid: 'bad' };

function offlineQualityRows(q) {
  if (!q) return [['오프라인 등급', '—', 'dim']];
  if (q.status !== 'ok') {
    const why = { missing: '리포트 없음', stale: '판정 이전 리포트', error: '리포트 읽기 실패' };
    return [['오프라인 등급', `unknown — ${why[q.status] || q.status}`, 'dim']];
  }
  const cls = Q_CLASS[q.quality] || 'dim';
  const rows = [['오프라인 등급', q.quality, cls]];
  // a georeferenced bundle has an RMSE; a --no_georef one is unjudged (t = 0)
  if (q.rmse_m == null) {
    rows.push(['오프라인 RMSE', '측정 없음', 'dim']);
  } else {
    rows.push(['오프라인 RMSE', `${fmt(q.rmse_m, 3)} m (기준 ${q.pass_threshold_m} m)`, cls]);
  }
  // the grade is rmse only; sample size and registration rate qualify it
  if (q.n_obs) rows.push(['판정 표본', `${q.n_obs}점 / ${q.n_frames_measured}프레임`, 'dim']);
  if (q.rate != null) {
    rows.push(['오프라인 등록률', `${q.registered}/${q.total} (${(100 * q.rate).toFixed(1)}%)`, 'dim']);
  }
  return rows;
}

function kv(rows) {
  return rows.map(([k, v, cls]) =>
    `<span class="k">${k}</span><span class="v ${cls || ''}">${v}</span>`).join('');
}

// ---------------------------------------------------------------- calibrate
async function calibrate(opts) {
  const freeze = !!(opts && opts.freeze);
  if (S.busy) return;
  S.busy = true;
  const btn = freeze ? $('btn-freeze') : $('btn-calib');
  btn.classList.add('busy');
  $('btn-calib').disabled = true;
  $('btn-freeze').disabled = true;
  try {
    const res = await (await fetch('/api/calibrate', { method: 'POST' })).json();
    if (!res.ok) {
      renderCalib(res);
      pill($('pill-calib'), '캘리브레이션 실패', 'bad');
      return;
    }
    S.calib = res;
    S.frame = res.frame_size;
    await loadSnapshot();
    await loadOverlay();
    setFrozen(freeze);
    await resolveAllPoints();   // re-solve existing clicks against the new pose
    fillHoldoutTargets();
    await scoreHoldout();       // ... and re-score the holdout measurements
    renderCalib(res);
    await refreshStatus();      // pulls the fresh (near-zero) drift reading
  } catch (e) {
    renderCalib({ ok: false, reason: String(e) });
  } finally {
    S.busy = false;
    btn.classList.remove('busy');
    $('btn-calib').disabled = false;
    $('btn-freeze').disabled = false;
  }
}

function loadSnapshot() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => { S.snapshot = img; drawFrozen(); resolve(); };
    img.onerror = () => resolve();
    img.src = '/api/snapshot.jpg?t=' + Date.now();
  });
}

async function loadOverlay() {
  try {
    const z = encodeURIComponent($('plane-z').value || '0');
    S.overlay = await (await fetch('/api/overlay?plane_z=' + z)).json();
  } catch (e) {
    S.overlay = null;
  }
}

function renderCalib(res) {
  const b = $('calib-body');
  if (!res.ok) {
    b.innerHTML = kv([
      ['상태', '실패', 'bad'],
      ['키포인트', res.num_keypoints ?? '—'],
      ['최다 매칭', res.candidates && res.candidates.length ? `${res.candidates[0].ref} · inlier ${res.candidates[0].inlier}` : '—'],
    ]) + `<div class="full">${res.reason || '알 수 없는 실패'}</div>`;
    $('refthumb').hidden = true;
    return;
  }
  const c = res.camera, t = res.timing_s;
  b.innerHTML = kv([
    ['상태', '성공', 'ok'],
    ['참조 프레임', res.best_ref],
    ['매칭 inlier', res.num_matches],
    ['2D-2D 재투영', fmt(res.reproj_error_px, 2) + ' px', res.reproj_error_px < 5 ? 'ok' : ''],
    ['LM 반복', res.num_iterations],
    ['초점 f', fmt(c.fx, 1) + ' px'],
    ['FOV', `${fmt(c.fov_x_deg, 1)}° × ${fmt(c.fov_y_deg, 1)}°`],
    ['왜곡 k1', fmt(c.dist[0], 5)],
    ['카메라 위치', c.pos.map((v) => fmt(v, 3)).join(', ')],
    ['소요', `추출 ${Math.round(t.extract * 1000)} · 매칭 ${Math.round(t.match * 1000)} · 해 ${Math.round(t.solve * 1000)} ms`],
    ['프레임 수신', (res.recv_at || '').split('T')[1] || '—'],
  ]);
  const th = $('refthumb');
  th.src = '/api/refimage/' + encodeURIComponent(res.best_ref);
  th.hidden = false;
}

// ------------------------------------------------- holdout reprojection error
// A clicked pixel -> ray -> Z = GT's Z -> compare with the surveyed 3D point.
// The pixel must come from the user: projecting the survey point to get one and
// back-projecting it again is the identity and always reads 0.000 m.
// The offline georef+dist solve scores RMSE 0.078 m / max 0.110 m on the holdout
// set, so a live calibration near 0.08 m is as good as the reference bundle;
// past ~0.30 m the pose is wrong, not noisy.
const HD_OK = 0.12, HD_WARN = 0.30;

// server reasons, shortened for the table cell.  The unknown-target one echoes
// the id the client sent, so it is mapped to a fixed string rather than injected.
const hdReason = (r) =>
  /unknown holdout point/.test(r) ? '그런 holdout 점이 없습니다'
  : /never meets/.test(r) ? '평면과 교차 없음 (지평선 위?)'
  : r;

async function scoreHoldout() {
  if (!S.hold.length) { S.holdSummary = null; renderHoldout(); draw(); return; }
  if (!S.calib) { flashHint('먼저 [캘리브레이션]을 누르세요'); return; }
  try {
    const res = await fetch('/api/holdout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: S.hold.map((m) => [m.u, m.v]),
        targets: S.hold.map((m) => m.target || null),
        source: S.frozen ? 'frozen' : 'live',   // the server re-checks the drift
      }),
    });
    const body = await res.json();
    if (!res.ok) {
      if (body.view_drift) {
        S.status = Object.assign({}, S.status, { view_drift: body.view_drift });
        renderStatus();
      }
      flashHint(body.error || 'holdout 오차 계산 실패');
      return;
    }
    S.hold.forEach((m, i) => Object.assign(m, body.points[i]));
    S.holdSummary = body.summary;
  } catch (e) {
    flashHint(String(e));
    return;
  }
  renderHoldout();
  draw();
}

function addHoldout(u, v) {
  if (!isFinite(u) || !isFinite(v)) { flashHint('u, v 픽셀값을 입력하세요'); return; }
  S.hold.push({ u, v, target: $('hd-target').value || null, valid: false });
  scoreHoldout();
}

function renderHoldout() {
  const num = $('hd-num'), sub = $('hd-sub'), body = $('hd-body'), tbl = $('hd-pts');
  const sm = S.hold.length ? S.holdSummary : null;

  if (!sm) {
    num.parentElement.className = 'bignum none';
    num.textContent = '—';
    const pending = S.hold.some((m) => !m.valid && !m.reason);
    sub.textContent = pending ? '계산 중…'
      : S.hold.length ? '유효한 측정이 없습니다 — 지평선 위나 바닥 밖을 찍었는지 확인하세요'
      : S.calib ? '2D 픽셀을 입력하거나 화면을 클릭하세요' : '캘리브레이션 후 측정할 수 있습니다';
    body.innerHTML = '';
  } else {
    num.parentElement.className = 'bignum ' +
      (sm.rmse <= HD_OK ? 'ok' : sm.rmse <= HD_WARN ? 'warn' : 'bad');
    num.textContent = fmt(sm.rmse, 3);
    sub.textContent = `RMSE · 평균 ${fmt(sm.mean, 3)} · 중앙값 ${fmt(sm.median, 3)} · 최대 ${fmt(sm.max, 3)} m`;

    const ok = S.hold.filter((m) => m.valid);
    const bias = [0, 1].map((k) =>
      ok.reduce((a, m) => a + (k ? m.dy_m : m.dx_m), 0) / ok.length);
    const px = Math.sqrt(ok.reduce((a, m) => a + m.error_px ** 2, 0) / ok.length);
    body.innerHTML = kv([
      ['측정 점', `${sm.n} / ${S.hold.length}`],
      ['픽셀 오차', fmt(px, 1) + ' px (RMSE)'],
      ['X/Y 편향', `${fmt(bias[0], 3)}, ${fmt(bias[1], 3)} m`],
      ['대상', [...new Set(ok.map((m) => m.target))].sort().join(', ')],
    ]);
  }

  tbl.hidden = !S.hold.length;
  tbl.querySelector('tbody').innerHTML = S.hold.map((m, i) => {
    const sel = i === S.holdSel ? ' class="sel"' : '';
    const del = `<td><button class="mini" data-del="${i}" title="삭제">×</button></td>`;
    if (!m.valid) {
      return `<tr${sel} data-h="${i}"><td>${i + 1}</td><td>${m.target || '—'}</td>` +
        `<td>${fmt(m.u, 1)}, ${fmt(m.v, 1)}</td>` +
        `<td class="bad" colspan="3">${m.reason ? hdReason(m.reason) : '계산 중…'}</td>${del}</tr>`;
    }
    // an automatic pick further than 0.30 m is closer to being between two
    // markers than on one - the id, not the error, is what is in doubt
    const amb = m.ambiguous
      ? ` <span class="warn" title="가장 가까운 holdout 점이 ${fmt(m.nearest_m, 2)} m 떨어져 있습니다 - 대상을 직접 지정하세요">⚠</span>`
      : '';
    return `<tr${sel} data-h="${i}">
      <td>${i + 1}</td>
      <td>${m.target}${m.target_auto ? '<span class="dim">*</span>' : ''}${amb}</td>
      <td>${fmt(m.u, 1)}, ${fmt(m.v, 1)}</td>
      <td class="xyz">${fmt(m.world[0], 2)}, ${fmt(m.world[1], 2)}</td>
      <td class="err">${fmt(m.error_m, 3)}</td>
      <td>${fmt(m.error_px, 1)}</td>${del}</tr>`;
  }).join('');

  tbl.querySelectorAll('button[data-del]').forEach((b) => {
    b.onclick = (ev) => {
      ev.stopPropagation();
      S.hold.splice(+b.dataset.del, 1);
      S.holdSel = -1;
      scoreHoldout();
    };
  });
  tbl.querySelectorAll('tr[data-h]').forEach((tr) => {
    tr.onclick = () => { S.holdSel = +tr.dataset.h; renderHoldout(); draw(); };
  });
}

// the holdout targets worth offering are the ones currently in frame, which is
// exactly what /api/overlay already reports
function fillHoldoutTargets() {
  const sel = $('hd-target');
  const keep = sel.value;
  const ids = ((S.overlay && S.overlay.survey) || [])
    .filter((s) => s.split === 'holdout').map((s) => s.id).sort();
  sel.innerHTML = '<option value="">자동 (최근접)</option>' +
    ids.map((id) => `<option value="${id}">${id}</option>`).join('');
  if (ids.includes(keep)) sel.value = keep;
}

function drawHoldout(g, P) {
  S.hold.forEach((m, i) => {
    if (!m.valid) return;
    const [x, y] = P(m.u, m.v);
    const [gx, gy] = P(m.pixel_gt[0], m.pixel_gt[1]);
    const sel = i === S.holdSel;
    // GT projected by the current solve --- the click; the gap IS the error
    g.strokeStyle = m.error_m <= HD_OK ? '#4ade80' : m.error_m <= HD_WARN ? '#fbbf24' : '#f87171';
    g.lineWidth = sel ? 2.5 : 1.5;
    const colour = g.strokeStyle;
    g.beginPath(); g.moveTo(gx, gy); g.lineTo(x, y); g.stroke();
    g.beginPath(); g.arc(x, y, sel ? 6 : 4, 0, Math.PI * 2); g.stroke();
    g.fillStyle = colour;
    g.beginPath(); g.arc(gx, gy, 2.5, 0, Math.PI * 2); g.fill();

    const label = `${m.target} ${fmt(m.error_m, 3)} m`;
    g.font = '600 11px ui-monospace, monospace';
    const w = g.measureText(label).width;
    g.fillStyle = 'rgba(5,7,10,0.82)';
    g.fillRect(x + 9, y + 4, w + 10, 17);
    g.fillStyle = colour;
    g.fillText(label, x + 14, y + 16);
  });
}

// ---------------------------------------------------------------- view mode
function setFrozen(on) {
  S.frozen = on;
  stage.classList.toggle('frozen', on);
  $('btn-live').disabled = !on;
  $('badge').textContent = on ? 'CALIBRATED FRAME' : 'LIVE';
  $('badge').classList.toggle('frozen', on);
  loupe.style.display = 'none';
  updateHint();
  draw();
}

function updateHint() {
  const h = $('hint');
  if (flashHint._t) return;   // a flash message is on screen
  const d = drift();
  let text, alarm = false;
  if (!S.calib) {
    text = '캘리브레이션을 누르면 라이브 화면에서 바로 클릭할 수 있습니다';
  } else if ($('hd-pick').checked) {
    text = 'holdout 측정 모드 — 분홍 십자(holdout 점)의 중심을 클릭하면 GT와의 3D 오차가 나옵니다';
  } else if (S.frozen) {
    text = '캘리브레이션된 정지 프레임 — 클릭하면 2D 픽셀과 바닥 평면 3D 좌표가 나옵니다';
  } else if (d && d.stale) {
    text = `화면이 ${fmt(d.shift_px, 1)} px 이동했습니다 — 캘리브레이션이 더 이상 유효하지 않습니다. 다시 캘리브레이션하세요`;
    alarm = true;
  } else {
    text = '라이브 화면 — 클릭하면 2D 픽셀과 바닥 평면 3D 좌표가 나옵니다'
         + (d ? ` (화면 이동 ${fmt(d.shift_px, 2)} px)` : '');
  }
  h.textContent = text;
  h.classList.toggle('alarm', alarm);
}

function drawFrozen() {
  if (!S.snapshot) return;
  frozenC.width = S.snapshot.naturalWidth;
  frozenC.height = S.snapshot.naturalHeight;
  frozenC.getContext('2d').drawImage(S.snapshot, 0, 0);
}

// ---------------------------------------------------------------- overlay draw
function stageSize() {
  const r = stage.getBoundingClientRect();
  return [r.width, r.height];
}

function draw() {
  const [w, h] = stageSize();
  if (w < 2 || h < 2) return;
  const dpr = window.devicePixelRatio || 1;
  overlayC.width = Math.round(w * dpr);
  overlayC.height = Math.round(h * dpr);
  const g = overlayC.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const sx = w / S.frame[0], sy = h / S.frame[1];
  const P = (u, v) => [u * sx, v * sy];

  // frame pixels only mean something while the calibration applies to what is
  // on screen - on the frozen snapshot always, on the live view until it moves
  if (calibValid() && S.overlay) {
    if ($('show-grid').checked) drawGrid(g, P, w, h);
    if ($('show-horizon').checked) drawHorizon(g, P);
    if ($('show-kpts').checked) drawKeypoints(g, P);
    if ($('show-survey').checked) drawSurvey(g, P);
    drawHoldout(g, P);
    drawMarkers(g, P);
  }
  drawBEV();
}

// small dark plate behind a caption so it stays readable over the video
function drawChip(g, x, y, text, color) {
  g.font = '11px ui-monospace, monospace';
  const cw = g.measureText(text).width;
  g.fillStyle = 'rgba(5,7,10,0.75)';
  g.fillRect(x, y, cw + 10, 16);
  g.fillStyle = color;
  g.fillText(text, x + 5, y + 12);
}

// a point on the polyline that is actually on screen, so the caption is visible;
// 'right' suits the near-horizontal Y=0 line, 'bottom' the near-vertical X=0 one
function polylineAnchor(pixels, P, w, h, prefer) {
  let best = null, bestKey = -Infinity;
  for (const [u, v] of pixels) {
    const [x, y] = P(u, v);
    if (x < 0 || x > w || y < 0 || y > h) continue;
    const key = prefer === 'right' ? x : y;
    if (key > bestKey) { bestKey = key; best = [x, y]; }
  }
  return best;
}

function drawGrid(g, P, w, h) {
  const lines = S.overlay.grid || [];
  for (const ln of lines) {
    g.beginPath();
    let started = false;
    for (const [u, v] of ln.pixels) {
      const [x, y] = P(u, v);
      if (!started) { g.moveTo(x, y); started = true; } else { g.lineTo(x, y); }
    }
    g.strokeStyle = ln.major ? 'rgba(87,183,255,0.55)' : 'rgba(255,255,255,0.16)';
    g.lineWidth = ln.major ? 1.6 : 1;
    g.stroke();
  }

  // name the two origin lines - without this the highlighted pair is just
  // "some brighter grid line" and the reader cannot tell which axis is which
  for (const ln of lines) {
    if (!ln.major) continue;
    const anchor = polylineAnchor(ln.pixels, P, w, h, ln.axis === 'y' ? 'right' : 'bottom');
    if (!anchor) continue;
    const cap = ln.axis === 'x' ? 'X = 0 m (+Y 북 방향)' : 'Y = 0 m (+X 동 방향)';
    g.font = '11px ui-monospace, monospace';
    const cw = g.measureText(cap).width + 10;
    const x = Math.min(Math.max(anchor[0] + 6, 4), w - cw - 4);
    const y = Math.min(Math.max(anchor[1] - 20, 4), h - 20);
    drawChip(g, x, y, cap, 'rgba(87,183,255,0.95)');
  }
}

function drawHorizon(g, P) {
  const hz = S.overlay.horizon;
  if (!hz) return;
  g.beginPath();
  hz.forEach(([u, v], i) => { const [x, y] = P(u, v); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = 'rgba(251,191,36,0.8)';
  g.lineWidth = 1.5;
  g.setLineDash([7, 5]);
  g.stroke();
  g.setLineDash([]);
  // anchor the caption a third of the way in so it clears the LIVE/CALIBRATED badge
  const mid = hz[Math.floor(hz.length / 3)];
  const [x0, y0] = P(mid[0], mid[1]);
  drawChip(g, x0 + 4, y0 + 4, '지평선 — 위쪽은 바닥 평면과 만나지 않음', 'rgba(251,191,36,0.95)');
}

function drawKeypoints(g, P) {
  g.fillStyle = 'rgba(74,222,128,0.55)';
  for (const [u, v] of (S.overlay.matched_keypoints || [])) {
    const [x, y] = P(u, v);
    g.fillRect(x - 1, y - 1, 2, 2);
  }
}

function drawSurvey(g, P) {
  g.font = '11px ui-monospace, monospace';
  for (const s of (S.overlay.survey || [])) {
    const [x, y] = P(s.pixel[0], s.pixel[1]);
    const c = s.split === 'holdout' ? '#f472b6' : '#4ade80';
    g.strokeStyle = c;
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(x - 7, y); g.lineTo(x + 7, y);
    g.moveTo(x, y - 7); g.lineTo(x, y + 7);
    g.stroke();
    g.fillStyle = c;
    g.fillText(s.id, x + 9, y - 4);
  }
}

function drawMarkers(g, P) {
  S.points.forEach((p, i) => {
    const [x, y] = P(p.u, p.v);
    const sel = i === S.selected;
    g.strokeStyle = p.valid ? '#57b7ff' : '#f87171';
    g.lineWidth = sel ? 2.5 : 1.8;
    g.beginPath(); g.arc(x, y, sel ? 10 : 7, 0, Math.PI * 2); g.stroke();
    g.beginPath();
    g.moveTo(x - 12, y); g.lineTo(x - 3, y);
    g.moveTo(x + 3, y); g.lineTo(x + 12, y);
    g.moveTo(x, y - 12); g.lineTo(x, y - 3);
    g.moveTo(x, y + 3); g.lineTo(x, y + 12);
    g.stroke();

    const label = p.valid
      ? `#${i + 1}  ${fmt(p.world[0], 2)}, ${fmt(p.world[1], 2)}, ${fmt(p.world[2], 2)} m`
      : `#${i + 1}  평면과 교차 없음`;
    g.font = '600 12px ui-monospace, monospace';
    const wpx = g.measureText(label).width;
    g.fillStyle = 'rgba(5,7,10,0.82)';
    g.fillRect(x + 14, y - 20, wpx + 12, 20);
    g.fillStyle = p.valid ? '#e6ebf2' : '#f87171';
    g.fillText(label, x + 20, y - 6);
  });
}

// ---------------------------------------------------------------- clicking
function eventToFramePixel(ev) {
  const r = stage.getBoundingClientRect();
  const u = ((ev.clientX - r.left) / r.width) * S.frame[0];
  const v = ((ev.clientY - r.top) / r.height) * S.frame[1];
  return [u, v];
}

function flashHint(text) {
  const h = $('hint');
  h.textContent = text;
  h.style.color = '#fbbf24';
  clearTimeout(flashHint._t);
  flashHint._t = setTimeout(() => { flashHint._t = null; h.style.color = ''; updateHint(); }, 2600);
}

async function addPoint(ev) {
  if (!S.calib) {
    flashHint('먼저 [캘리브레이션]을 누르세요');
    return;
  }
  if (!calibValid()) {
    flashHint('화면이 이동했습니다 — 다시 캘리브레이션한 뒤 클릭하세요');
    return;
  }
  const [u, v] = eventToFramePixel(ev);
  if ($('hd-pick').checked) { addHoldout(u, v); return; }
  const p = { u, v, valid: false, pending: true };
  S.points.push(p);
  S.selected = S.points.length - 1;
  renderPoints();
  draw();

  const solved = await unproject([[u, v]]);
  if (solved && solved.points[0]) Object.assign(p, normalise(solved.points[0]));
  p.pending = false;
  renderPoints();
  draw();
}

function normalise(r) {
  return {
    valid: r.valid,
    world: r.world || null,
    roundtrip: r.roundtrip_px,
    range: r.range_m,
    ground: r.ground_range_m,
    origin: r.origin_range_m,
    reason: r.reason || '',
  };
}

async function unproject(pts) {
  try {
    const res = await fetch('/api/unproject', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: pts,
        plane_z: parseFloat($('plane-z').value) || 0,
        source: S.frozen ? 'frozen' : 'live',   // the server re-checks the drift
      }),
    });
    const body = await res.json();
    if (!res.ok) {
      if (body.view_drift) {
        S.status = Object.assign({}, S.status, { view_drift: body.view_drift });
        renderStatus();
      }
      flashHint(body.error || '좌표 계산 실패');
      return null;
    }
    return body;
  } catch (e) { return null; }
}

async function resolveAllPoints() {
  if (!S.points.length) return;
  const solved = await unproject(S.points.map((p) => [p.u, p.v]));
  if (!solved) return;
  S.points.forEach((p, i) => Object.assign(p, normalise(solved.points[i])));
  renderPoints();
}

function renderPoints() {
  const tb = document.querySelector('#pts tbody');
  $('pts-empty').style.display = S.points.length ? 'none' : '';
  tb.innerHTML = S.points.map((p, i) => {
    if (p.pending) return `<tr><td>${i + 1}</td><td>${fmt(p.u, 1)}, ${fmt(p.v, 1)}</td><td colspan="3">계산 중…</td></tr>`;
    if (!p.valid) return `<tr class="${i === S.selected ? 'sel' : ''}"><td>${i + 1}</td>` +
      `<td>${fmt(p.u, 1)}, ${fmt(p.v, 1)}</td><td class="bad" colspan="3">${p.reason || '평면과 교차 없음'}</td></tr>`;
    // the click -> 3D -> click round trip only breaks when the distortion model
    // is extrapolated outside the image; flag it instead of spending a column
    const warn = p.roundtrip > 0.5
      ? ` <span class="warn" title="역변환 왕복 오차 ${fmt(p.roundtrip, 2)} px — 화면 밖이라 왜곡 모델 외삽 구간입니다">⚠</span>`
      : '';
    return `<tr class="${i === S.selected ? 'sel' : ''}" data-i="${i}">
      <td>${i + 1}</td>
      <td>${fmt(p.u, 1)}, ${fmt(p.v, 1)}${warn}</td>
      <td class="xyz">${fmt(p.world[0])}, ${fmt(p.world[1])}, ${fmt(p.world[2])}</td>
      <td>${fmt(p.origin, 2)} m</td>
      <td>${fmt(p.range, 2)} m</td></tr>`;
  }).join('');

  tb.querySelectorAll('tr[data-i]').forEach((tr) => {
    tr.onclick = () => { S.selected = +tr.dataset.i; renderPoints(); draw(); };
  });
}

// ---------------------------------------------------------------- loupe
function onMove(ev) {
  if (!S.frozen || !S.snapshot) return;   // the loupe magnifies the still frame
  const [u, v] = eventToFramePixel(ev);
  S.hover = [u, v];
  const r = stage.getBoundingClientRect();
  const size = 150, zoom = 6;
  loupe.style.display = 'block';
  loupe.style.left = Math.min(Math.max(ev.clientX - r.left + 20, 4), r.width - size - 4) + 'px';
  loupe.style.top = Math.min(Math.max(ev.clientY - r.top - size - 20, 4), r.height - size - 4) + 'px';

  const g = loupe.getContext('2d');
  const src = size / zoom;
  g.imageSmoothingEnabled = false;
  g.clearRect(0, 0, size, size);
  g.drawImage(S.snapshot, u - src / 2, v - src / 2, src, src, 0, 0, size, size);
  g.strokeStyle = '#57b7ff'; g.lineWidth = 1;
  g.beginPath();
  g.moveTo(size / 2, 0); g.lineTo(size / 2, size);
  g.moveTo(0, size / 2); g.lineTo(size, size / 2);
  g.stroke();
  g.fillStyle = 'rgba(5,7,10,0.8)';
  g.fillRect(0, size - 18, size, 18);
  g.fillStyle = '#cfd8e6';
  g.font = '11px ui-monospace, monospace';
  g.fillText(`${u.toFixed(1)}, ${v.toFixed(1)}`, 8, size - 5);
}

// ---------------------------------------------------------------- BEV
function drawBEV() {
  const c = $('bev');
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 460, h = 330;
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const world = [];
  const cam = S.calib ? S.calib.camera.pos : null;
  if (cam) world.push([cam[0], cam[1]]);
  for (const p of S.points) if (p.valid) world.push([p.world[0], p.world[1]]);
  const survey = (S.overlay && S.overlay.survey) || [];
  for (const s of survey) world.push([s.world[0], s.world[1]]);
  if (!world.length) {
    g.fillStyle = '#93a0b4'; g.font = '13px sans-serif';
    g.fillText('캘리브레이션 후 표시됩니다', 14, 24);
    return;
  }

  const xs = world.map((p) => p[0]), ys = world.map((p) => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = Math.max(1, 0.12 * Math.max(x1 - x0, y1 - y0));
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const scale = Math.min((w - 40) / Math.max(x1 - x0, 1e-6), (h - 40) / Math.max(y1 - y0, 1e-6));
  const cxp = (w - (x1 - x0) * scale) / 2, cyp = (h - (y1 - y0) * scale) / 2;
  const M = (x, y) => [cxp + (x - x0) * scale, h - cyp - (y - y0) * scale];  // +Y up

  // 1 m grid
  g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 1;
  const step = scale < 12 ? 5 : 1;
  for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
    const [px] = M(x, y0); g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
  }
  for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
    const [, py] = M(x0, y); g.beginPath(); g.moveTo(0, py); g.lineTo(w, py); g.stroke();
  }
  // origin axes
  if (x0 < 0 && x1 > 0) { const [px] = M(0, 0); g.strokeStyle = 'rgba(255,255,255,0.22)'; g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke(); }
  if (y0 < 0 && y1 > 0) { const [, py] = M(0, 0); g.strokeStyle = 'rgba(255,255,255,0.22)'; g.beginPath(); g.moveTo(0, py); g.lineTo(w, py); g.stroke(); }

  g.font = '10px ui-monospace, monospace';
  for (const s of survey) {
    const [px, py] = M(s.world[0], s.world[1]);
    g.fillStyle = s.split === 'holdout' ? '#f472b6' : '#4ade80';
    g.beginPath(); g.arc(px, py, 3, 0, Math.PI * 2); g.fill();
  }
  if (cam) {
    const [px, py] = M(cam[0], cam[1]);
    g.fillStyle = '#fbbf24';
    g.beginPath(); g.arc(px, py, 5, 0, Math.PI * 2); g.fill();
    g.fillText(`cam z=${fmt(cam[2], 2)}`, px + 8, py - 6);
    for (const p of S.points) {
      if (!p.valid) continue;
      const [qx, qy] = M(p.world[0], p.world[1]);
      g.strokeStyle = 'rgba(87,183,255,0.35)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(px, py); g.lineTo(qx, qy); g.stroke();
    }
  }
  S.points.forEach((p, i) => {
    if (!p.valid) return;
    const [px, py] = M(p.world[0], p.world[1]);
    g.fillStyle = i === S.selected ? '#ffffff' : '#57b7ff';
    g.beginPath(); g.arc(px, py, 4.5, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#cfd8e6';
    g.fillText('#' + (i + 1), px + 7, py + 4);
  });

  g.fillStyle = '#93a0b4'; g.font = '11px ui-monospace, monospace';
  g.fillText(`1칸 ${step} m · +X 동 / +Y 북`, 10, h - 10);
}

// ---------------------------------------------------------------- PTZ control
// The camera speaks the ONVIF generic space (pan/tilt -1..1, zoom 0..1); people
// think in degrees and magnification.  The server converts with constants
// measured against this camera, and hands the ranges over so the input limits
// are never a second copy of them.
const ptzRaw = () => $('ptz-raw').checked;

function ptzAxis(name) {
  const u = S.ptz && S.ptz.units && S.ptz.units[name];
  return u || { norm: [-1, 1], human: [-1, 1], unit: '' };
}

// value shown in the box <-> value the API wants, for the current unit mode
const ptzShown = (name, pos) =>
  ptzRaw() ? pos[name] : pos[{ pan: 'pan_deg', tilt: 'tilt_deg', zoom: 'zoom_x' }[name]];

async function refreshPtz(withPresets) {
  const want = withPresets || !(S.ptz && S.ptz.presets);
  try {
    const res = await (await fetch('/api/ptz?presets=' + (want ? '1' : '0'))).json();
    if (!res.available) { $('ptz-card').hidden = true; return; }
    // a presets=0 poll must not drop the list we already have
    S.ptz = Object.assign({}, S.ptz, res);
  } catch (e) {
    return;
  }
  renderPtz();
}

function renderPtz() {
  const p = S.ptz;
  if (!p || !p.position) return;
  $('ptz-card').hidden = false;
  const q = p.position;

  $('ptz-now').innerHTML = kv([
    ['현재', `${fmt(q.pan_deg, 2)}° · ${fmt(q.tilt_deg, 2)}° · ${fmt(q.zoom_x, 2)}×`],
    ['정규화', `${fmt(q.pan, 6)} / ${fmt(q.tilt, 6)} / ${fmt(q.zoom, 6)}`],
    ['상태', q.moving ? '이동 중' : '정지', q.moving ? 'moving' : 'ok'],
  ]);

  // the first payload is also what fills the boxes - typing into an empty form
  // and pressing 이동 should not be how you discover the camera's units
  if (!$('ptz-pan').value) ptzFillFromCurrent();
  ptzApplyUnits();

  const sel = $('ptz-preset');
  if (p.presets && sel.dataset.n !== String(p.presets.length)) {
    sel.dataset.n = String(p.presets.length);
    const user = p.presets.filter((x) => x.kind === 'user');
    const hidden = p.presets.length - user.length;
    sel.innerHTML = '<option value="">선택…</option>' + user.map((x) =>
      `<option value="${x.token}">#${x.token} ${x.name} — ` +
      `${fmt(x.pan_deg, 1)}° ${fmt(x.tilt_deg, 1)}° ${fmt(x.zoom_x, 1)}×</option>`).join('');
    sel.title = `설정된 프리셋 ${user.length}개 · 빈 슬롯과 벤더 명령 ${hidden}개는 숨김`;
  }
}

function ptzFillFromCurrent() {
  const q = S.ptz && S.ptz.position;
  if (!q) return;
  const n = ptzRaw() ? 6 : 2;
  $('ptz-pan').value = ptzShown('pan', q).toFixed(n);
  $('ptz-tilt').value = ptzShown('tilt', q).toFixed(n);
  $('ptz-zoom').value = ptzShown('zoom', q).toFixed(n);
}

function ptzApplyUnits() {
  const raw = ptzRaw();
  for (const name of ['pan', 'tilt', 'zoom']) {
    const a = ptzAxis(name), [lo, hi] = raw ? a.norm : a.human;
    const box = $('ptz-' + name);
    box.min = lo; box.max = hi; box.step = raw ? 0.001 : 0.1;
    $('ptz-u-' + name).textContent = raw ? '' : a.unit;
    box.title = `${lo} ~ ${hi}${raw ? '' : ' ' + a.unit}`;
  }
}

function ptzMsg(text, cls) {
  const el = $('ptz-msg');
  el.textContent = text || '';
  el.className = cls || 'dim';
}

// A move takes seconds and the 1.5 s status poll would show it as a jump, so
// poll the pose harder while the camera is on its way.  The camera needs a
// moment to start reporting MOVING, so the first IDLE reading cannot be taken
// as "already arrived" - hence a floor on how long the chase runs.
function ptzChase() {
  const now = Date.now();
  S.ptzChaseMin = now + 3000;
  S.ptzChaseUntil = now + 30000;   // hard stop, in case IDLE never comes back
  if (ptzChase._t) return;         // one chase at a time; the deadlines moved
  const tick = async () => {
    await refreshPtz(false);
    const t = Date.now();
    const moving = !!(S.ptz && S.ptz.position && S.ptz.position.moving);
    if (moving) S.ptzChaseMin = t + 1200;   // ... and a beat after it stops
    const done = (!moving && t > S.ptzChaseMin) || t > S.ptzChaseUntil;
    ptzChase._t = done ? null : setTimeout(tick, 600);
  };
  ptzChase._t = setTimeout(tick, 400);
}

async function ptzPost(url, body, btn) {
  if (S.ptzBusy) return null;
  S.ptzBusy = true;
  if (btn) btn.classList.add('busy');
  try {
    const res = await (await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })).json();
    if (!res.ok) { ptzMsg(res.reason || '실패', 'bad'); return null; }
    return res;
  } catch (e) {
    ptzMsg(String(e), 'bad');
    return null;
  } finally {
    S.ptzBusy = false;
    if (btn) btn.classList.remove('busy');
  }
}

async function ptzMove() {
  const num = (id) => { const v = parseFloat($(id).value); return isFinite(v) ? v : null; };
  const body = {
    pan: num('ptz-pan'), tilt: num('ptz-tilt'), zoom: num('ptz-zoom'),
    units: ptzRaw() ? 'norm' : 'deg',
    speed: parseFloat($('ptz-speed').value) || undefined,
  };
  if (body.pan === null && body.tilt === null && body.zoom === null) {
    ptzMsg('값을 입력하세요', 'bad');
    return;
  }
  const res = await ptzPost('/api/ptz/move', body, $('ptz-move'));
  if (!res) return;
  const t = res.target;
  ptzMsg(`이동 → ${fmt(t.pan_deg, 2)}° ${fmt(t.tilt_deg, 2)}° ${fmt(t.zoom_x, 2)}×`, 'ok');
  ptzChase();
}

async function ptzGoto() {
  const token = ($('ptz-preset-token').value || $('ptz-preset').value || '').trim();
  if (!token) { ptzMsg('프리셋 번호를 고르거나 입력하세요', 'bad'); return; }
  const res = await ptzPost('/api/ptz/preset', {
    token, speed: parseFloat($('ptz-speed').value) || undefined,
  }, $('ptz-goto'));
  if (!res) return;
  ptzMsg(`프리셋 #${res.preset.token} ${res.preset.name} 으로 이동`, 'ok');
  ptzChase();
}

// ---------------------------------------------------------------- wiring
$('btn-calib').onclick = () => calibrate({ freeze: false });
$('btn-freeze').onclick = () => calibrate({ freeze: true });
$('btn-live').onclick = () => setFrozen(false);
$('btn-clear').onclick = () => { S.points = []; S.selected = -1; renderPoints(); draw(); };
$('btn-copy').onclick = () => {
  const rows = ['idx,u_px,v_px,X_m,Y_m,Z_m,origin_range_m,camera_range_m,ground_range_m,roundtrip_px'];
  S.points.forEach((p, i) => {
    if (!p.valid) return;
    rows.push([i + 1, p.u.toFixed(2), p.v.toFixed(2), p.world[0].toFixed(4), p.world[1].toFixed(4),
      p.world[2].toFixed(4), (p.origin ?? 0).toFixed(3), (p.range ?? 0).toFixed(3),
      (p.ground ?? 0).toFixed(3), (p.roundtrip ?? 0).toFixed(3)].join(','));
  });
  navigator.clipboard.writeText(rows.join('\n')).then(
    () => { $('btn-copy').textContent = '복사됨'; setTimeout(() => ($('btn-copy').textContent = 'CSV 복사'), 1200); },
    () => alert(rows.join('\n')));
};

$('hd-add').onclick = () => addHoldout(parseFloat($('hd-u').value), parseFloat($('hd-v').value));
$('hd-v').onkeydown = $('hd-u').onkeydown = (ev) => { if (ev.key === 'Enter') $('hd-add').onclick(); };
$('hd-clear').onclick = () => {
  S.hold = []; S.holdSummary = null; S.holdSel = -1; renderHoldout(); draw();
};
$('hd-copy').onclick = () => {
  const rows = ['idx,target,auto,u_px,v_px,X_pred_m,Y_pred_m,X_gt_m,Y_gt_m,error_m,dx_m,dy_m,error_px'];
  S.hold.forEach((m, i) => {
    if (!m.valid) return;
    rows.push([i + 1, m.target, m.target_auto ? 1 : 0, m.u.toFixed(2), m.v.toFixed(2),
      m.world[0].toFixed(4), m.world[1].toFixed(4), m.world_gt[0].toFixed(4), m.world_gt[1].toFixed(4),
      m.error_m.toFixed(4), m.dx_m.toFixed(4), m.dy_m.toFixed(4), m.error_px.toFixed(2)].join(','));
  });
  navigator.clipboard.writeText(rows.join('\n')).then(
    () => { $('hd-copy').textContent = '복사됨'; setTimeout(() => ($('hd-copy').textContent = 'CSV 복사'), 1200); },
    () => alert(rows.join('\n')));
};
$('hd-pick').onchange = updateHint;

$('ptz-move').onclick = ptzMove;
$('ptz-goto').onclick = ptzGoto;
$('ptz-here').onclick = ptzFillFromCurrent;
$('ptz-stop').onclick = async () => {
  if (await ptzPost('/api/ptz/stop', {}, $('ptz-stop'))) { ptzMsg('정지 명령 전송', 'ok'); ptzChase(); }
};
$('ptz-raw').onchange = () => { ptzApplyUnits(); ptzFillFromCurrent(); };
$('ptz-preset').onchange = () => { $('ptz-preset-token').value = $('ptz-preset').value; };
for (const id of ['ptz-pan', 'ptz-tilt', 'ptz-zoom']) {
  $(id).onkeydown = (ev) => { if (ev.key === 'Enter') ptzMove(); };
}
$('ptz-preset-token').onkeydown = (ev) => { if (ev.key === 'Enter') ptzGoto(); };

overlayC.addEventListener('click', addPoint);
overlayC.addEventListener('mousemove', onMove);
overlayC.addEventListener('mouseleave', () => { loupe.style.display = 'none'; });

for (const id of ['show-survey', 'show-horizon', 'show-kpts', 'show-grid']) $(id).onchange = draw;

$('plane-z').onchange = async () => {
  if (!S.calib) return;
  await loadOverlay();
  fillHoldoutTargets();
  await resolveAllPoints();
  draw();
};

$('auto').onchange = () => {
  clearInterval(S.autoTimer);
  S.autoTimer = null;
  if ($('auto').checked) {
    const sec = Math.max(2, parseInt($('auto-sec').value, 10) || 5);
    S.autoTimer = setInterval(() => calibrate({ freeze: false }), sec * 1000);
    calibrate({ freeze: false });
  }
};
$('auto-move').onchange = updateHint;
$('auto-sec').onchange = () => { if ($('auto').checked) $('auto').onchange(); };

window.addEventListener('resize', draw);
new ResizeObserver(draw).observe(stage);

refreshStatus();
setInterval(refreshStatus, 1500);
setFrozen(false);
renderHoldout();

// ?calib=1 runs one calibration on load (kiosk / screenshot use);
// ?auto=<sec> starts the periodic re-calibration straight away.
const qs = new URLSearchParams(location.search);
if (qs.has('auto')) {
  $('auto-sec').value = Math.max(2, parseInt(qs.get('auto'), 10) || 5);
  $('auto').checked = true;
  $('auto').onchange();
} else if (qs.get('calib') === '1') {
  calibrate({ freeze: qs.get('freeze') === '1' });
}
