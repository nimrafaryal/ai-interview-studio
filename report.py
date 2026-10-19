"""
report.py  -  Builds a self-contained HTML report (open in any browser, print to PDF).
"""

import base64
import html
import os


def _e(x):
    return html.escape(str(x or ""))


def _thumb(path, max_w=220):
    if not path or not os.path.exists(path):
        return ""
    try:
        import cv2
        img = cv2.imread(path)
        h, w = img.shape[:2]
        s = max_w / w
        img = cv2.resize(img, (max_w, int(h * s)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except Exception:
        pass
    return ""


def score_color(pct):
    return "#16a34a" if pct >= 70 else "#d97706" if pct >= 45 else "#dc2626"


def build_html(meta, results, events, summary):
    overall = meta["overall"]
    integ = meta["integrity"]
    attn = meta["attention"]

    def ring(val, label):
        c = score_color(val)
        return (f"<div class='ring'><div class='num' style='color:{c}'>{val:.0f}</div>"
                f"<div class='lbl'>{_e(label)}</div></div>")

    qhtml = []
    for i, r in enumerate(results, 1):
        pct = r.get("score", 0) * 10
        li = lambda xs: "".join(f"<li>{_e(x)}</li>" for x in xs) or "<li>-</li>"
        qhtml.append(f"""
        <div class='card'>
          <div class='qhead'><span class='qn'>Q{i}</span>
            <span class='qt'>{_e(r['question'])}</span>
            <span class='pill' style='background:{score_color(pct)}'>{r.get('score',0)}/10</span></div>
          <div class='bar'><div style='width:{pct}%;background:{score_color(pct)}'></div></div>
          <p class='lbl2'>Candidate's answer</p>
          <div class='ans'>{_e(r.get('clean_answer') or r.get('answer')) or '<i>No answer was captured.</i>'}</div>
          {f"<p class='raw'><b>Exact transcript:</b> {_e(r.get('answer'))}</p>" if r.get('clean_answer') and r.get('answer') else ''}
          {f"<p class='warn'>{_e(r.get('error'))}</p>" if r.get('error') else ''}
          <p><b>Feedback:</b> {_e(r.get('feedback'))}</p>
          <div class='cols'><div><b>Strengths</b><ul>{li(r.get('strengths', []))}</ul></div>
          <div><b>To improve</b><ul>{li(r.get('improvements', []))}</ul></div></div>
          {f"<p class='model'><b>Example strong answer:</b> {_e(r.get('model_answer'))}</p>" if r.get('model_answer') else ''}
        </div>""")

    rows = []
    for e in events:
        src = _thumb(e.get("snapshot"))
        img = f"<img src='{src}'>" if src else ""
        rows.append(f"<tr><td>{_e(e['time'])}</td><td>{_e(e['event'])}</td>"
                    f"<td>-{e['points']}</td><td>{e['integrity']}</td><td>{img}</td></tr>")
    ev_rows = "".join(rows) or "<tr><td colspan=5>No integrity events. Clean session.</td></tr>"

    st_list = "".join(f"<li>{_e(x)}</li>" for x in summary.get("top_strengths", []))
    fa_list = "".join(f"<li>{_e(x)}</li>" for x in summary.get("focus_areas", []))

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Interview report - {_e(meta['name'])}</title>
<style>
body{{font-family:Segoe UI,Inter,Arial,sans-serif;background:#f5f6fa;color:#1f2330;margin:0;padding:32px}}
.wrap{{max-width:900px;margin:auto}}
h1{{margin:0;font-size:26px}} .sub{{color:#666;margin:4px 0 24px}}
.rings{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}}
.ring{{background:#fff;border-radius:14px;padding:16px 22px;min-width:140px;box-shadow:0 1px 3px #0001}}
.num{{font-size:34px;font-weight:800}} .lbl{{color:#777;font-size:13px}}
.card{{background:#fff;border-radius:14px;padding:18px 22px;margin-bottom:14px;box-shadow:0 1px 3px #0001}}
.qhead{{display:flex;gap:10px;align-items:flex-start}} .qn{{font-weight:800;color:#6d4aff}}
.qt{{flex:1;font-weight:600}} .pill{{color:#fff;border-radius:999px;padding:2px 10px;font-weight:700;font-size:13px}}
.bar{{height:6px;background:#eee;border-radius:9px;margin:10px 0}} .bar div{{height:6px;border-radius:9px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} ul{{margin:4px 0 0 18px;padding:0}}
.warn{{color:#b45309}} .lbl2{{font-size:12px;font-weight:700;color:#6d4aff;text-transform:uppercase;letter-spacing:.05em;margin:12px 0 4px}} .ans{{border-left:3px solid #6d4aff;background:#f7f5ff;border-radius:8px;padding:10px 14px;line-height:1.6}} .raw{{color:#777;font-size:13px}} .model{{background:#f3f0ff;border-radius:10px;padding:10px}}
.rec{{display:inline-block;background:#6d4aff;color:#fff;border-radius:999px;padding:4px 14px;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td,th{{text-align:left;padding:6px;border-bottom:1px solid #eee}}
td img{{width:120px;border-radius:6px}}
@media print{{body{{background:#fff;padding:0}} .card,.ring{{box-shadow:none;border:1px solid #ddd}}}}
</style></head><body><div class='wrap'>
<h1>Interview report: {_e(meta['name'])}</h1>
<div class='sub'>{_e(meta['role'])} · {_e(meta['date'])} · {len(results)} questions</div>
<div class='rings'>{ring(overall,'Answer score /100')}{ring(integ,'Integrity /100')}{ring(attn,'Attention %')}</div>
<div class='card'><span class='rec'>{_e(summary.get('recommendation','-'))}</span>
<p>{_e(summary.get('summary',''))}</p>
<div class='cols'><div><b>Top strengths</b><ul>{st_list or '<li>-</li>'}</ul></div>
<div><b>Focus areas</b><ul>{fa_list or '<li>-</li>'}</ul></div></div></div>
<h2>Answers</h2>{''.join(qhtml)}
<h2>Proctoring log</h2><div class='card'><table><tr><th>Time</th><th>Event</th><th>Penalty</th><th>Integrity</th><th>Snapshot</th></tr>{ev_rows}</table></div>
<p class='sub'>Generated by AI Interview Studio. AI scores are guidance for a human reviewer, not a final decision.</p>
</div></body></html>"""