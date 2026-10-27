#!/usr/bin/env python3
"""One-page pgfplots preview of candidate figures for the CAIN paper.

Not wired into main.tex: this renders the figure candidates side by side so a
human can pick which, if any, earn their page. Build it with

    tectonic paper/fig-candidates/figure-candidates.tex

tectonic, not pdflatex, because only tectonic is installed here. Do not add
`helvet`: under tectonic's unicode engine TU/phv/b/n is undefined, bold
silently degrades to regular with only a font warning, and every heading on the
sheet goes flat.

Long LaTeX blocks are plain raw strings; f-strings are used only on short
interpolated lines, to keep LaTeX braces away from format parsing.

Figure B asserts its own correctness: the six band boundaries are *derived*
from the published (tokens, recall) pairs and checked against the six numbers
printed in main.tex, so the curve crossings and the band strip cannot drift
apart, nor drift from the paper. See check_bands().
"""
import csv, json, math, statistics, pathlib

REPO   = pathlib.Path(__file__).resolve().parents[2]
OUTDIR = REPO / 'paper' / 'fig-candidates'
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / 'figure-candidates.tex'

# --- Fig A/B: frontier_bands.py PUBLISHED control output ---------------------
FRONTIER = [  # (name, tokens, recall%, on_frontier)
    ('zero context',       0,  0.0, True),  ('prompts-only',   202, 31.5, True),
    ('prose @2k',        478, 37.9, False), ('rollup',         929, 55.6, True),
    ('prose @8k',       1861, 76.6, True),  ('answer-only',   2533, 84.7, True),
    ('prose-only',      3055, 88.7, True),  ('dialogue record',3928, 74.2, False),
    ('call-only',       6329, 46.3, False), ('action record',  6393, 95.2, True),
    ('output-stripped', 8880, 90.3, False), ('raw',          137230, 94.4, False),
]
# the seven winners, in the order they win, with the colour each keeps in both B panels
WINNERS = [('zero',             0,  0.0, 'cgray'),
           ('prompts-only',   202, 31.5, 'csky'),
           ('rollup',         929, 55.6, 'cblue'),
           ('prose @8k',     1861, 76.6, 'cgreen'),
           ('answer-only',   2533, 84.7, 'cpink'),
           ('prose-only',    3055, 88.7, 'corange'),
           ('action record', 6393, 95.2, 'cverm')]
PUBLISHED_BOUNDS = [641, 3017, 4438, 8296, 13050, 51354]   # as printed in main.tex
F_LO, F_HI = 100, 200_000

def total(tokens, recall, F):
    """The frontier objective: read cost now, plus the expected cost of a miss."""
    return tokens + (1.0 - recall / 100.0) * F

def check_bands():
    """Derive each boundary from the pair that crosses there, and refuse to
    emit a figure that disagrees with the number printed in the paper."""
    derived = []
    for (_, t1, r1, _), (_, t2, r2, _) in zip(WINNERS, WINNERS[1:]):
        derived.append((t2 - t1) / ((1 - r1 / 100) - (1 - r2 / 100)))
    for pub, got in zip(PUBLISHED_BOUNDS, derived):
        assert abs(pub - got) < 1.0, f"band boundary drift: paper {pub}, derived {got:.1f}"
    print("fig B control PASS: boundaries %s derive from the published pairs"
          % ", ".join(f"{d:.0f}" for d in derived))
    return derived

BOUNDS = check_bands()
BANDS  = list(zip([w[0] for w in WINNERS],
                  [F_LO] + BOUNDS,
                  BOUNDS + [F_HI],
                  [w[3] for w in WINNERS]))

# --- Fig D: Table 5 (tab:retention), cheapest read first ---------------------
RETENTION = [  # (budget, artifact, recall, lo, hi)
    ('cap', 'rollup (pointer)',       0.0,  0.0,  4.1),
    ('cap', 'rollup body',           31.5, 22.8, 41.7),
    ('cap', 'output-stripped chunk', 34.8, 25.7, 45.2),
    ('all', 'rollup',                55.6, 46.9, 64.1),
    ('all', 'dialogue record',       74.2, 65.8, 81.1),
    ('all', 'action record',         95.2, 89.8, 97.8),
    ('all', 'call-only transcript',  46.3, 37.8, 55.1),
    ('all', 'output-stripped',       90.3, 83.8, 94.4),
    ('all', 'raw transcript',        94.4, 87.5, 97.6),
]

# --- fig:retention, the whole-artifact arms that replaced tab:retention ------
# This figure REPLACES the table rather than supplementing it, which is why it
# carries wrong-of-attempted as a printed column instead of only the recall
# bars. The earlier read on candidate D -- "you pay for the figure and keep the
# table" -- was correct about a bars-only figure and is what the annotation
# columns answer. Five of these numbers (every wrong rate except the rollup's
# 10.3 and output-stripped's 1.8, plus the dialogue record's 71.8 recall) appear
# nowhere else in the paper, so dropping a column here deletes a result.
#
# The capped-budget arms of RETENTION above are deliberately absent: their
# discussion is Section VI-E, which already states all three recalls with their
# intervals in prose, and they run on a different n with a different denominator
# for wrongness, which is what forced the two incompatible notations the table
# needed three caption lines to explain.
#
# `cls` drives colour and is the paper's content-class claim, not decoration:
#   keep  -- retains assistant prose verbatim
#   gen   -- prose replaced by generated summary
#   drop  -- prose deleted, tool input kept (the inversion)
# `flag` marks an arm whose row needs a caption note: 'n' for a reduced n,
# 'x' for exploratory (designed with the questions visible).
# Computed from the graded records, never typed: after the 2026-10-09 regrade the
# literal rows that stood here disagreed with the data, and the gate
# (check_paper_invariants.check_retention_figure) reads the same source.
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arm_rates as _ar

# Ordered by ROLE, not by recall (2026-10-14 review): the recommended and planned
# arms read first and the exploratory action record sits last, below a rule, so
# the daggered bar no longer reads as the winner. Listed BOTTOM to TOP, because
# row 0 is drawn at the bottom of the axis; top to bottom the figure reads
# output-stripped, prose-only, raw, dialogue record, rollup, call-only, then the
# action record.
#
# The no-digest action record was dropped from this figure on purpose. Its only
# bar was the design-set 93.5%, the least trustworthy of its three measurements
# because the record was designed with those questions visible; all three
# (design, held-out, Replication 1) are now in tab:nodigest in main.tex.
_WHOLE = [  # (arm_rates label, figure label, content class, flag), bottom to top
    ('action record', 'action record', 'keep', 'x'),
    ('call-only', 'call-only', 'drop', 'n'),
    ('rollup', 'rollup', 'gen', ''),
    ('dialogue record', 'dialogue record', 'keep', ''),
    ('raw', 'raw transcript', 'keep', 'n'),
    ('prose-only', 'prose-only', 'keep', ''),
    ('output-stripped', 'output-stripped', 'keep', ''),
]
# the explicit order the control asserts, bottom to top
RETENTION_ORDER = [fig for _lab, fig, _c, _f in _WHOLE]
# arms of RETENTION['all'] that may be absent from the figure, and why
RETENTION_DROPPED = {'action record no digest': 'its three measurements are in tab:nodigest'}
RETENTION_WHOLE = [  # (artifact, recall, lo, hi, wrong_of_att, n, cls, flag)
    (fig, _r.recall, _r.recall_ci[0], _r.recall_ci[1], _r.wrong_rate, _r.n, cls, flag)
    for lab, fig, cls, flag in _WHOLE for _r in [_ar.any_rate(lab)]]
_CLS_COLOUR = {'keep': 'cgreen', 'gen': 'cverm', 'drop': 'corange'}


def _retention_control():
    """Refuse to emit a figure whose rows are out of order or whose point
    estimate sits outside its own interval. The figure is ordered by role, not
    by recall (caption: "ordered by role, not recall"), so the control asserts
    that explicit order rather than an ascending sort."""
    order = [r[0] for r in RETENTION_WHOLE]
    assert order == RETENTION_ORDER, f"rows out of the role order: {order}"
    assert order[0] == 'action record', "the exploratory action record must sit last (bottom)"
    for name, r, lo, hi, w, n, _c, _f in RETENTION_WHOLE:
        assert lo <= r <= hi, f"{name}: {r} outside [{lo}, {hi}]"
        assert 0 <= w <= 100 and 0 < n <= 124, f"{name}: bad wrong/n"
    # every whole-artifact arm in RETENTION must survive into the figure
    old = {a.replace(' transcript', '') for b, a, *_ in RETENTION if b == 'all'}
    new = {a.replace(' transcript', '') for a, *_ in RETENTION_WHOLE}
    missing = old - new - set(RETENTION_DROPPED)
    assert not missing, f"arms dropped from the figure: {missing}"
    assert not set(RETENTION_DROPPED) & new, "a dropped arm is back in the figure"
    print("fig retention control PASS: %d arms in role order, intervals contain "
          "their estimates, no arm of RETENTION['all'] dropped except %s"
          % (len(RETENTION_WHOLE), sorted(RETENTION_DROPPED)))


_retention_control()


def fig_retention_body(axis_width, axis_height='1.62in'):
    """The fig:retention tikzpicture and nothing else. Dots are point
    estimates, bars Wilson 95% intervals, and the two right-hand columns carry
    recall and wrong-of-attempted so this float replaces tab:retention instead
    of sitting beside it. Every coordinate derives from RETENTION_WHOLE."""
    rows = RETENTION_WHOLE
    top = len(rows) - 1
    labels = ",".join(
        n + ('$^{\\dagger}$' if f == 'x' else '$^{*}$' if f == 'n' else '')
        for n, *_r, f in rows)
    L = ["\\begin{tikzpicture}",
         "\\begin{axis}[width=%s,height=%s,scale only axis," % (axis_width, axis_height),
         "  xmin=38,xmax=102,ymin=-0.7,ymax=%.2f," % (top + 0.7),
         "  xlabel={Recoverable (\\%)},",
         "  xlabel style={font=\\tiny,yshift=2pt},",
         "  ytick={%s}," % ",".join(str(i) for i in range(len(rows))),
         "  yticklabels={%s}," % labels,
         "  y tick label style={font=\\tiny},",
         "  x tick label style={font=\\tiny},",
         "  xtick={40,60,80,100},scaled x ticks=false,xmajorgrids,",
         "  axis y line*=left,axis x line*=bottom,",
         "  clip mode=individual]",
         "\\addplot[draw=none,forget plot] coordinates {(40,0) (100,%d)};" % top]
    for i, (name, rec, lo, hi, wrong, n, cls, flag) in enumerate(rows):
        L.append("%% %s: %.1f [%.1f, %.1f], wrong %.1f, n=%d" % (name, rec, lo, hi, wrong, n))
        L.append("\\draw[cink,line width=0.4pt] (axis cs:%.6g,%d) -- (axis cs:%.6g,%d);"
                 % (lo, i, hi, i))
        for e in (lo, hi):
            L.append("\\draw[cink,line width=0.4pt] (axis cs:%.6g,%.2f) -- (axis cs:%.6g,%.2f);"
                     % (e, i - 0.18, e, i + 0.18))
        L.append("\\fill[%s] (axis cs:%.6g,%d) circle (1.5pt);" % (_CLS_COLOUR[cls], rec, i))
    # a rule between the exploratory action record (row 0) and the rest
    L.append("\\draw[cink!50,densely dashed,line width=0.3pt] (axis cs:38,0.5) -- (axis cs:102,0.5);")
    # right-hand columns: recall, then wrong-of-attempted
    L.append("\\node[font=\\tiny,anchor=east] at (axis cs:117,%.2f) {\\%%};" % (top + 0.55))
    L.append("\\node[font=\\tiny,anchor=east] at (axis cs:135,%.2f) {wrong};" % (top + 0.55))
    for i, (_n, rec, _lo, _hi, wrong, *_r) in enumerate(rows):
        L.append("\\node[font=\\tiny,anchor=east] at (axis cs:117,%d) {%.1f};" % (i, rec))
        L.append("\\node[font=\\tiny,anchor=east] at (axis cs:135,%d) {%.1f};" % (i, wrong))
    L += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(L)
# --- Fig C: RQ2 "89.5% of compression is deletion" --------------------------
# Read from the measurement rather than typed. These were hardcoded as 73.7 and
# 15.8 until 2026-10-02, which drifted from the measured 73.53 and 16.01 and,
# because the pair was chosen to sum to exactly the 89.5% headline, made the
# figure's two components a back-fit to its own bracket label.
_SL = json.loads((REPO / 'data' / 'paper-rollups' / 'rq3_structural_loss.json').read_text())
_AGG = _SL['aggregate']
SH_TOTAL_TOKENS = _AGG['total_transcript_tokens']
SH_RESULTS  = round(_AGG['tool_result_share'] * 100, 2)   # % of all transcript tokens
SH_CALLS    = round(_AGG['tool_use_share'] * 100, 2)
SH_RETAINED = round(100.0 - SH_RESULTS - SH_CALLS, 2)
FILTER      = 100.0 / SH_RETAINED           # x, the deletion step alone
SUMMARISER  = round(_AGG['fair_compression_ratio'], 1)  # x, against retained material
FINAL       = SH_RETAINED / SUMMARISER      # rollup as % of transcript tokens
print("fig C: deletion %.1fx, summariser %.1fx, rollup %.2f%% of tokens"
      % (FILTER, SUMMARISER, FINAL))

def co(pts): return " ".join("({:.6g},{:.6g})".format(a, b) for a, b in pts)


def fig_c_body(axis_width):
    """The fig C tikzpicture and nothing else, so the preview page and
    paper/main.tex render the same bar from one source. Every share and every
    bracket end is computed from SH_RESULTS and SH_CALLS rather than typed, so
    the label cannot drift from the geometry it annotates. axis_width differs
    between the two callers only because an IEEEtran column is narrower than
    half of the preview page."""
    deleted = SH_RESULTS + SH_CALLS
    b = ["\\begin{tikzpicture}",
         "\\begin{axis}[width=%s,height=1.3in,scale only axis," % axis_width,
         "  xmin=-1,xmax=103,ymin=-0.52,ymax=2.05,",
         "  xlabel={Transcript composition (\\% of tokens)},",
         "  ytick={0,1},yticklabels={after summariser,as persisted},",
         "  scaled x ticks=false,",
         "  y tick label style={font=\\tiny},xmajorgrids,xtick={0,25,50,75,100},",
         "  clip mode=individual]",
         "\\addplot[draw=none,forget plot] coordinates {(0,0) (100,1)};"]
    # Label ink is per segment, not uniform, because contrast against these two
    # fills is not comparable: white on corange #E69F00 is 2.25:1, where cink
    # #222222 on it is 7.06:1. On cverm #D55E00 the two are a wash (3.87 vs
    # 4.11) and white reads better on the darker fill, so only the orange flips.
    for x0, wd, col, ink, name, inner in [
            (0.0, SH_RESULTS, 'cverm', 'white', 'tool results', '%.2f\\%%' % SH_RESULTS),
            (SH_RESULTS, SH_CALLS, 'corange', 'cink', 'tool calls', '%.2f\\%%' % SH_CALLS),
            (deleted, SH_RETAINED, 'cgreen', 'white', '', '')]:
        b.append("\\fill[%s] (axis cs:%.2f,0.70) rectangle (axis cs:%.2f,1.30);"
                 % (col, x0 + 0.25, x0 + wd - 0.25))
        if inner:
            b.append("\\node[font=\\tiny,color=%s,align=center,inner sep=0pt] at (axis cs:%.2f,1.0) "
                     "{%s\\\\{\\bfseries %s}};" % (ink, x0 + wd / 2, name, inner))
    # step 1: the deletion, named for what makes it worth a figure -- it is free
    b.append("\\draw[cink!60,line width=0.45pt,|-|] (axis cs:0.25,1.48) -- (axis cs:%.1f,1.48);"
             % deleted)
    b.append("\\node[font=\\tiny,color=cink,anchor=south] at (axis cs:%.2f,1.56) "
             "{free, mechanical deletion: %.1f\\%%};" % (deleted / 2 + 0.25, deleted))
    # step 2: the model call, which is the only part anyone pays for
    b.append("\\draw[cgreen,line width=0.45pt] (axis cs:%.2f,0.70) -- (axis cs:97.0,0.62);"
             % (deleted + SH_RETAINED / 2))
    b.append("\\node[font=\\tiny,color=cgreen,anchor=north east] at (axis cs:99,0.64) "
             "{summariser shrinks %.2f\\%% to %.2f\\%%};" % (SH_RETAINED, FINAL))
    b.append("\\fill[cgreen] (axis cs:0.25,-0.30) rectangle (axis cs:%.3f,0.30);"
             % max(FINAL, 0.45))
    b.append("\\node[font=\\tiny,color=cink,anchor=west,xshift=4pt] at (axis cs:%.3f,-0.04) "
             "{the rollup};" % max(FINAL, 0.45))
    b += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(b)


L = []; w = L.append

w(r"""\documentclass[10pt]{article}
\usepackage[margin=0.5in,letterpaper]{geometry}
\usepackage{tikz,pgfplots,amsmath,amssymb,lmodern}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\pagestyle{empty}\setlength{\parindent}{0pt}
\definecolor{cblue}{HTML}{0072B2}\definecolor{cverm}{HTML}{D55E00}
\definecolor{cgreen}{HTML}{009E73}\definecolor{corange}{HTML}{E69F00}
\definecolor{csky}{HTML}{56B4E9}\definecolor{cgray}{HTML}{999999}
\definecolor{cink}{HTML}{222222}\definecolor{cpink}{HTML}{CC79A7}
\pgfplotsset{every axis/.append style={
    font=\scriptsize, axis line style={cink!55,line width=0.4pt},
    tick label style={cink!75}, label style={cink},
    major grid style={cink!12,line width=0.3pt},
    minor grid style={cink!7,line width=0.3pt},
    tick align=outside, tickpos=left}}
\newcommand{\figttl}[2]{{\footnotesize\bfseries #1}\\[-0.15em]{\scriptsize\color{cink!70} #2}\\[0.45em]}
\newcommand{\figcap}[1]{\\[0.25em]{\scriptsize\color{cink!70} #1}}
\begin{document}
{\large\bfseries Candidate figures --- CAIN paper}\hfill
{\scriptsize\color{cink!70} preview only, not in \texttt{main.tex}. Okabe--Ito palette, as \texttt{make\_figures.py}.}\\[0.7em]
""")

# ===================== FIG A: Pareto frontier ===============================
front = sorted((t, r) for n, t, r, f in FRONTIER if f and t <= 9200)
dom   = [(t, r) for n, t, r, f in FRONTIER if not f and t <= 9200]
w(r"""\begin{minipage}[t]{0.49\textwidth}
\figttl{A. The failure-cost frontier}{replaces the uncaptioned table in \S7.2}
\begin{tikzpicture}
\begin{axis}[width=2.95in,height=2.15in,scale only axis,
  xlabel={Read cost (tokens)},ylabel={Recall (recoverable, \%)},
  xmin=-2700,xmax=10400,ymin=-9,ymax=108,grid=both,
  xtick={0,2000,4000,6000,8000},xticklabels={0,2k,4k,6k,8k},
  scaled x ticks=false,
  ytick={0,25,50,75,100},
  legend style={font=\tiny,at={(0.995,0.025)},anchor=south east,draw=cink!30,
                fill=white,fill opacity=0.92,text opacity=1},
  clip mode=individual]""")
w(r"\addplot[cblue,line width=1.1pt,mark=*,mark size=2.2pt] coordinates {" + co(front) + r"};")
N_FRONT = sum(1 for *_, f in FRONTIER if f)
N_DOM   = sum(1 for *_, f in FRONTIER if not f)
w(r"\addlegendentry{on frontier (%d)}" % N_FRONT)
w(r"\addplot[only marks,mark=o,mark size=2.2pt,cverm,line width=0.8pt] coordinates {" + co(dom) + r"};")
w(r"\addlegendentry{dominated (%d)}" % N_DOM)
# prose @2k's gap to the frontier edge as the figure draws it: the polyline
# through `front`, interpolated linearly at prose @2k's own token cost.
_p2k = next((t, r) for n, t, r, f in FRONTIER if n == 'prose @2k')
(_t0, _r0), (_t1, _r1) = next((a, b) for a, b in zip(front, front[1:]) if a[0] <= _p2k[0] <= b[0])
_edge = _r0 + (_p2k[0] - _t0) / (_t1 - _t0) * (_r1 - _r0)
P2K_GAP = _edge - _p2k[1]
print("fig A prose@2k gap: edge between (%d,%.1f) and (%d,%.1f) at %d tok = %.2f; minus %.1f = %.2f pt"
      % (_t0, _r0, _t1, _r1, _p2k[0], _edge, _p2k[1], P2K_GAP))
w(r"""\draw[cverm,line width=0.4pt] (axis cs:580,36.0) -- (axis cs:2150,22.0);
\node[font=\tiny,color=cverm,anchor=west,align=left] at (axis cs:2200,21.0)
  {""" + "%.1f" % P2K_GAP + r""" pt under the frontier edge ---\\dominated, but by less than\\its own interval};""")
LAB = {'zero context':('west',6,-2),      'prompts-only':('east',-6,0),
       'prose @2k':('east',-5,1),         'rollup':('north west',4,-2),
       'prose @8k':('east',-5,1),         'answer-only':('west',6,0),
       'prose-only':('east',-5,2),        'dialogue record':('west',6,0),
       'call-only':('north west',4,-2),   'action record':('south',0,5),
       'output-stripped':('north',0,-5)}
for n, t, r, f in FRONTIER:
    if t > 9200: continue
    a, dx, dy = LAB[n]
    w("\\node[font=\\tiny,color=cink,anchor=%s,xshift=%dpt,yshift=%dpt] at (axis cs:%d,%s) {%s};"
      % (a, dx, dy, t, r, n))
_raw = next((t, r, f) for n, t, r, f in FRONTIER if n == 'raw')
assert not _raw[2]
w(r"""\node[font=\tiny,color=cverm,anchor=east,align=right] at (axis cs:10300,62)
  {raw: """ + "{:,}".format(_raw[0]).replace(",", "{,}") + r" tok,\\" + "%.1f" % _raw[1]
  + r"""\% --- off scale,\\also dominated};
\end{axis}
\end{tikzpicture}
\figcap{Linear token axis on purpose: the hull for $\text{read}+(1-\text{recall})F$ is
convex in \emph{linear} space, so the frontier's chords stay straight and the prose-2k
gap is checkable by eye. A log axis would bend them.}
\end{minipage}\hfill""")

# ===================== FIG D: dot-and-whisker ===============================
rows   = RETENTION[::-1]
ylabel = ",".join(a for _, a, _, _, _ in rows)
w(r"""\begin{minipage}[t]{0.49\textwidth}
\figttl{D. Retention with Wilson intervals}{renders the ``indistinguishable, not ranked'' argument}
\begin{tikzpicture}
\begin{axis}[width=2.45in,height=2.15in,scale only axis,
  xlabel={Recoverable (\%), Wilson 95\%},
  xmin=-5,xmax=116,ymin=-0.8,ymax=9.5,
  ytick={0,1,2,3,4,5,6,7,8},yticklabels={""" + ylabel + r"""},
  xmajorgrids,xtick={0,25,50,75,100},
  y tick label style={font=\tiny},clip mode=individual]
\addplot[draw=none,forget plot] coordinates {(0,0) (100,8)};
\fill[cverm,fill opacity=0.13] (axis cs:47.7,-0.8) rectangle (axis cs:55.9,9.5);
\node[font=\tiny,color=cverm,anchor=south,align=center] at (axis cs:51.8,8.45)
  {rollup $\cap$ call-only};""")
for i, (budget, art, rec, lo, hi) in enumerate(rows):
    col = 'cblue' if budget == 'all' else 'corange'
    w("\\draw[%s,line width=0.9pt] (axis cs:%s,%d) -- (axis cs:%s,%d);" % (col, lo, i, hi, i))
    w("\\draw[%s,line width=0.9pt] (axis cs:%s,%.2f) -- (axis cs:%s,%.2f);" % (col, lo, i-0.17, lo, i+0.17))
    w("\\draw[%s,line width=0.9pt] (axis cs:%s,%.2f) -- (axis cs:%s,%.2f);" % (col, hi, i-0.17, hi, i+0.17))
    w("\\node[circle,fill=%s,draw=white,line width=0.4pt,inner sep=1.5pt] at (axis cs:%s,%d) {};" % (col, rec, i))
    w("\\node[font=\\tiny,color=cink,anchor=west,xshift=4pt] at (axis cs:%s,%d) {%.1f};" % (hi, i, rec))
w(r"""\end{axis}
\end{tikzpicture}
\figcap{\textcolor{cblue}{$\bullet$} whole artifact \quad \textcolor{corange}{$\bullet$} capped
read budget. The shaded column is [47.7, 55.9], where the rollup and call-only intervals
overlap --- the separation the paper declines to read as an ordering.}
\end{minipage}\\[1.1em]""")

# ===================== FIG B: cost curves + winner bands ====================
# Top panel shows the question (what each substrate costs in total, as a miss gets
# more expensive); bottom panel is just the projection of which line is lowest.
FS = [F_LO * (F_HI / F_LO) ** (i / 120.0) for i in range(121)]
w(r"""\figttl{B. Which substrate is cheapest, and why}{the decision guide --- currently a sentence of six parenthesised numbers}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 2,vertical sep=2.5mm,
    x descriptions at=edge bottom},width=6.5in,scale only axis,
  xmode=log,xmin=100,xmax=200000,
  xtick={""" + ",".join(str(x) for x in [F_LO] + PUBLISHED_BOUNDS + [F_HI]) + r"""},
  xticklabels={""" + ",".join(["100"] + ["{:,}".format(b).replace(",", "{,}") for b in PUBLISHED_BOUNDS] + ["200k"]) + r"""},
  x tick label style={font=\tiny},clip mode=individual]

\nextgroupplot[height=1.15in,ymode=log,ymin=70,ymax=600000,
  ylabel={Total cost (tokens)},ytick={100,1000,10000,100000},
  yticklabels={100,1k,10k,100k},ymajorgrids,
  y label style={font=\tiny},]""")
# each substrate's total-cost line, in the colour it keeps in the strip below
for name, tok, rec, col in WINNERS:
    pts = [(F, total(tok, rec, F)) for F in FS]
    w("\\addplot[%s,line width=0.7pt,forget plot] coordinates {%s};" % (col, co(pts)))
# the lower envelope: the line you would actually pay, picked out in dark ink
env = [(F, min(total(t, r, F) for _, t, r, _ in WINNERS)) for F in FS]
w("\\addplot[cink,line width=1.5pt,opacity=0.75,forget plot] coordinates {%s};" % co(env))
for F in BOUNDS:
    w("\\draw[cink!35,densely dotted,line width=0.4pt] (axis cs:%.0f,70) -- (axis cs:%.0f,600000);" % (F, F))
w(r"""\node[font=\tiny,color=cink,anchor=north west,align=left,fill=white,
      fill opacity=0.85,text opacity=1,inner sep=1.5pt] at (axis cs:115,500000)
  {Each thin line is one substrate's total bill: what you pay to read it,\\
   plus what a miss costs you. Every line's colour is its band below.\\
   The dark line is the cheapest available --- and the substrate it\\
   belongs to changes as you move right.};

\nextgroupplot[height=0.55in,ymin=0,ymax=1,ytick=\empty,axis y line=none,
  xlabel={$F$ --- what one failed recall costs you, in tokens burned re-deriving it}]
\addplot[draw=none,forget plot] coordinates {(100,0) (200000,1)};""")
for name, lo, hi, col in BANDS:
    w("\\fill[%s] (axis cs:%.0f,0) rectangle (axis cs:%.0f,1);" % (col, lo, hi))
    # a band too narrow on the log axis for its horizontal label gets it rotated
    rot = ",rotate=90" if math.log10(hi / lo) < 0.25 else ""
    w("\\node[font=\\tiny\\bfseries,color=white%s] at (axis cs:%.1f,0.5) {%s};"
      % (rot, math.sqrt(lo * hi), name))
w(r"""\end{groupplot}
\end{tikzpicture}
\figcap{\textbf{How to read it:} find what a miss costs you on the bottom axis; the band
names the substrate to store. A cheap miss (a throwaway exchange) means storing nothing
wins outright; an expensive one (re-deriving a debugging session's reasoning) is worth
paying to read the richest artifact you have. The crossings in the top panel \emph{are}
the band edges --- derived from the same cost and recall figures, not fitted, and the
script asserts they match the six boundaries printed in \S7.2.}\\[0.9em]""")

# ===================== FIG C: deletion waterfall ============================
# Shared with paper/main.tex through fig_c_body(), written out below as
# paper/fig/fig-compression.tex. Do not inline a second copy here.
w(r"""\begin{minipage}[t]{0.49\textwidth}
\figttl{C. 89.5\% of compression is deletion}{landed as fig:deletion in main.tex, from one shared source}""")
w(fig_c_body('2.75in'))
w(r"""\figcap{\textcolor{cverm}{$\blacksquare$} tool results \quad \textcolor{corange}{$\blacksquare$} tool
invocations \quad \textcolor{cgreen}{$\blacksquare$} kept. Ending at 0.70\% is 141.6$\times$
over the whole corpus added together as one ratio. The paper's 32.0$\times$ headline is the
\emph{typical single session} (a median of per-session ratios), a different statistic, so
the two are not meant to agree.}
\end{minipage}\hfill""")

# ===================== the decision, not a figure ===========================
w(r"""\begin{minipage}[t]{0.47\textwidth}
\figttl{Where each one belongs}{the paper sits at exactly 10 of 10 content pages}
{\scriptsize
\textbf{A --- belongs here, not in a spinoff.} The action-record spinoff plan assigns CAIN
``the cost frontier'' and ``the substrate decision guide'' by name. A also uses only read
cost and recall, with no $F$ in it, so it carries none of B's grounding risk.\\[0.45em]
\textbf{B --- best figure, but spoken for.} It is spinoff candidate 1, \emph{pricing a
failed recall (cost bands across $F$)}, aimed at ICSE NIER, whose stated vulnerability is
that $F$ is an abstract scalar reviewers reject as ungrounded arithmetic until it is
anchored to 3--5 empirical failure scenarios. The portfolio review sanctions either
folding it in here or publishing it there --- but not both.\\[0.45em]
\textbf{C --- cheapest add.} About 1.1\,in in one column, and it renders the title claim.\\[0.45em]
\textbf{D --- good argument, bad economics.} Table~5 also carries wrong-of-attempted and
the artifact sizes, so you pay for the figure \emph{and} keep the table.\\[0.45em]
\textbf{E --- dropped,} on your read: an RQ2 side-finding about generation economics, off
the deletion spine, and \texttt{tab:bands} already carries the trend.
\par}
\end{minipage}
\end{document}""")

OUT.write_text("\n".join(L))
print("wrote", OUT)

# --- the paper's own copy, narrower because an IEEEtran column is narrower ----
PAPER_FIG = REPO / 'paper' / 'fig' / 'fig-compression.tex'
PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
PAPER_FIG.write_text(
    "%% GENERATED by scripts/paper-rollups/make_candidate_figures.py -- do not edit.\n"
    "%% Inputted by paper/main.tex as fig:deletion. Edit fig_c_body() instead.\n"
    + fig_c_body('2.45in') + "\n")
print("wrote", PAPER_FIG)

RETENTION_FIG = REPO / 'paper' / 'fig' / 'fig-retention.tex'
RETENTION_FIG.write_text(
    "%% GENERATED by scripts/paper-rollups/make_candidate_figures.py -- do not edit.\n"
    "%% Inputted by paper/main.tex as fig:retention, which replaced the old\n"
    "%% tab:retention. Edit RETENTION_WHOLE / fig_retention_body() instead.\n"
    + fig_retention_body('1.30in') + "\n")
print("wrote", RETENTION_FIG)
