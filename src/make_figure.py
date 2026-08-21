"""Mission 2 화자별 주파수 특성 figure 생성."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rc('font', family=next((n for n in ['AppleGothic','NanumGothic','Malgun Gothic'] if any(n in ff.name for ff in mpl.font_manager.fontManager.ttflist)), 'DejaVu Sans')); mpl.rc('axes', unicode_minus=False); mpl.rc('pdf', fonttype=42)
import argparse
_p = argparse.ArgumentParser(); _p.add_argument("--npz", default="artifacts/spec.npz")
_p.add_argument("--out", default="figures/mission2_speaker_spectrum")
_a = _p.parse_args()
z = np.load(_a.npz, allow_pickle=True)
f, P0, P1 = z["freq"], z["p0"], z["p1"]
def band(P):
    db = 10*np.log10(P+1e-15)
    d = db - db[:, (f>=300)&(f<1000)].mean(1, keepdims=True)
    return d.mean(0), np.percentile(d,25,0), np.percentile(d,75,0)
m0,l0,h0 = band(P0); m1,l1,h1 = band(P1)
u = z["utt"]; y = np.array([int(r[1]) for r in u]); v = np.array([float(r[2]) for r in u])
B, O = "#2a78d6", "#eb6834"

fig, ax = plt.subplots(1, 2, figsize=(13,4.6), gridspec_kw=dict(width_ratios=[1.35,1]))
s = (f>=150)&(f<=3950)
ax[0].axvspan(2200, 3600, color="0.9", zorder=0)
for m,l,h,c,lab in [(m0,l0,h0,B,"접수요원 (speaker 0)"), (m1,l1,h1,O,"신고자 (speaker 1)")]:
    ax[0].fill_between(f[s], l[s], h[s], color=c, alpha=.15, lw=0)
    ax[0].plot(f[s], m[s], color=c, lw=2, label=lab)
ax[0].text(2900, 2, "2.2–3.6 kHz", ha="center", fontsize=9, color="0.35")
ax[0].set(xlabel="주파수 (Hz)", ylabel="상대 레벨 (dB, 300–1000Hz = 0)",
          title="평균 스펙트럼 — 100통화, 겹침 구간 제외", xlim=(150,3950), ylim=(-45,4))
ax[0].legend(frameon=False, loc="lower left"); ax[0].grid(alpha=.25, lw=.6)

bins = np.arange(-55,-2.5,2.5)
ax[1].hist([v[y==0], v[y==1]], bins=bins, color=[B,O], label=["접수요원","신고자"])
thr = max(np.linspace(v.min(), v.max(), 600), key=lambda t: ((v<t)==y).mean())
ax[1].axvline(thr, color="0.3", ls="--", lw=1.4)
ax[1].text(thr+.8, ax[1].get_ylim()[1]*.92, f"임계값 {thr:.1f}dB\n정확도 70.1%", fontsize=9, color="0.25")
ax[1].set(xlabel="고역 에너지비 (2.2–3.6k ÷ 0.3–2k, dB)", ylabel="발화 수",
          title=f"발화별 고역비 분포 (n={len(v):,})")
ax[1].legend(frameon=False); ax[1].grid(alpha=.25, lw=.6, axis="y")
for a in ax: a.spines[["top","right"]].set_visible(False)
fig.suptitle("화자별 주파수 특성", fontsize=13, y=1.0)
fig.tight_layout(); fig.savefig(_a.out + ".pdf", bbox_inches="tight"); fig.savefig(_a.out + ".png", dpi=160, bbox_inches="tight")
print("saved", thr)
