"""화자별 스펙트럼 추출 + 채널 누수 진단 (docs/mission2_speaker_spectrum.md 근거).

    python src/spectrum.py --label_dir <02.라벨링데이터> --audio_dir <01.원천데이터>
"""
import argparse, json, wave, numpy as np
from pathlib import Path
from scipy.signal import welch

_p = argparse.ArgumentParser()
_p.add_argument("--label_dir", default="Sample/02.라벨링데이터")
_p.add_argument("--audio_dir", default="Sample/01.원천데이터")
_p.add_argument("--out", default="artifacts")
_a = _p.parse_args()
LAB, SRC = Path(_a.label_dir), Path(_a.audio_dir)
FS, NPER = 8000, 256
OUT = Path(_a.out); OUT.mkdir(parents=True, exist_ok=True)
wavmap = {f.stem: f for f in SRC.rglob("*.wav")}

def merge(iv):
    iv = sorted(iv); out=[]
    for a,b in iv:
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out

psd={0:[],1:[]}; utt=[]; sil=[]
for jf in sorted(LAB.rglob("*.json")):
    d=json.load(open(jf)); fid=jf.stem
    x=np.frombuffer(wave.open(str(wavmap[fid])).readframes(10**9),"<i2").astype(np.float64)
    us=d["utterances"]
    spans={s:merge([[u["startAt"],u["endAt"]] for u in us if u["speaker"]==s]) for s in (0,1)}
    for spk in (0,1):
        other=spans[1-spk]; clean=[]
        for a,b in spans[spk]:                      # 상대 화자 겹침 구간 제거
            cur=[(a,b)]
            for c,e in other:
                nxt=[]
                for s,t in cur:
                    if e<=s or c>=t: nxt.append((s,t)); continue
                    if c>s: nxt.append((s,c))
                    if e<t: nxt.append((e,t))
                cur=nxt
            clean+=cur
        clips=[x[int(s*FS/1000):int(t*FS/1000)] for s,t in clean if t-s>=250]
        clips=[c for c in clips if len(c)>=NPER]
        if not clips: continue
        f,p=welch(np.concatenate(clips),FS,nperseg=NPER); psd[spk].append(p/p.sum())
        for c in clips:                              # 발화(겹침 제거 조각) 단위 특징
            n=len(c)//NPER; fr=c[:n*NPER].reshape(n,NPER)
            e=(fr**2).mean(1)+1e-9
            fq,P=welch(fr,FS,nperseg=NPER,axis=-1)
            lo=P[:,(fq>=300)&(fq<2000)].sum(1); hi=P[:,(fq>=2200)&(fq<3600)].sum(1)
            r=10*np.log10(hi/(lo+1e-12)+1e-12)
            loud=e>np.percentile(e,60); quiet=e<np.percentile(e,20)
            if loud.any(): utt.append((fid,spk,float(np.median(r[loud]))))
            if quiet.any() and n>=8: sil.append((fid,spk,float(np.median(r[quiet]))))

P0,P1=np.array(psd[0]),np.array(psd[1])
np.savez(OUT/"spec.npz",freq=f,p0=P0,p1=P1,
         utt=np.array([(a,b,c) for a,b,c in utt],dtype=object),
         sil=np.array([(a,b,c) for a,b,c in sil],dtype=object))

db0,db1=10*np.log10(P0+1e-15),10*np.log10(P1+1e-15)
r0=db0.mean(0)-db0.mean(0)[(f>=300)&(f<1000)].mean(); r1=db1.mean(0)-db1.mean(0)[(f>=300)&(f<1000)].mean()
print("300–1000Hz 기준 상대 레벨")
for hz in (1000,2000,2500,3000,3500,3900):
    i=np.argmin(abs(f-hz)); print(f"  {hz:>4}Hz  접수요원 {r0[i]:+6.1f}dB   신고자 {r1[i]:+6.1f}dB   차 {r0[i]-r1[i]:+.1f}")

def report(name,R):
    fid=np.array([r[0] for r in R]); y=np.array([int(r[1]) for r in R]); v=np.array([r[2] for r in R])
    ts=np.linspace(v.min(),v.max(),600)
    acc=max(max(((v<t)==y).mean(),((v>t)==y).mean()) for t in ts)
    within=[]
    for k in set(fid):
        m=fid==k
        if len(set(y[m]))==2: within.append(v[m][y[m]==0].mean()>v[m][y[m]==1].mean())
    # 화자별 통화 절반으로 학습/절반 테스트 (전이 확인)
    calls=sorted(set(fid)); tr=set(calls[::2]); m=np.isin(fid,list(tr))
    t=max(ts,key=lambda t:((v[m]<t)==y[m]).mean()); trans=((v[~m]<t)==y[~m]).mean()
    print(f"\n[{name}] n={len(v)}조각  접수요원 {v[y==0].mean():.1f}dB / 신고자 {v[y==1].mean():.1f}dB (차 {v[y==0].mean()-v[y==1].mean():.1f})")
    print(f"  단일 임계값 최고 정확도 {acc:.1%} | 절반 학습→나머지 전이 {trans:.1%} | 통화 내 화자쌍 방향 일치 {np.mean(within):.1%} ({len(within)}통화)")
report("유성(큰 에너지) 프레임",utt)
report("무음/저에너지 프레임",sil)
