"""서울 샘플 라벨/오디오 EDA. README.md의 모든 수치를 재생성한다.

    python src/eda.py --label_dir <02.라벨링데이터> --audio_dir <01.원천데이터>
"""
import argparse, json, re, wave
from collections import Counter, defaultdict
from pathlib import Path

TARGET9 = {"고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"}


def load(label_dir):
    recs = []
    for f in sorted(Path(label_dir).rglob("*.json")):
        d = json.load(open(f))
        d["_stem"] = f.stem
        recs.append(d)
    return recs


def merged_span(utts):
    """겹침을 합친 발화 구간 총합(ms)."""
    iv, out = sorted([u["startAt"], u["endAt"]] for u in utts), []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return sum(b - a for a, b in out)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main(label_dir, audio_dir):
    recs = load(label_dir)
    utts = [u for r in recs for u in r["utterances"]]
    durs = [u["endAt"] - u["startAt"] for u in utts]
    total = sum(r["endAt"] - r["startAt"] for r in recs)
    wavs = {f.stem: f for f in Path(audio_dir).rglob("*.wav")}

    print(f"# EDA — 통화 {len(recs)} / 발화 {len(utts)}\n")

    print("## 0. 무결성")
    print(f"JSON/WAV: {len(recs)}/{len(wavs)} | 이름 매칭 실패 {len(set(r['_stem'] for r in recs) - set(wavs))}")
    print(f"root startAt 전부 0: {all(r['startAt'] == 0 for r in recs)}")
    gaps, over = [], 0
    for r in recs:
        with wave.open(str(wavs[r["_stem"]])) as w:
            ms = w.getnframes() / w.getframerate() * 1000
        gaps.append(abs(ms - r["endAt"]))
        over += sum(1 for u in r["utterances"] if u["endAt"] > ms)
    print(f"endAt vs 실제 오디오 오차(ms): 최대 {max(gaps):.0f} / 중앙 {pct(gaps, .5):.0f}")
    print(f"오디오 길이 초과 발화: {over}건")
    print(f"빈 필드: " + ", ".join(f"{k} {sum(1 for r in recs if r.get(k) in (None, '', []))}"
                                   for k in ("triage", "symptom", "address", "gender")))

    print("\n## 1. 오디오 포맷")
    meta = Counter()
    for f in wavs.values():
        with wave.open(str(f)) as w:
            meta[(w.getnchannels(), w.getframerate(), w.getsampwidth() * 8)] += 1
    for k, v in meta.items():
        print(f"channels={k[0]} rate={k[1]}Hz width={k[2]}bit → {v}/{len(wavs)}")
    cd = [r["endAt"] - r["startAt"] for r in recs]
    print(f"통화 길이(ms): 최소 {min(cd)} / 중앙 {pct(cd, .5)} / 최대 {max(cd)}")
    print(f"파일당 평균 {sum(f.stat().st_size for f in wavs.values()) / len(wavs) / 1e6:.2f}MB")

    print("\n## 2. 발화 커버리지")
    uni = sum(merged_span(r["utterances"]) for r in recs)
    print(f"전체 {total/60000:.1f}분 / 발화합(중복포함) {sum(durs)/60000:.1f}분 ({sum(durs)/total:.1%}) "
          f"/ union {uni/60000:.1f}분 ({uni/total:.1%})")

    print("\n## 3. Mission 1 — 성별")
    g = Counter(r["gender"] for r in recs)
    print(f"통화 단위: {dict(g)} → majority baseline {max(g.values())/len(recs):.1%}")
    gs = Counter((r["gender"], u["speaker"]) for r in recs for u in r["utterances"])
    print("발화 단위 (gender, speaker):", dict(gs))
    n0 = sum(1 for u in utts if u["speaker"] == 0)
    print(f"수보자(speaker=0) 발화 비율 {n0/len(utts):.1%}")
    d1 = [u["endAt"] - u["startAt"] for u in utts if u["speaker"] == 1]
    print(f"신고자 발화 길이(ms): p10 {pct(d1,.1)} / 중앙 {pct(d1,.5)} / p90 {pct(d1,.9)}")

    print("\n## 4. Mission 2 — 화자")
    kw = Counter(u["speaker"] for u in utts if "119" in u["text"] or "상황실" in u["text"])
    q = Counter(u["speaker"] for u in utts if u["text"].strip().endswith("?"))
    print(f"'119/상황실' 언급: {dict(kw)} | 물음표 종결: {dict(q)}")
    print(f"speaker 분포: {dict(Counter(u['speaker'] for u in utts))}")
    for s in (0, 1):
        d = [u["endAt"] - u["startAt"] for u in utts if u["speaker"] == s]
        print(f"  speaker{s} 길이(ms): p10 {pct(d,.1)} 중앙 {pct(d,.5)} p90 {pct(d,.9)}")
    for th in (1000, 700, 500):
        n = sum(1 for x in durs if x < th)
        print(f"  {th}ms 미만 조각: {n} ({n/len(durs):.1%})")
    short = Counter(u["text"] for u in utts if u["endAt"] - u["startAt"] < 700)
    print("  0.7초 미만 상위 내용:", short.most_common(5))
    ov = sum(1 for r in recs for a, b in zip(r["utterances"], r["utterances"][1:]) if b["startAt"] < a["endAt"])
    print(f"인접 발화 겹침: {ov}/{len(utts)-len(recs)} ({ov/(len(utts)-len(recs)):.1%})")
    alt = sum(1 for r in recs if all(a["speaker"] != b["speaker"]
                                     for a, b in zip(r["utterances"], r["utterances"][1:])))
    print(f"완전 교대 통화: {alt}/{len(recs)} | 첫 발화 speaker: {dict(Counter(r['utterances'][0]['speaker'] for r in recs))}")

    print("\n## 5. Mission 3 — 증상")
    tok = [s for r in recs for s in r["symptom"]]
    print(f"원본 종류 {len(set(tok))} / 토큰 {len(tok)}")
    print("상위:", ", ".join(f"{k} {v}" for k, v in Counter(tok).most_common(14)))
    keep = [t for t in tok if t in TARGET9]
    print(f"9종 필터 후: {len(keep)}/{len(tok)} ({len(keep)/len(tok):.1%})")
    print("  클래스별:", {t: sum(1 for x in keep if x == t) for t in sorted(TARGET9)})
    print("  라벨 개수 분포:", dict(sorted(Counter(
        len([s for s in r["symptom"] if s in TARGET9]) for r in recs).items())))
    hit = tot9 = 0
    for r in recs:
        txt = "".join(u["text"] for u in r["utterances"])
        for s in r["symptom"]:
            if s in TARGET9:
                tot9 += 1
                hit += s in txt
    print(f"정답 단어가 전사에 문자 그대로 등장: {hit}/{tot9} ({hit/tot9:.0%})")
    cross = defaultdict(Counter)
    for r in recs:
        if any(s in TARGET9 for s in r["symptom"]):
            cross["증상有"][r["disasterLarge"]] += 1
    print("증상 있는 통화의 대분류:", dict(cross["증상有"]))

    print("\n## 6. 전사 품질")
    for pat, name in ((r"\{", "중괄호"), (r"\(", "소괄호"), (r"\*", "별표"),
                      (r"[A-Za-z]", "영문"), (r"\d{3}", "숫자3자리+")):
        print(f"  {name}: {sum(1 for u in utts if re.search(pat, u['text']))}건")

    print("\n## 7. 재난 분류 / 지역")
    print("대분류:", dict(Counter(r["disasterLarge"] for r in recs)))
    print("자치구:", dict(Counter((r["address"] or " ").split()[1] if len((r["address"] or " ").split()) > 1
                                 else "?" for r in recs).most_common()))
    print("파일명 날짜:", dict(sorted(Counter(re.search(r"_(\d+)$", r["_stem"]).group(1) for r in recs).items())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir", default="Sample/02.라벨링데이터")
    p.add_argument("--audio_dir", default="Sample/01.원천데이터")
    a = p.parse_args()
    main(a.label_dir, a.audio_dir)
