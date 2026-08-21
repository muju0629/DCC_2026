# DCC 2026 — 119 응급상황 인식

2026 데이터+AI 크리에이터 캠프 대학부 예선 공용 레포.
AI Hub "위급상황 음성/음향(고도화) — 119 지능형 신고접수 음성 인식 데이터"로 세 가지 분류 문제를 푼다.

**예선 최종 순위 = Mission 1·2·3 각각의 평가 순위 합산.** 한 미션도 버릴 수 없다.

| | 과제 | 입력 | 출력 | 지표 | 상태 |
|---|---|---|---|---|---|
| **Mission 1** | 신고자 성별 | 음성 | `gender` | Accuracy | 미착수 |
| **Mission 2** | 화자 분류 | 발화 조각 음성 | `speaker` 0=수보자 / 1=신고자 | Accuracy | 미착수 |
| **Mission 3** | 환자 증상 | 전사 텍스트 | `symptom` 9종 멀티라벨 | 클래스별 F1 평균 | 미착수 |

규칙 원문은 **[docs/competition_brief.md](docs/competition_brief.md)** 에 있다. 미션별 입력 제약이 빡빡하니 코드 쓰기 전에 읽을 것.

---

## 레포 구조

```
docs/
  competition_brief.md                 대회 규정 원문 + 샘플 대조표
  mission2_speaker_spectrum.md         Mission 2 채널 누수 진단
  mentoring_draft.md                   멘토링 게시판 문의 초안
  worklog/
    2026-08-21-sample-eda.md           서울 샘플 100건 EDA
src/
  eda.py                               라벨·오디오 EDA
  spectrum.py                          화자별 스펙트럼 추출 + 채널 진단
  make_figure.py                       figures/ 생성
figures/
  mission2_speaker_spectrum.pdf/.png
```

원본 데이터(`Sample/`, `*.wav`)와 중간 산출물(`artifacts/`)은 커밋하지 않는다.

```bash
pip install numpy scipy matplotlib
python src/eda.py      --label_dir Sample/02.라벨링데이터 --audio_dir Sample/01.원천데이터
python src/spectrum.py --label_dir Sample/02.라벨링데이터 --audio_dir Sample/01.원천데이터
python src/make_figure.py
```

---

## 지금까지 확정된 것

샘플 100건 EDA에서 나온 것 중 **모델 설계에 바로 영향을 주는 것들.** 근거와 전체 수치는 [작업일지](docs/worklog/2026-08-21-sample-eda.md).

**오디오는 8kHz mono다.** Wav2Vec2 / HuBERT / AST는 전부 16kHz 전제라 `Resample(8000, 16000)`이 필수고, Nyquist가 4kHz라 Mel filterbank `f_max`를 4000으로 잡아야 한다. mono라 채널 분리로 Mission 2를 푸는 길은 없다.

**타임스탬프는 믿어도 된다.** 100건 전부 `endAt`이 실제 오디오 길이와 오차 0ms. 자르기 전처리에 보정 로직이 필요 없다.

**용량 지렛대는 자르기가 아니라 통화 수다.** 발화 구간만 남겨도 22%밖에 안 줄어든다(무음이 거의 없음). 서울 10만 건 ≈ 110GB는 층화 샘플링으로 줄여야 한다.

**speaker 0 = 수보자가 맞다.** 가이드라인 문서 예시 JSON이 잘못 실린 것. 멘토에게 물어볼 필요 없다.

**Mission 2의 상한은 짧은 조각이 정한다.** 1초 미만이 27.1%이고 내용은 "네." "예."다. 텍스트도 순서도 못 쓰니 순수 음향으로 판정해야 한다. 인접 발화 24.9%가 시간상 겹치는 것도 제거 불가능한 라벨 노이즈다.

**Mission 2에 채널 누수가 있다.** 고역 에너지비 하나로 정확도 70.1%가 나오는데, 이 특징이 보는 건 목소리가 아니라 녹음 경로다 → [분석](docs/mission2_speaker_spectrum.md)

**Mission 3이 승부처다.** 9개 타깃이 전체 증상 토큰의 34%뿐이고, 정답 단어가 전사에 문자 그대로 등장하는 비율이 10%(4/39)다. 어휘 매칭으로는 안 되고 macro-F1이라 고르게 맞히는 쪽이 이긴다. 한 클래스가 F1=0이면 0.111을 통째로 잃는다.

**중간 점수를 볼 수 없다.** 리더보드가 없고 Validation은 학습 금지. Training(서울)에서 뗀 자체 held-out이 유일한 나침반이라, 데이터 파이프라인 짤 때부터 반영해야 한다.

---

## 다음 할 일

**데이터 받으면 즉시**

- [ ] 전체 Training의 gender 분포 재측정 — 샘플은 M 72 / F 28인데 문서 전체 통계는 M 52.09 / F 47.91. majority baseline이 20%p 달라진다
- [ ] 9개 타깃 symptom의 실제 클래스별 건수
- [ ] Validation/Test 지역 구성 — 파일명으로 판별 가능 (서울 8자리, 인천·광주 14자리)

**파이프라인 설계**

- [ ] 자체 held-out split 기준 결정 (랜덤 말고 시간/자치구 기준 검토)
- [ ] Mission 2 겹침 처리 방침 — **라벨을 보지 않고** 결정해야 규칙 위반이 아니다
- [ ] 층화 샘플링으로 서브셋 구성 (대분류·성별·증상 유무)

**제출 전 필수**

- [ ] `inference.py`를 Validation 폴더에 실제로 실행해서 행 수·컬럼·결측·파일명 매칭 확인 (라벨은 안 봄)
- [ ] Mission 2 길이 구간별 accuracy 분해 — 전체 평균만 보면 어디서 지는지 안 보인다
- [ ] Mission 3 클래스별 threshold를 K-fold로 정하고 fold 간 분산 확인

---

## 제출물 체크리스트

평가는 아래 명령 **1회 실행**으로 결과가 나와야 한다.

```
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path {checkpoint file} --output ./outputs/mission{미션번호}.csv
```

| 미션 | 출력 CSV 컬럼 |
|---|---|
| 1 | `[audio file name], [startAt], [endAt], [gender]` |
| 2 | `[audio file name], [startAt], [endAt], [speaker]` |
| 3 | `[label file name], [symptom]` (string, 0~9개) |

- [ ] `inference.py`
- [ ] 학습 코드 (**학습 로그 포함된 .ipynb**)
- [ ] 모델 가중치 (.pt / .pth / .ckpt)
- [ ] 모델 코드 (.py)
- [ ] `requirements.txt`
- [ ] 결과 PPT (데이터 분석 / 모델 설명 / 결과 분석 / 사회안전 시사점)
- [ ] 불필요 코드 삭제, 주석 정리

---

## 일정

| 날짜 | 내용 |
|---|---|
| 8/20 ~ 10/14 | 교육 프로그램 8차시 — 팀 평균 진도율 80%↑ 5점 / 50~79% 3점 |
| 9/5, 9/19, 10/10 | 예선 멘토링데이 (Zoom, 팀당 50분, 캠 필수) — 미참석 2회↓ 5점 |
| **10/14** | **Colab Pro 지원서류 마감** → bigdatamanager@nia.or.kr |
| 11/14 | 본선 멘토링데이 |

Colab Pro는 팀 인원수만큼 1회 지원. 팀장 또는 팀원 1명이 일괄 구매 후 서류 제출(출금내역·영수증·통장사본 PDF 병합), 메일 제목은 `대학부_팀명_코랩프로 지원서류`. **구매를 미루면 지원을 못 받는다.**

---

## 열린 질문

**멘토에게** (50분은 판단이 필요한 것에만 쓴다)

1. **Mission 2 화자 클러스터링 허용 여부** — 규칙은 "순서 사용 불가"만 막았고 클러스터링은 언급이 없다. 허용이면 통화 내 2그룹 클러스터링으로 짧은 조각 문제(27%)가 상당 부분 풀린다. 금지면 완전히 다른 모델이 필요하다.
2. **자체 validation 구성 기준** — 랜덤 split은 실제 성능을 낙관적으로 보게 만든다. 시간/지역 기준이 나은지.
3. **Colab Pro로 현실적인 규모** — 10만 건 중 어느 정도부터 포화되는지, Wav2Vec2 fine-tuning이 현실적인지.
4. **채널 특성 사용 가능 여부** — 게시판 문의 초안: [docs/mentoring_draft.md](docs/mentoring_draft.md)

**사무국 / Q&A 채널로**

- 슬라이드 10의 Mission 3 F1 상세 정의 (자료 배포 예정)
- Mission 2에서 인접 발화가 24.9% 겹치는데 조각 자르기 기준이 따로 있는지
- AI Hub 데이터 전체 다운로드 필요 여부 (서울 Training만 부분 다운로드 가능한지)
