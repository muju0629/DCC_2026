# DCC 예선 브리프 (2026 데이터+AI 크리에이터 캠프 대학부)

문제: 전화대화 음성 및 전사 데이터를 이용한 응급상황 인식
데이터: AI Hub "위급상황 음성/음향(고도화) - 119 지능형 신고접수 음성 인식 데이터" (dataSetSn=71768, 87.29GB, 2023 구축)

## 1. 미션 3개

예선 최종 평가 = mission 1, 2, 3 각각의 **평가 순위를 합산**. 셋 다 버리면 안 됨.

| | 과제 | 입력 | 출력 | 지표 |
|---|---|---|---|---|
| Mission 1 | 신고자 성별 분류 | 음성(원천데이터) | gender (남/여) | Accuracy |
| Mission 2 | 신고자/119대원 분류 | 한 사람의 한 발화 조각 | speaker (0=수보자 / 1=신고자) | Accuracy |
| Mission 3 | 환자 증상 인식 | 대화 전사 텍스트 | symptom (9종 멀티라벨) | 클래스별 F1 계산 후 평균 |

### Mission 1 세부

- 라벨링데이터는 **학습 시 startAt, endAt, speaker만** 사용 가능
- **inference 시에는 startAt, endAt만** 사용 가능 (speaker 못 씀)
- 힌트: 짧은 음성 padding, 긴 음성 sliding window + stride
- CSV: `[audio file name], [startAt], [endAt], [gender]`

### Mission 2 세부

- 조각 기준은 라벨링데이터의 startAt, endAt
- startAt, endAt 이외 annotation 사용 불가. **text도 사용 불가**
- **발화의 홀짝 등 순서 사용 불가**
- CSV: `[audio file name], [startAt], [endAt], [speaker]`

### Mission 3 세부

- 입력은 전사 텍스트만. **전사 텍스트 외 라벨링데이터 사용 불가**
- 출력 symptom 개수는 **0개~9개**
- 대상 클래스 9개: 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란
- 9개 외 symptom은 평가 제외, 전처리에서 라벨에서만 제거 (샘플 자체는 유지)
- symptom은 string type으로 출력
- CSV: `[label file name], [symptom]`
- 성능평가: 증상 종류별 F1을 따로 계산 후 평균 (슬라이드 10에 상세, 미확보)

## 2. 조건 (슬라이드 5)

- AI Hub 제공 데이터만 사용, 외부 데이터 추가 금지
- 각 mission이 지정한 입력/출력만 사용 (지정되지 않은 라벨링 데이터를 입력으로 사용 금지)
- 공개 오픈소스 pre-trained 모델 허용 (ResNet50 등). **별도 개발 없는 상용 AI API(Google Vision, GPT-4o 등) 금지**
- 예선 최종 평가는 mission 1, 2, 3의 평가 순위를 합산
- **Training set으로는 서울 데이터만 사용**
- Validation 폴더 내 데이터는 학습에 사용 금지
- Sample instance마다 label에 따라 상이한 전처리 금지
- 평가 데이터에서만 유리한 특별 전처리 금지

도구 후보: KoBERT, MFCC, Mel-spectrogram, Wav2Vec2, AST, RandomForest, ResNet

음성 처리 경로 (슬라이드 참고용):

- Raw waveform → librosa로 MFCC / Mel-spectrogram
- MFCC/Mel-spectrogram을 2D 이미지로 취급 → ResNet, ViT 적용
- End-to-end → Wav2Vec2, HuBERT, AST fine-tuning

## 3. 실행규칙 (슬라이드 11)

평가 시 아래 명령 **1회 실행**으로 결과 파일이 생성되어야 함.

```
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path {checkpoint file} --output ./outputs/mission{미션번호}.csv
```

내부적으로 전처리, feature 추출, 추론, 후처리 여러 단계를 쓰더라도 최종 실행은 이 한 번으로 끝나야 함.

## 4. 제출물 (슬라이드 12)

- inference.py (11쪽 조건 충족)
- 학습 코드 파일 (**학습 로그가 포함된 .ipynb**)
- 학습 완료된 모델 가중치 (.pt / .pth / .ckpt)
- 모델 코드 파일 (.py)
- requirements.txt
- 결과 정리 PPT (데이터 분석 내용, 모델 설명, 결과 분석, 사회안전 시사점)
- 코드 내 불필요 부분 삭제, 주석 필수

## 5. 일정과 점수 반영

- 교육 프로그램: 8차시, 8/20 ~ 10/14, https://creatoredu.kbig.kr/ (ID/PW = 참가자 핸드폰 번호)
  - **팀원 진도율 평균 80% 이상 5점 / 50~79% 3점 / 1~49% 1점**
- 멘토링데이: 예선 9/5, 9/19, 10/10 / 본선 11/14. 9시~18시, Zoom, 팀당 50분
  - 참여도 평가 반영. 캠 켜고 본인 확인. **미참석 2회 이하 5점 / 3~5회 3점 / 6회 이상 1점**
- Colab Pro 지원: 팀 인원수만큼 1회. 서류 8/20~10/14, bigdatamanager@nia.or.kr
  - 서류: 한화 출금내역, 영수증, 통장 사본 (PDF 하나로 병합)
  - 메일 제목: `대학부_팀명_코랩프로 지원서류`
  - 팀장 또는 팀원 1명이 일괄 구매, 지원금은 10/31 이내 입금

## 6. 데이터 구조 (활용 가이드라인 문서 기준)

- 디렉토리: 지역(서울/인천/광주) > 대분류(구급/구조/화재/기타)
- 파일명: `[작업아이디]_[신고접수일(시)].[wav|json]`
  - 서울은 신고접수**일**, 인천/광주는 신고접수일**시** → 파일명으로 지역 구분 가능
- 서울 건수: 구급 77,849 / 구조 14,707 / 화재 8,677 / 기타 3,263 = 104,496건
- 통화 길이: endAt 30,000~180,000 msec (30초~180초)
- 전체 라벨 분포: 성별 M 52.09% / F 47.91%

라벨 JSON 스키마:

```json
{
  "_id": "...", "audioPath": "...", "recordId": "...", "status": 12,
  "startAt": 0, "endAt": 94200,
  "utterances": [
    {"id": "...", "startAt": 25173, "endAt": 29406, "text": "...", "speaker": 0}
  ],
  "mediaType": "mobile", "gender": "M", "address": "...",
  "disasterLarge": "구급", "disasterMedium": "질병(중증 외)",
  "urgencyLevel": "중", "sentiment": "불안/걱정",
  "symptom": ["기타통증"], "triage": "준응급증상"
}
```

speaker: 0 = 수보자(119대원), 1 = 신고자

**주의**: 활용 가이드라인 문서의 예시 JSON에서 speaker 0이 주소를 말하고(신고자처럼) speaker 1이 연락처를 묻는다(수보자처럼). 정의와 반대로 보임. Mission 2는 이 라벨이 전부이므로 실제 데이터로 반드시 검증할 것.
→ **검증 완료. 정의가 맞다** ([작업일지 §4](worklog/2026-08-21-sample-eda.md) 참조). 가이드라인 문서의 예시 JSON이 잘못 실린 것.

## 7. 참고: AI Hub 공개 벤치마크

같은 데이터로 이미 공개된 모델 성능. 우리 미션과 과제가 다르지만 난이도 감각용.

- NLP 기반 긴급재난/중증질환 분류 (Kc-ELECTRA): 89.49%
- 멀티모달 (Kc-ELECTRA + AST): 91.13%
- STT: CER < 10
- 16종 분류: F1 90+

---

## 샘플 데이터와의 대조 (EDA 결과)

| 브리프 | 서울 샘플 100건 실측 | 판정 |
|---|---|---|
| 통화 길이 30,000~180,000ms | 30,120 ~ 163,800ms | 일치 |
| 서울 104,496건 (구급 74.5 / 구조 14.1 / 화재 8.3 / 기타 3.1%) | 구급 75 / 구조 15 / 화재 7 / 기타 3 | 일치 |
| 성별 M 52.09 / F 47.91% | M 72 / F 28 | **불일치** — 표본오차 or 서울 편향. 전체 Training으로 재측정 필요 |
| speaker 0 = 수보자 | '119입니다' 발화 전부 speaker 0 | 정의가 맞음 |
| mediaType "mobile" | 실제 값은 `"Mobile"` (대문자 M), 100/100 | 대소문자 주의 |
| 파일명 = 작업아이디_신고접수일 | `651e464d...f06268f6_20220101.json` (8자리) | 일치 |
