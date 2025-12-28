# metacode_study_project
메타코드 스터디 실습프로젝트

프로젝트 개요: Pokémon Domain-Constrained RAG QA
1. 한줄 소개

포켓몬의 **규칙 기반 지식(타입 상성, 학습 가능 여부 등)**은 룰 엔진으로 결정적 계산하고, 설명/요약/근거 제시는 RAG로 보강하는 한국어 질의응답 시스템.

2. 목표

LLM이 자주 틀리는 “정답이 딱 정해진” 영역을 코드(룰 엔진)로 통제해서 환각을 줄인다.

오픈 데이터(PokeAPI)를 활용해 데이터 수집 부담 없이 완성도 높은 RAG 시스템을 만든다.

“왜 RAG가 필요한지”를 도메인 규칙 + 근거 인용으로 설득력 있게 보여준다.

3. 핵심 기능 (MVP)
(1) 한국어 질문 → 영문 검색 → 한국어 답변

사용자 질문(한국어)을 영문 쿼리로 변환

영문 코퍼스(기술 설명/포켓몬 설명)를 벡터 검색

결과를 근거로 한국어로 답변 생성

(2) 룰 엔진 기반 타입 상성 Q&A

예)

“이상해씨 상대로 유리한 타입 알려줘”

“불꽃 타입이 강한/약한 타입은?”

“물/땅 복합 타입 상대로 전기 기술이 몇 배?”

→ 룰 엔진이 배율 계산을 확정하고, LLM은 결과 해설만 담당

(3) 근거 제시(출처 링크/스니펫)

“어떤 문서/기술 설명을 근거로 답했는지”를 함께 제공

RAG 결과 문단/요약을 함께 보여줌

4. 확장 기능 (선택)

learnset(기술 학습 가능 여부) 판정 룰 노드 추가

팀 분석: 팀 타입 약점/커버리지 분석(룰 + 데이터 기반)

하이브리드 검색(BM25 + 벡터) + 랭크 융합(RRF)

5. 데이터 소스

PokeAPI: 포켓몬/타입/기술/특성/진화 등 구조 데이터(JSON)

(선택) Kaggle Pokémon stats CSV: 팀 분석/정량 계산용

벡터DB 인덱싱 대상:

기술 설명(effect text)

포켓몬/특성 설명(flavor text)

6. 시스템 아키텍처
파이프라인

Router(질문 분류)

상성/배율 질문 → type_rule

설명/정보 질문 → rag_search

혼합 질문 → type_rule + rag_search

Rule Engine

타입 상성 배율 계산

결정값(structured output) 생성

RAG

청크 생성 → 임베딩 → 벡터 검색

Top-k 근거 문서 반환

Answer Generator

룰 결과(팩트) + RAG 근거를 합쳐 한국어로 답변

“룰 결과 수치/결론은 변경 금지” 가드 포함

7. 기술 스택(예시)

Python, FastAPI (또는 Flask)

Vector DB: OpenSearch(k-NN) / Chroma / FAISS 중 택1

Embedding: multilingual/e5 계열 or bge-m3 같은 다국어 임베딩

Orchestration: LangGraph (router + 노드 분기)

(선택) 번역: LLM 기반 번역 또는 간단 번역모델

8. 평가 방법

룰 질문: 정답(배율/유불리) 정확도 100% 목표

RAG 질문: 근거 포함 여부, 답변 일관성, 회수율(top-k hit)

혼합 질문: “룰 값 일치 + 근거 인용” 동시 만족률

9. 데모 시나리오(샘플)

“리자몽 상대로 유리한 타입 추천해줘(이유 포함)”

“전기 기술이 땅 타입에 왜 안 통함?”

“이 기술(Thunderbolt) 효과 설명하고, 어떤 포켓몬이 잘 쓰는지 알려줘”



사용 데이터셋

https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description/data

https://www.kaggle.com/datasets/cristobalmitchell/pokedex?utm_source=chatgpt.com

https://www.kaggle.com/datasets/arnavvvvv/pokemon-pokedex?utm_source=chatgpt.com

https://github.com/PokeAPI/api-data?utm_source=chatgpt.com

초기에는 번호 151번까지 데이터를 정제하고 이후 E2E 개발 후 확장


데이터가 영문 베이스이므로 영문 챗봇으로 개발 후 확장

