# vLLM 기여 가이드 (요약)

> AI 최적화 원본: `.agents/context/contributing-guide.yaml`
> 원본 출처: vllm-project/vllm AGENTS.md + CONTRIBUTING.md

## 기여 전 필수 체크리스트

### 1. 중복 PR 확인
```bash
gh issue view <이슈번호> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<이슈번호> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<키워드>"
```
이미 같은 변경을 다루는 PR이 열려 있으면 **새 PR 열지 말 것**.

### 2. 실질성 확인
단독 오타 수정, 단독 스타일 변경 → **PR 금지**. 실질적 변경에 묶어야 함.

### 3. AI 사용 공개
AI 보조 사용 시 PR 설명에 반드시:
- 기존 PR과 중복이 아닌 이유
- 실행한 테스트 명령과 결과
- AI 사용 명시

커밋 트레일러: `Co-authored-by: Claude <noreply@anthropic.com>`

## 환경 설정

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성
uv venv --python 3.12
source .venv/bin/activate

# lint 도구 설치 + pre-commit 설치
uv pip install -r requirements/lint.txt
pre-commit install

# Python만 변경 시
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# C/C++ 변경 포함 시
uv pip install -e .
```

## 테스트

```bash
uv pip install pytest pytest-asyncio tblib
pytest tests/path/to/test.py -v -s -k test_name
pytest tests/path/to/dir -v -s
```

## pre-commit (커밋 전 필수)

```bash
pre-commit run               # 스테이지된 파일에 실행
pre-commit run --all-files   # 전체 파일에 실행
pre-commit run ruff-check --all-files
```

로컬에서 실행되는 훅:
- ruff-check + ruff-format
- typos (오타 검사)
- mypy (Python 3.10)
- shellcheck
- signoff-commit (Signed-off-by 자동 추가)
- check-spdx-header
- check-root-lazy-imports

## 커밋 메시지 형식

```
feat: add SupportsAudioOutput interface for TTS models

Implements Protocol interface following SupportsTranscription pattern.

Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: Luke N <luke@nextain.io>
```

## PR 설명 (AI 보조 시 필수 섹션)

1. 기존 PR과 중복이 아닌 이유
2. 변경 내용과 이유
3. 실행한 테스트 명령 + 결과 (터미널 출력 붙여넣기)
4. AI 사용 도구 및 역할 명시
