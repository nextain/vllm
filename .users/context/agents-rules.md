# nextain/vllm 포크 규칙

> AI 최적화 원본: `.agents/context/agents-rules.json`

## 포크 목적

vLLM 공식 레포(`vllm-project/vllm`)의 Nextain 포크.
목표: `SupportsAudioOutput` 인터페이스 추가 + MiniCPM-o 오디오 출력 지원 (Naia OS용).

## 최우선 규칙: upstream 존중

**업스트림 규칙(`AGENTS.md`)이 모든 포크 규칙보다 우선.**

vLLM 메인테이너는 AI가 생성한 PR에 매우 회의적입니다.
모든 코드는 "AI PR을 반기지 않는 메인테이너"가 봤을 때 통과할 수 있어야 합니다.

## AI의 역할 분담

| 영역 | 담당 |
|------|------|
| 코딩 규칙 위반 (SPDX, ruff, DCO) | 하네스 자동 검사 |
| upstream 논의 논거 준비 (Q&A, 반론 대응) | AI가 컨텍스트로 준비 |
| 설계 판단, 트레이드오프 결정 | Luke 직접 판단 |
| 실제 upstream 논의 주도 | Luke 직접 수행 |

> AI slop으로 보이지 않으려면 코드 규칙은 하네스가 막고,
> 설계 논거는 AI가 준비하고, 논의는 Luke가 직접 해야 함.
> `audio-output-design.yaml` §6 upstream_discussion_prep 참조.

## 세션 시작 시 필수 읽기

1. `AGENTS.md` — 업스트림 vLLM AI 기여 정책 (가장 높은 우선순위)
2. `.agents/context/agents-rules.json` — 포크 규칙 (SoT)
3. `.agents/context/project-index.yaml` — 컨텍스트 인덱스
4. `.agents/context/contributing-guide.yaml` — vLLM 기여 체크리스트
5. `.agents/context/coding-conventions.yaml` — vLLM 코딩 컨벤션

## Upstream 준수 규칙

| 규칙 | 설명 |
|------|------|
| 중복 PR 체크 | upstream PR 열기 전 항상 `gh pr list` 로 중복 확인 |
| 저가치 PR 금지 | 단독 오타/스타일 수정 PR 금지 — 실질적 변경에 묶어서 |
| 인간 방어 | PR 제출자는 모든 변경 라인을 이해하고 설명할 수 있어야 함 |
| AI 공개 | AI 보조 커밋에는 `Co-authored-by: Claude` 트레일러 필수 |
| Signed-off-by | DCO 서명 필수 (pre-commit이 자동 추가) |
| pre-commit | 모든 pre-commit 훅 통과 후 커밋 |

## 하네스 게이트

커밋 전 3개 Claude Code 훅이 자동 검사:

1. `spdx-header-check.js` — Python 파일 SPDX 헤더 확인
2. `upstream-quality-check.js` — `print()`, 빈 `except:`, logger f-string 감지
3. `commit-ai-guard.js` — AI 공개 트레일러 + 모호한 커밋 메시지 차단

합격 기준: FN=0% (체크리스트 항목), FP<5%

## 브랜치 정책

- 실험 작업: `feature/audio-output/{설명}` 또는 `fix/{설명}` (포크 main으로만)
- upstream PR: 하네스 3개 훅 + pre-commit + 테스트 + 중복 체크 모두 통과 후
