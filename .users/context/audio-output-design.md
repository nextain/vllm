# SupportsAudioOutput 인터페이스 설계

**상태**: e2e-validated
**최종 수정**: 2026-03-17
**관련 이슈**: naia-os#73

---

## 범위 결정

- **Qwen2-Audio**: 범위 외. 테스트 표면이 너무 넓음. MiniCPM-o 전용 구현 후 실증.
- upstream이 generality를 요구하면 Luke가 직접 논의. 범위 확장은 별도 PR.

---

## 구현 원칙

코드는 가능한 한 간결하게. Luke가 메인테이너 앞에서 모든 라인을 방어해야 함.

- Protocol 정의: 필수 메서드만 (확장 가능성을 위한 메서드 미리 추가 금지)
- MiniCPM-o 구현: Token2wav 연동만
- API 엔드포인트: OpenAI spec 최소 구현
- 주석: 왜(why) 설명만
- 헬퍼 함수: 두 곳 이상 쓰이지 않으면 만들지 않음

---

## 배경 / 문제

vLLM은 SupportsTranscription (오디오 입력 → 텍스트)과 SupportsRealtime (스트리밍 양방향)을 지원하지만, 텍스트 입력에서 오디오를 생성하는 모델(TTS)을 위한 인터페이스가 없음.

MiniCPM-o 4.5는 텍스트 토큰과 함께 오디오 토큰을 생성할 수 있지만, vLLM에 이를 API로 라우팅하는 표준 방법이 없음.

**대상 모델**: openbmb/MiniCPM-o-4_5
**사용 사례**: Naia OS 음성 응답 (모델이 채팅 응답으로 음성 오디오 생성)

---

## 기존 패턴

| 인터페이스 | 방향 | 예시 모델 |
|------------|------|----------|
| SupportsTranscription | 오디오 입력 → 텍스트 출력 | Whisper |
| SupportsRealtime | 스트리밍 양방향 오디오 I/O | - |
| SupportsMultiModal | 멀티모달 입력 + 텍스트 출력 | 여러 모델 |

---

## 제안 인터페이스

```python
@runtime_checkable
class SupportsAudioOutput(Protocol):
    """텍스트 입력으로 오디오 토큰을 생성하는 모델을 위한 인터페이스."""

    supports_audio_output: ClassVar[Literal[True]] = True
    audio_output_sample_rate: ClassVar[int]  # e.g. 24000 for MiniCPM-o

    def decode_audio_tokens(
        self,
        token_ids: list[int],
    ) -> "np.ndarray | None":
        """오디오 토큰 ID를 파형으로 디코딩 (float32, shape [samples]).

        TTS 스팬(<|tts_bos|> 마커)이 없으면 None 반환.
        """
        ...


def supports_audio_output(
    model: "type[object] | object",
) -> ...:
    return getattr(model, "supports_audio_output", False)
```

**위치**: `vllm/model_executor/models/interfaces.py`
**패턴**: SupportsTranscription + SupportsRealtime 동일 패턴

---

## API 서버 통합

**결정**: Chat completions 응답에 audio 필드 포함 (OpenAI 오디오 출력 방식 준수)

```
POST /v1/chat/completions
→ response.choices[0].message.audio.data = base64(WAV bytes)
```

---

## MiniCPM-o 특이사항

- **오디오 출력 메커니즘**: Token2wav 보코더
- **샘플레이트**: 24,000 Hz
- **TTS 초기화 3단계**:
  1. `from_pretrained(init_tts=True)` — TTS 스켈레톤만 생성
  2. `model.init_tts()` — Token2wav 보코더 실제 초기화
  3. `chat(generate_audio=True, output_audio_path=...)` — 파일로 저장
- **알려진 제한**:
  - INT4 양자화: TTS 미지원
  - Duplex 모드: ~28GB VRAM 필요 (RTX 3090 24GB에서 반이중만 사용)
  - Token2wav: reference audio 필수

---

## 업스트림 RFC 전략

1. Fork에서 동작하는 구현 완성 (Phase 4)
2. vllm-project/vllm에 GitHub Discussion / RFC 이슈 오픈
3. 인터페이스 설계에 대한 피드백 수집
4. 업스트림 품질 기준으로 polish (harness 게이트 모두 통과)
5. 테스트 증거 + 중복 체크와 함께 PR 오픈

**PR 분할 계획**:
- PR-1 (소): `interfaces.py` + `registry.py` — Protocol만
- PR-2 (PR-1 머지 후): MiniCPM-o 구현 + Token2wav 선택적 의존성
- PR-3 (논의 후): API 서빙 레이어

---

## E2E 검증 결과 (naia-os#73 Phase 2)

**날짜**: 2026-03-17
**환경**: RunPod RTX 3090 (24GB VRAM), vllm 0.17.1 prebuilt + Python 파일 오버레이

**패치 전략**:
- 직접 복사: `minicpmo.py`, `outputs.py`, `engine/__init__.py`, `output_processor.py`
- 수술적 패치: `interfaces.py`, `registry.py`, `gpu_model_runner.py`, `serving.py`, `outputs.py(CompletionOutput)`
- 버그 수정: `scheduler.py` — `finished` 변수 `UnboundLocalError` (`finished=False` 기본값 누락)

**테스트 결과**:

| 테스트 | 프롬프트 | 예상 | 결과 |
|--------|----------|------|------|
| Test 1: 텍스트 전용 | "What is 2+2?" | audio is None | **PASS** |
| Test 2: TTS 요청 | "Please say 'Hello' in audio." | WAV bytes 또는 None | **PARTIAL PASS** (audio=None, 크래시 없음) |

**결론**: 서빙 경로 전체(`gpu_model_runner → outputs → scheduler → engine/__init__ → output_processor → outputs.py → serving.py`)가 `audio_output` 필드를 크래시 없이 전파함을 확인.

**알려진 이슈**:
- `scheduler.py` `finished=False` 버그: fork 코드에도 잠재. upstream PR 전 수정 필요.
- TTS 전체 경로 검증(실제 WAV 바이트)은 오디오 입력 있는 테스트케이스로 별도 진행 필요.

**다음 단계**:
- Phase 3: nextain/vllm 커밋 (SPDX + AI disclosure + Signed-off-by)
- Phase 4: Luke가 vllm-project/vllm에 Design Issue 오픈 후 PR 3단계 진행

---

## 업스트림 논의 대비 Q&A

**Q1**: 왜 기존 Protocol을 확장하지 않고 새 Protocol을 만드나?
→ SupportsTranscription(ASR), SupportsRealtime(스트리밍 양방향), SupportsAudioOutput(TTS)은 직교하는 능력. 합치면 단일 책임 원칙 위반 + 기존 구현 파손.

**Q2**: MiniCPM-o 하나만을 위한 인터페이스가 너무 좁지 않나?
→ PoC 구현. Protocol은 모델 불가지론적으로 설계. Qwen2-Audio 확장은 follow-up PR.

**Q3**: Token2wav 새 의존성 처리는?
→ `decode_audio_tokens()` 내부에서 lazy import + 명확한 ImportError 메시지. 기존 선택적 의존성 패턴 동일.

**Q4**: `/v1/audio/speech` 엔드포인트에 RFC가 필요한가?
→ Protocol은 PR-1로 먼저, 엔드포인트는 PR-3로 분리. 메인테이너가 RFC 요구 시 Discussion 먼저 오픈.
