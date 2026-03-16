# vLLM 코딩 컨벤션

> AI 최적화 원본: `.agents/context/coding-conventions.yaml`

## SPDX 라이선스 헤더 (필수)

모든 `.py` 파일 최상단 2줄:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

예외: 빈 `__init__.py`, `vllm/third_party/` 하위 파일

## Ruff 린트 규칙

활성화된 규칙: E (pycodestyle), F (Pyflakes), UP (pyupgrade), B (bugbear), ISC (implicit str concat), SIM (simplify), I (isort), G (logging-format)

### 자주 걸리는 위반

```python
# ❌ print 사용 금지
print('debug info')

# ✅ logger 사용
from vllm.logger import init_logger
logger = init_logger(__name__)
logger.info('debug info')

# ❌ 빈 except 금지
try:
    ...
except:
    pass

# ✅ 구체적인 예외 타입 지정
try:
    ...
except Exception:
    pass

# ❌ logger에 f-string 금지
logger.info(f'value={x}')

# ✅ %s 포맷 사용
logger.info('value=%s', x)
```

## 타입 어노테이션 (mypy, Python 3.10+ 호환)

```python
# ✅ collections.abc 사용 (not typing)
from collections.abc import Callable, Sequence, Mapping, AsyncGenerator

# ✅ typing_extensions 사용 (3.10 호환)
from typing_extensions import TypeIs, TypeAlias, Self

# ✅ TYPE_CHECKING 가드로 순환 임포트 방지
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = object
```

## Protocol 인터페이스 패턴

새 인터페이스 추가 시 `SupportsTranscription` 패턴을 따릅니다:

```python
from typing import ClassVar, Literal, Protocol, runtime_checkable
from typing_extensions import TypeIs

@runtime_checkable
class SupportsFoo(Protocol):
    supports_foo: ClassVar[Literal[True]] = True

    def foo_method(self, arg: SomeType) -> ReturnType: ...

@overload
def supports_foo(model: type[object]) -> TypeIs[type[SupportsFoo]]: ...

@overload
def supports_foo(model: object) -> TypeIs[SupportsFoo]: ...

def supports_foo(
    model: type[object] | object,
) -> TypeIs[type[SupportsFoo]] | TypeIs[SupportsFoo]:
    return getattr(model, "supports_foo", False)
```

추가 위치:
- 인터페이스: `vllm/model_executor/models/interfaces.py`
- 등록: `vllm/model_executor/models/registry.py`

## PR 거절 트리거

- SPDX 헤더 없는 새 .py 파일
- production 코드에 `print()`
- 빈 `except:`
- public 함수에 타입 어노테이션 없음
- `typing.Callable/Sequence` 직접 import (collections.abc 사용)
- logger에 f-string 사용
- 새 기능에 테스트 없음
- 함수 시그니처에 mutable default 인자
