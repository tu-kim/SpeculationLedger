# PATCHES.md — vLLM 등 외부 스택 최소 패치 대장

> CLAUDE.md §10: vLLM core fork 금지. plugin/entrypoint로만 통합하고, 불가피한 최소
> 패치는 diff와 대상 커밋을 여기에 남긴다.

**현재 패치 없음.** (2026-07-11)

vLLM `04d553f` 기준 custom_class proposer 경로와 `_sample` 래핑(HOOKS.md)으로 Phase 1
통합이 무패치로 가능할 것으로 실측 판단. 런타임 검증에서 패치가 불가피해지면 아래
양식으로 기록한다.

```
## P-<n>. <제목>
- 대상: <repo> @ <commit>
- 사유: <플러그인 경로로 불가능한 이유>
- diff:
<unified diff>
- 업스트림 계획: <이슈/PR 링크 또는 불필요 사유>
```
