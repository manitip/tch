# Deep Testing Report

Date: 2026-02-11  
Repository: `tch`

## Scope executed
- Existing automated tests (`pytest`) for keyboard behavior, reports API, month summary isolation, and PNG report rendering.
- Extended run with coverage reporting over `app.py` and `cashflow_routes.py`.
- Syntax sanity check via bytecode compilation for primary Python modules.

## Commands and outcomes
1. Initial run failed in collection due to missing import path (`ModuleNotFoundError: app`).
2. Re-run with `PYTHONPATH=.` fixed path issue.
3. Installed missing optional dependency `Pillow` to unlock skipped PNG tests.
4. Installed `pytest-cov` to generate coverage.

## Final status
- **Automated tests**: 9 passed, 0 failed, 0 skipped.
- **Coverage**:
  - `app.py`: 34%
  - `cashflow_routes.py`: 17%
  - Total: 32%
- **Compilation sanity**: success for core modules.

## Key findings
1. Test suite is stable after proper environment setup (`PYTHONPATH=.`).
2. PNG rendering tests require `Pillow`; otherwise visual checks are skipped.
3. Functional coverage of business-critical areas exists but is far from full-system confidence, especially for:
   - Route handlers in `cashflow_routes.py`
   - Broad parts of `app.py` (Telegram flows, edge conditions, auth/session branches, scheduler code)

## Risks / gaps
- Low measured line coverage indicates many code paths are unverified by automation.
- End-to-end runtime flows (bot + FastAPI + DB + auth integration) are not fully covered in current tests.
- No load/performance or chaos/reliability testing observed.

## Recommendations
1. Add API integration tests for `cashflow_routes.py` endpoints (happy path + validation + auth errors).
2. Add test matrix for roles/permissions and account isolation.
3. Add scheduler tests for weekly/monthly jobs with time-zone edge cases.
4. Add attachment upload tests (size/mime limits, corrupted payloads).
5. Add smoke E2E scenario (start app, seed DB, call key endpoints, generate report artifacts).

## Repro baseline
```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. pytest --cov=app --cov=cashflow_routes --cov-report=term-missing -q
python -m compileall app.py cashflow_routes.py cashflow_bot.py cashflow_models.py server.py
```
