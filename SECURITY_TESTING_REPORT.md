# Security Testing Report

Date: 2026-02-11  
Repository: `tch`

## Scope
- Dynamic/API security checks via pytest + FastAPI TestClient.
- Auth/session hardening checks around bearer handling and HMAC session tokens.
- Basic path traversal hardening check for backup download route.
- Static analysis with Bandit.
- Dependency vulnerability audit with pip-audit.

## Executed checks
1. `PYTHONPATH=. pytest -q tests/test_security_auth_and_hardening.py -vv`
2. `PYTHONPATH=. pytest -q`
3. `bandit -q -r app.py cashflow_routes.py cashflow_models.py cashflow_bot.py -f txt`
4. `pip-audit`

## Automated test results
- Security-focused tests added and passed:
  - tampered session token signature is rejected (`401`)
  - expired session token is rejected (`401`)
  - `/api/me` rejects missing bearer token (`401`)
  - `/api/me` denies inactive user even with valid token (`403`)
  - path traversal pattern to backup download does not expose files (`404`)
- Full project test suite status: **20 passed**.

## Static analysis findings (Bandit)
- Reported issues: **37 total** (Medium: 16, Low: 21, High: 0).
- Notable categories:
  - `B608` (string-constructed SQL query warnings) in `app.py` and `cashflow_models.py`
  - `B110/B112` (broad exception handling with `pass/continue`)
  - `B104` (binding to `0.0.0.0` in run configuration)
- Important context: part of `B608` items are likely false positives where interpolated fragments are locally controlled, but each flagged place should be reviewed and either refactored or explicitly justified.

## Dependency audit findings (pip-audit)
- Vulnerabilities found: **2**
  1. `pip 25.3` — `CVE-2026-1703` (fix: `26.0`)
  2. `python-multipart 0.0.21` — `CVE-2026-24486` (fix: `0.0.22`)

## Risk assessment summary
- **Auth core checks**: baseline protections work for tested cases.
- **Main risks requiring follow-up**:
  1. Broad exception swallowing in critical paths can hide security-relevant failures.
  2. SQL-string construction warnings should be triaged and reduced for defense-in-depth.
  3. Known vulnerable dependency (`python-multipart`) should be upgraded promptly.

## Recommended remediation plan
1. Upgrade vulnerable dependencies:
   - `python-multipart` to `0.0.22+`
   - refresh toolchain package `pip` in CI/runtime images.
2. Triage Bandit `B608` warnings and migrate to stricter parameterized query builders where feasible.
3. Replace `except Exception: pass/continue` with narrower exception types + structured logging.
4. Add CI security gate:
   - run `bandit` and `pip-audit` on each PR,
   - fail builds for High severity and known exploitable Mediums.

## Repro commands
```bash
PYTHONPATH=. pytest -q tests/test_security_auth_and_hardening.py -vv
PYTHONPATH=. pytest -q
bandit -q -r app.py cashflow_routes.py cashflow_models.py cashflow_bot.py -f txt
pip-audit
```
