# Security Testing Report (Remediation Update)

Date: 2026-02-11  
Repository: `tch`

## What was fixed in code
1. **Session token hardening**
   - `verify_session_token` now safely handles malformed base64/signature/payload and returns explicit `401` errors instead of unexpected decode exceptions.
2. **Backup restore filename sanitization**
   - Upload filename for `/api/backups/restore` is now normalized to basename (`Path(...).name`) before writing temporary file.
   - This closes absolute/relative path injection risks through crafted multipart filename values.
3. **Unsafe runtime assert removed**
   - Replaced `assert m2 is not None` after month creation with explicit `HTTPException(500)`.
4. **Reduced broad exception swallowing in scheduler paths**
   - `parse_hhmm` now catches specific exceptions.
   - scheduler job removal now catches `JobLookupError` specifically.
   - logging persistence fallbacks catch `sqlite3.Error` and print traceback.

## Validation executed
```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. pytest -q tests/test_security_auth_and_hardening.py -vv
bandit -q -r app.py cashflow_routes.py cashflow_models.py cashflow_bot.py -f json -o bandit_report.json
pip-audit
```

## Results
- Dynamic tests: **23 passed**.
- Security-focused tests: **8 passed**.
- Bandit findings reduced from previous 37 to **32** (High: 0).
- `pip-audit` still reports external dependency vulnerabilities:
  - `python-multipart 0.0.21` → CVE-2026-24486 (fix: 0.0.22)
  - `pip 25.3` → CVE-2026-1703 (fix: 26.0)

## Remaining risks and remediation
1. Upgrade runtime dependencies in deployment/lock files:
   - `python-multipart >= 0.0.22`
   - `pip >= 26.0` in build/runtime images.
2. Continue triage of residual Bandit warnings (`B608`, `B110`, `B112`) and gradually refactor to stricter SQL/query and exception handling patterns.

## Repro
```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. pytest -q tests/test_security_auth_and_hardening.py -vv
bandit -q -r app.py cashflow_routes.py cashflow_models.py cashflow_bot.py -f json -o bandit_report.json
pip-audit
```
