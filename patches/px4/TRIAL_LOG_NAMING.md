# Trial log naming (0003)

## Per-trial UTC setup

`scripts/run_sync_bench.py` now invokes `px4_trial_log.py` before each non-idle motor trial:
Pi NTP-synchronized UTC check → confirm logger idle → `system_time set <Unix seconds>` →
`system_time get` readback (within 2 s, accounting for integer-second resolution) →
`logger on` → confirm full-file recording and save ULog path. A Pi clock step during
exchange, unknown logger status, UTC mismatch, or missing recording blocks executor launch.
An existing recording is not stopped or retimed. No `SDLOG_*` parameter is changed.

The shell closes its UART before the executor starts. After executor exit/cleanup,
`logger off` and stopped-status confirmation run before waiting for RPM capture completion.
If logger start was attempted but failed to confirm, cleanup still attempts off;
if UTC failed before on, cleanup does not stop somebody else's logger.
Idle acquisition never touches PX4 UTC or logger. No automatic motor retries.

The helper is sent as Python source through the existing Pi pymavlink interpreter
(`/home/fftgyro/venvs/mavlink/bin/python`); this change needs no executor/PX4 recompilation.
The Pi must report `timedatectl show -p NTPSynchronized --value` as yes/true.
No silently substituted unsynchronized Pi clock or Windows timestamp cached before SSH is used.
This validates wall-clock naming, not sub-second sensor synchronization.

Per-run `flightlog/start.json` and `stop.json` retain UTC source, sent/readback values,
responses, errors, start/stop completion, and ULog path; collected summaries are added to
`run.json`. ULog bytes remain on PX4 SD; this is not automatic ULog download.
Tests use mock clocks/shells/child processes only. Legacy/manual `logger on` outside this coordinator is not intercepted.

The native cascade profile's existing 15 s budget is also admitted by the coordinator;
its native reference capture replaces the old additional 5 s Python reference acquisition.

## Firmware and filename format

Base: d6f12ad1c4f70ad3230afd7d86e971421e02fef4, together with0002 motor bridge.

When RTC is valid, the full log is named `YYYYMMDD_HHMMSS_NNN.ulg` in the local-date directory. `SDLOG_UTC_OFFSET=540` selects Japan time while RTC stays UTC. NNN starts001 each date and advances from the largest existing numbered filename, including aborted/diagnostic recordings. It is a recording sequence, not a scientific condition ID. Keep the coordinator's run-ID/path correspondence. Restarting logger or PX4 does not reset the number.999 exhausts the day; existing files are not overwritten.

Invalid RTC retains PX4's session/logNNN fallback; never claim a boot-relative clock is a real date. Synchronize RTC before a trial. Previously recorded files are unchanged. Standard topic profiles and per-trial on/off are unchanged. Replay/encryption suffixes are retained; filename storage increased to48 bytes.

Apply: `git apply --check --unidiff-zero 0003-fftgyro-trial-log-name-d6f12ad1c4.patch`, then the same command without `--check`.
Revert: `git apply --check --reverse --unidiff-zero 0003-fftgyro-trial-log-name-d6f12ad1c4.patch`, then omit `--check`.

