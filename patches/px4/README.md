# PX4 Patch Policy

PX4-Autopilot source code must not be committed to this repository.

If PX4 changes are needed for the research workflow, manage them as patch files here. Each patch should document:

- the PX4 commit or tag it applies to
- the purpose of the change
- how to apply and revert it
- any validation command or SITL scenario used

## Current: `0002-fftgyro-motor-set-bridge-d6f12ad1c4.patch`

- Base: PX4-Autopilot commit `d6f12ad1c4f70ad3230afd7d86e971421e02fef4`
- Targets: `px4_fmu-v6x_fftgyro` and `px4_sitl_fftgyro`
- Purpose: add the minimal receiver for the Raspberry Pi executor's atomic
  four-motor MAVLink message. PX4 validates target, values, timeout, run,
  sequence, disarmed/safety/lockdown/Kill state and `COM_MOT_TEST_EN`, then
  publishes Motor 1--4 `actuator_test` entries with one timestamp.
- Protocol: `src/modules/mavlink/fftgyro.xml` is copied byte-for-byte from this
  repository's canonical `protocol/fftgyro.xml`. `ACCEPTED` means that the four
  uORB entries were queued; it does not prove physical DShot transmission.
- Run handling: sequence gaps are accepted, duplicate/old sequences are
  rejected, and the most recent 16 terminated run IDs cannot be reused during
  the same PX4 boot.

Do not apply this together with the legacy `0001` FC waveform executor. Apply
it from a clean PX4 worktree at the base commit:

```sh
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout d6f12ad1c4f70ad3230afd7d86e971421e02fef4
git submodule update --init --recursive

git apply --check /path/to/0002-fftgyro-motor-set-bridge-d6f12ad1c4.patch
git apply /path/to/0002-fftgyro-motor-set-bridge-d6f12ad1c4.patch
cmp /path/to/repository/protocol/fftgyro.xml src/modules/mavlink/fftgyro.xml
make px4_sitl_fftgyro
make px4_fmu-v6x_fftgyro
```

The flashable artifact is:

```text
build/px4_fmu-v6x_fftgyro/px4_fmu-v6x_fftgyro.px4
```

Flash it as QGroundControl custom firmware, or with the Pixhawk connected by USB:

```sh
make px4_fmu-v6x_fftgyro upload
```

After flashing, configure TELEM1 through normal PX4 parameters rather than
`extras.txt`:

- assign one MAVLink instance to `TELEM 1`;
- set `SER_TEL1_BAUD` to `921600 8N1`;
- set `COM_MOT_TEST_EN=1`;
- reboot PX4, then use `mavlink status` to confirm the TELEM1 instance and baud.

Do not copy raw numeric enum values for `MAV_*_CONFIG` between unrelated PX4
versions. Use the parameter dropdown/name on this pinned firmware and save a
parameter backup before changing it. TELEM1 is dedicated to the Pi in the current
setup; leave the Holybro radio disconnected.

Revert with:

```sh
git apply --check --reverse /path/to/0002-fftgyro-motor-set-bridge-d6f12ad1c4.patch
git apply --reverse /path/to/0002-fftgyro-motor-set-bridge-d6f12ad1c4.patch
```

## Legacy: `0001-fftgyro-waveform-executor-d6f12ad1c4.patch`

- Base: PX4-Autopilot commit `d6f12ad1c4f70ad3230afd7d86e971421e02fef4`
- Target: `px4_fmu-v6x_default` and `px4_sitl_default`
- Purpose: generate one guarded differential step directly for Motor1--4 on
  the FC clock. The prepare command carries collective, delta, duration,
  positive-motor mask, ramp, and preroll; MAVLink then only starts or aborts.
- Safety posture: disabled by default (`FGX_EN=0`), prepare is disarmed-only,
  start requires disarmed state, released safety, `COM_MOT_TEST_EN=1`, valid
  a two-positive/two-negative motor mask, and a matching configuration token. Output uses
  PX4 actuator-test overrides with a 150 ms timeout. Link loss/failsafe enters
  ramp-down; armed/lockdown/kill immediately releases all four overrides.

Apply from a clean PX4 worktree at the base commit:

```sh
git apply --check /path/to/0001-fftgyro-waveform-executor-d6f12ad1c4.patch
git apply /path/to/0001-fftgyro-waveform-executor-d6f12ad1c4.patch
```

Revert before making other overlapping changes:

```sh
git apply --check --reverse /path/to/0001-fftgyro-waveform-executor-d6f12ad1c4.patch
git apply --reverse /path/to/0001-fftgyro-waveform-executor-d6f12ad1c4.patch
```

Validate the build and hardware interlocks on the intended installation before use.
