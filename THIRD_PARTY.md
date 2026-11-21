# Third-party sources and dependencies

The project-wide license has not been selected. The notice below preserves
upstream terms and does not license the rest of this repository.

## PX4

The patches in `patches/px4/` target PX4-Autopilot commit
`d6f12ad1c4f70ad3230afd7d86e971421e02fef4`. Existing copyright and license
headers in the patch files are retained. The pinned upstream license is copied
verbatim into `LICENSES/PX4-BSD-3-Clause.txt`.

Source: [PX4 license at the patch base](https://github.com/PX4/PX4-Autopilot/blob/d6f12ad1c4f70ad3230afd7d86e971421e02fef4/LICENSE).
The PX4 source tree and binaries are not distributed here.

## Stepwise model selection

The model-selection implementation in `analysis.py` identifies the reference
for the F/PSE selection rules:
[SihaoSun/stepwise_regression, pinned reference](https://github.com/SihaoSun/stepwise_regression/tree/b04095ccb9f0e5dcafae43db6ec1a4320cbd0b16/func).
The local Python implementation documents numerical adaptations and should not
be described as the upstream MATLAB package itself.

Upstream permission terms for translated or adapted implementations have not
been confirmed. No project-wide license is assigned to upstream work here.

## Runtime and build dependencies

NumPy, SciPy, pandas, Matplotlib, PyArrow, pyulog, pymavlink, pyserial,
nlohmann/json, libgpiod, MuJoCo and the test/build tools are external dependencies.
Their source trees are not vendored in this export. Their own licenses apply.
Gazebo-generated models remain outside Git. See `docs/DEPENDENCIES.md`.
