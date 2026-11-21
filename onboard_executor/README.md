# Raspberry Pi executor

C++20 implementation of experiment execution, session control, RPM capture and
MAVLink transport. See `CMakeLists.txt` for targets and dependencies and
`../protocol/fftgyro.xml` for the canonical dialect.

The full build requires an external PX4 MAVLink checkout and nlohmann/json.
The Linux hardware build additionally uses libgpiod. The files under
`config/` are research examples, not a deployment authorization.

Standalone RPM receiver build (no MAVLink dependency):

```sh
cmake -S onboard_executor -B build/rpm -DFFTGYRO_RPM_ONLY=ON
cmake --build build/rpm
```

The receiver uses Linux device APIs; build it on Linux. This command builds code
and does not operate connected hardware. The full executor also has C++ unit and
transport tests registered with CTest. They require its external build dependencies.
