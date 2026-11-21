# FFTGYRO MAVLink dialect

`fftgyro.xml` is the canonical protocol definition. The Raspberry Pi build
generates its C headers directly from this file.

The PX4 patch contains a byte-for-byte copy because PX4 is maintained in a
separate worktree. After applying or regenerating the patch, verify that copy
before building firmware:

```sh
cmp /path/to/this-repository/protocol/fftgyro.xml \
  /path/to/PX4-Autopilot/src/modules/mavlink/fftgyro.xml
```

Changing the XML changes the protocol. Rebuild both the Pi binary and PX4
firmware from the same repository revision; never mix artifacts whose
`fftgyro.xml` SHA-256 differs.
