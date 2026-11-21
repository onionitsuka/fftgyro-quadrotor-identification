"""Smooth attitude excitation and shared offline altitude controller."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HoverPitchProfile:
    amplitude_deg: float = 5.
    period_s: float = 2.
    cycles: int = 3
    hover_s: float = 2.
    recovery_s: float = 4.

    @property
    def duration_s(self):
        return self.hover_s+self.cycles*self.period_s+self.recovery_s

    def reference_deg(self,t: float) -> float:
        u = t-self.hover_s
        length = self.cycles*self.period_s
        if u <= 0 or u >= length:
            return 0.
        # Hann-windowed cosine: zero value/slope at both ends, no angle step.
        return self.amplitude_deg*math.cos(2*math.pi*u/self.period_s)*math.sin(math.pi*u/length)**2


def pitch_quaternion(pitch_deg: float) -> list[float]:
    half = math.radians(pitch_deg)/2
    return [math.cos(half),0.,math.sin(half),0.]


def altitude_thrust(z: float,vz: float,integral: float,dt: float,pitch_deg: float):
    """NED z=-2 target. Shared fixed gains, not per-model hover calibration.

    Normalized thrust .54 matches the native motor mapping near hover. Bounds
    are actuator-command saturation, not clipping the aircraft load model.
    """
    error = z+2
    candidate = max(-2.,min(2.,integral+dt*error))
    raw = (.54+.10*error+.08*vz+.03*candidate)/math.cos(math.radians(pitch_deg))
    if raw > .85 and error > 0 or raw < .15 and error < 0:
        candidate = integral
        raw = (.54+.10*error+.08*vz+.03*candidate)/math.cos(math.radians(pitch_deg))
    return max(.15,min(.85,raw)),candidate
