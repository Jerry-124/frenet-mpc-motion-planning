from dataclasses import dataclass
import numpy as np


@dataclass
class ReferencePath:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    @classmethod
    def sinusoidal(cls, length: float = 140.0, amplitude: float = 2.0, wavelength: float = 70.0):
        raw_x = np.linspace(0.0, length, 2500)
        raw_y = amplitude * np.sin(2.0 * np.pi * raw_x / wavelength)
        ds = np.hypot(np.diff(raw_x), np.diff(raw_y))
        arc = np.r_[0.0, np.cumsum(ds)]
        s = np.linspace(0.0, arc[-1], 1400)
        x = np.interp(s, arc, raw_x)
        y = np.interp(s, arc, raw_y)
        yaw = np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
        return cls(s=s, x=x, y=y, yaw=yaw)

    def sample(self, s_query: np.ndarray):
        sq = np.clip(np.asarray(s_query), self.s[0], self.s[-1])
        return (
            np.interp(sq, self.s, self.x),
            np.interp(sq, self.s, self.y),
            np.interp(sq, self.s, self.yaw),
        )

    def frenet_to_cartesian(self, s_query: np.ndarray, d_query: np.ndarray):
        x_ref, y_ref, yaw_ref = self.sample(s_query)
        x = x_ref - np.asarray(d_query) * np.sin(yaw_ref)
        y = y_ref + np.asarray(d_query) * np.cos(yaw_ref)
        return x, y

    def cartesian_to_frenet(self, x: float, y: float) -> tuple[float, float]:
        """Project a Cartesian point onto the sampled reference path."""
        index = int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))
        normal = np.array([-np.sin(self.yaw[index]), np.cos(self.yaw[index])])
        displacement = np.array([x - self.x[index], y - self.y[index]])
        return float(self.s[index]), float(np.dot(displacement, normal))
