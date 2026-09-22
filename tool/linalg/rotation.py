"""NumPy rotation representation conversions."""

import numpy as np
import scipy.spatial.transform as st

from tool.linalg.pose import mat_to_rot6d, rot6d_to_mat


_SCIPY_REPRESENTATION = {
    "axis_angle": "rotvec",
    "quaternion": "quat",
    "matrix": "matrix",
    "rotation_6d": None,
}


def transform_rotation(
    value: np.ndarray,
    from_rep: str,
    to_rep: str,
) -> np.ndarray:
    """Convert rotations between axis-angle, quaternion, matrix, and row-based 6D.

    Leading batch dimensions are preserved. Quaternions use SciPy's ``xyzw``
    component order, matrices have shape ``[..., 3, 3]``, and 6D rotations
    contain the first two matrix rows in row-major order.
    """
    try:
        scipy_from_rep = _SCIPY_REPRESENTATION[from_rep]
        scipy_to_rep = _SCIPY_REPRESENTATION[to_rep]
    except KeyError as exc:
        valid = ", ".join(_SCIPY_REPRESENTATION)
        raise ValueError(
            f"Unsupported rotation representation {exc.args[0]!r}; expected one of: {valid}"
        ) from exc

    if scipy_from_rep is not None and scipy_to_rep is not None:
        rotation = getattr(st.Rotation, f"from_{scipy_from_rep}")(value)
        return getattr(rotation, f"as_{scipy_to_rep}")()

    if scipy_from_rep is None:
        mat = rot6d_to_mat(value)
    else:
        mat = getattr(st.Rotation, f"from_{scipy_from_rep}")(value).as_matrix()
    if scipy_to_rep is None:
        return mat_to_rot6d(mat)
    return getattr(st.Rotation.from_matrix(mat), f"as_{scipy_to_rep}")()


class RotationTransformer:
    """Bidirectional adapter between two supported rotation representations."""

    def __init__(
        self,
        from_rep: str = "axis_angle",
        to_rep: str = "rotation_6d",
    ) -> None:
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, value: np.ndarray) -> np.ndarray:
        return transform_rotation(value, from_rep=self.from_rep, to_rep=self.to_rep)

    def inverse(self, value: np.ndarray) -> np.ndarray:
        return transform_rotation(value, from_rep=self.to_rep, to_rep=self.from_rep)
