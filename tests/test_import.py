import fftgyro_quad_id


def test_version_is_string() -> None:
    assert isinstance(fftgyro_quad_id.__version__, str)
    assert fftgyro_quad_id.__version__
