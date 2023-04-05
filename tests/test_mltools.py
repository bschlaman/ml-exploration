def test_mltools_importable():
    import mltools  # noqa: F401


def test_modules_importable():
    import mltools.modules  # noqa: F401


def test_utils_importable():
    import mltools.utils  # noqa: F401


def test_cli_importable():
    import mltools.cli  # noqa: F401


def test_cli_main_importable():
    from mltools.cli.cli import main  # noqa: F401
