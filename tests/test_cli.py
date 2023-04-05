from unittest.mock import patch

from mltools.cli.cli import main


@patch("mltools.cli.cli.argparse.ArgumentParser.parse_args")
def test_main_help_flag(mock_parse_args):
    with patch("mltools.modules"):
        main()
    mock_parse_args.assert_called_once()
