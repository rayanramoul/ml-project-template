"""This is an example of a mock test.

The goal of a mock test is to isolate the behavior of the function under test by replacing (mocking) it.
Its dependencies with controlled, simulated versions.
This allows us to test how the function handles certain scenarios without
actually calling external systems, like APIs or databases.

In this example:
- We are testing the `process_data()` function, which depends on `fetch_data()`.
- The `fetch_data()` function fetches data from an external API.
- To avoid making real API calls during testing, we use `unittest.mock.patch`
  to mock the `fetch_data()` function, simulating its behavior and controlling
  its return values.
- This allows us to test different outcomes (e.g., success or failure) without
  relying on external systems.
"""

from unittest.mock import patch

from src.utils.utils import process_data


def test_process_data_success() -> None:
    """Test the process_data function with a successful API call."""
    mock_data = {"key": "value", "key2": "value2"}  # Mock data to return

    with patch("src.utils.utils.fetch_data", return_value=mock_data) as mock_fetch:
        result = process_data("http://example.com/api")
        assert result == len(mock_data)  # Should return the length of the mock data

        mock_fetch.assert_called_once_with(
            "http://example.com/api"
        )  # Check that fetch_data was called with the correct URL


def test_process_data_failure() -> None:
    """Test the process_data function when the API call fails."""
    with patch("src.utils.utils.fetch_data", return_value=None) as mock_fetch:
        result = process_data("http://example.com/api")
        assert result == 0  # Since fetch_data returns None, process_data should return 0

        mock_fetch.assert_called_once_with("http://example.com/api")
