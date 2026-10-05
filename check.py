"""Check your calculate_f1 function.

Run it from the project folder:  python check.py
"""

from evaluation import calculate_f1


def main():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 1, 1, 0]
    result = calculate_f1(y_true, y_pred)
    assert abs(result - 0.444) < 0.01, f"Expected about 0.444, got {result}"

    result = calculate_f1([1, 1, 0], [0, 0, 0])
    assert result == 0, f"When precision and recall are both 0, F1 should be 0. Got {result}"

    print("All checks passed!")


if __name__ == "__main__":
    main()
