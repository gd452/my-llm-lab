import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _10_core.autograd_tiny.value import Value


def main() -> None:
    a, b = Value(1.0), Value(2.0)
    y = (a * b + a).tanh()
    try:
        y.backward()
    except NotImplementedError:
        print("Not implemented yet — fill in value.py TODOs first.")
        return
    print("y=", y)
    print("a.grad=", a.grad, "b.grad=", b.grad)


if __name__ == "__main__":
    main()
