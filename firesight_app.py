from firesight.config import load_config
from firesight.gui import launch


def main() -> None:
    launch(load_config())


if __name__ == "__main__":
    main()
