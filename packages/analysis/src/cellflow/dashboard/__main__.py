"""Launch the TissueGraph analysis dashboard.

Usage::

    python -m cellflow.dashboard [/path/to/dataset] [--port 8050] [--no-show]
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="TissueGraph Analysis Dashboard")
    parser.add_argument(
        "dataset", nargs="?", default=None,
        help="Path to a saved TissueGraph dataset directory",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port to serve on (default: 8050)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't open the browser automatically",
    )
    args = parser.parse_args()

    from .app import serve

    serve(
        dataset_path=args.dataset,
        port=args.port,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
