import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def build(rebuild=False):
    if rebuild:
        for pattern in ("build", "resize_repro*.so", "__pycache__"):
            for path in glob.glob(os.path.join(ROOT, pattern)):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)

    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("=== build FAILED ===")
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        sys.exit(10)
    print("build OK")


def run():
    import paddle

    paddle.enable_compat()
    import resize_repro

    t = paddle.ones([2], dtype=paddle.int32, device="cpu")
    print(f"Input tensor: shape={list(t.shape)}, dtype={t.dtype}")

    try:
        log = resize_repro.test_resize(t)
        print(log)
        print("=" * 50)
        print("TEST PASSED - resize_() bug is NOT present (fix is active).")
    except RuntimeError as exc:
        print(f"\nTEST FAILED - bug reproduced:\n  {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="clean build directory before building",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip build step and use the existing extension",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    if not args.no_build:
        build(rebuild=args.rebuild)

    run()


if __name__ == "__main__":
    main()
