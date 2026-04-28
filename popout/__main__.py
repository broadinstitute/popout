"""Allow `python -m popout` invocation. Used by --reproducible=seeding
to spawn a deterministic seeding subprocess from the parent popout
process without depending on the popout console-script being on PATH.
"""

from popout.cli import main

if __name__ == "__main__":
    main()
