from __future__ import annotations

import ucp


def main() -> None:
    print("UCP_MODULE", ucp.__file__)
    print("UCP_VERSION", ucp.get_ucx_version())
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            if "libucp.so" in line or "libuct.so" in line or "libucs.so" in line:
                path = line.strip().split()[-1]
                if path.startswith("/"):
                    paths.add(path)
    for path in sorted(paths):
        print(path)


if __name__ == "__main__":
    main()
