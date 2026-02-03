# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PATH = str(REPO_ROOT)
CONTRIBUTORS_FILE_PATH = REPO_ROOT / "CONTRIBUTORS.md"
SHORTLOG_REF = "main"


def collect_contributor_entries(repo_path: str = REPO_PATH, ref: str = SHORTLOG_REF) -> set[str]:
    """Return a set of contributor identifiers (email or GitHub username).

    - If a contributor has a GitHub noreply email, we extract the username from it and
      use that as their identifier.
    - Otherwise we fall back to the email address itself.
    """
    result = subprocess.run(
        ["git", "shortlog", "-sne", ref],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    username_or_email: set[str] = set()

    for line in result.stdout.splitlines():
        m = re.search(r"\s*\d+\s+(.*)\s+<(.+)>", line)
        if not m:
            continue
        name, email = m.groups()

        username: str | None = None
        # Extract GitHub username from noreply addresses like:
        #   username@users.noreply.github.com
        #   123456+username@users.noreply.github.com
        if email.endswith("@users.noreply.github.com"):
            local_part = email.split("@", 1)[0]
            # Strip any numeric prefix before '+' that GitHub may add
            if "+" in local_part:
                _, username = local_part.split("+", 1)
            else:
                username = local_part

        if username is not None:
            username_or_email.add(username)
        else:
            username_or_email.add(email)

    return username_or_email


def contributors_in_contributors_file(contributors_file_path: str, username_or_email: set[str]) -> bool:
    """Check that all contributor identifiers are present in CONTRIBUTORS.md.

    An identifier that contains '@' is treated as an email; others are treated as
    GitHub usernames and are expected to appear as '@username' in the file.
    """
    with open(contributors_file_path) as f:
        contributors_file_content = f.read()

    missing_entries = sorted(entry for entry in username_or_email if entry not in contributors_file_content)

    if not missing_entries:
        print("All contributor entries are present in CONTRIBUTORS.md.")
        return True

    print("The following contributor entries are missing from CONTRIBUTORS.md:")
    for entry in missing_entries:
        is_email = "@" in entry
        if is_email:
            print(f"email: {entry}")
        else:
            print(f"username: @{entry}")

    return False


def main() -> None:
    username_or_email = collect_contributor_entries()
    print("Contributors:")
    for entry in username_or_email:
        is_email = "@" in entry
        if is_email:
            print(f"\temail: {entry}")
        else:
            print(f"\tusername: @{entry}")
    result = contributors_in_contributors_file(CONTRIBUTORS_FILE_PATH, username_or_email)
    if not result:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
