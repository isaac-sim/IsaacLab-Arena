#!/bin/bash
# Keep compatibility tags consistent across build, launch, and publish commands.

default_tag_for_target() {
    case "$1" in
        dev) echo latest ;;
        dev-curobo) echo curobo ;;
        *) echo "Unsupported Arena target: $1" >&2; return 2 ;;
    esac
}
