#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."
docker build --pull --rm -f "examples/kubernetes/dockerfile" -t simpledistributedrl:latest .
