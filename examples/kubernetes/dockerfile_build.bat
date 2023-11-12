cd /D %~dp0../..
docker build --pull --rm -f "examples/kubernetes/dockerfile" -t simpledistributedrl:latest .
