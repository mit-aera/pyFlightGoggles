#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run \
    -it \
    --publish-all \
    --rm \
    --volume "${DIR}/../FlightGoggles-PythonClient:/root/FlightGoggles-PythonClient" \
    --gpus all \
    --name py_flightgoggles \
    --privileged \
    --net "host" \
    pyfg
