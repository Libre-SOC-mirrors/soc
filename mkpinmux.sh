#!/bin/sh
cd pinmux
python2 src/pinmux_generator.py -v -s ls180 -o ls180
python2 src/pinmux_generator.py -v -s ngi_router -o ngi_router
