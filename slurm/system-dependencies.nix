{ pkgs }:

with pkgs;

[ ]
# Not necessary to include zlib due to including it directly in
# LD_LIBRARY_PATH.
# [ zlib ]
