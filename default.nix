# Nix development environment
#
# build:
# nix-build -I "BBPpkgs=https://github.com/BlueBrain/bbp-nixpkgs/archive/master.tar.gz" default.nix
#
# build and test:
# nix-build -I "BBPpkgs=https://github.com/BlueBrain/bbp-nixpkgs/archive/master.tar.gz" --arg testExec true  default.nix  -j 4
#
# dev shell:
# nix-shell -I "BBPpkgs=https://github.com/BlueBrain/bbp-nixpkgs/archive/master.tar.gz"  default.nix
#
with import <BBPpkgs> { };


stdenv.mkDerivation rec {
      name = "Brion-DEV_ENV";
      src = ./.;
      buildInputs = [stdenv git pkgconfig boost zlib cmake hdf5-cpp bbptestdata ];
      cmakeFlags="-DBoost_USE_STATIC_LIBS=FALSE " ;   

}

