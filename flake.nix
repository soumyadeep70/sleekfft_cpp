{
  description = "Optimized FFT implementation in C++";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };
  outputs =
    inputs@ { flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      perSystem =
        { pkgs, ... }:
        let 
          commonPackages = with pkgs; [
            gcc
            gdb

            cmake
            ninja
            gnumake

            valgrind
            perf

            cppcheck
            pkg-config
            fftw
            catch2_3
          ];
        in 
        {
          packages.devtools = pkgs.buildEnv {
            name = "devtools";
            paths = commonPackages;
          };
          devShells.default = pkgs.mkShell {
            packages = commonPackages;
          };
        };
    };
}
