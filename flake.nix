{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling"; # packages are usually pre-built, but may be older
    # nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable"; # newest packages, but may build from source
    devenv.url = "github:cachix/devenv";

    # Comment this in if you need a specific patch version of Python.
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    nixpkgs-python.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    ...
  } @ inputs: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    # If you're using non-OSS software:
    # nixpkgs.allowUnfree = true;

    # If you need native CUDA support (doesn't apply to packages like PyTorch which bundle their own CUDA):
    # nixpkgs.config.cudaSupport = true;

    devShells.${system}.default = devenv.lib.mkShell {
      inherit inputs pkgs;
      # devenv.nix will contain your configuration
      modules = [(import ./devenv.nix)];
    };
  };
}
