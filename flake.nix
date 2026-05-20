{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling"; # packages are usually pre-built, but may be older
    # nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable"; # newest packages, but may build from source

    #nixpkgs.url = "github:NixOS/nixpkgs/35ad3c79b6c264aa73bd8e7ca1dd0ffb67bd73b1"; #to old, does not work with lib.mkpackageoption
    #nixpkgs.url = "github:NixOS/nixpkgs/687f05a9184cad4eaf905c48b63649e3a86f5433"; #new

    devenv.url = "github:cachix/devenv";
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
