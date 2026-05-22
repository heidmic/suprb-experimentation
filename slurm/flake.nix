{
  inputs = {
    # Dieser korrekte Commit enthält exakt Python 3.12.11
    nixpkgs.url = "github:nixos/nixpkgs/e5f9da4b679b361498b8c2be783ff3021966a938";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, devenv, ... } @ inputs:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [ (import ./devenv.nix) ];
      };
    };
}