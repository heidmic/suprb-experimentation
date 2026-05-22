{
  inputs = {
    # Dieser korrekte Commit enthält exakt Python 3.12.11
    nixpkgs.url = "github:nixos/nixpkgs/8cbadfa068534bdd8238eea362d2bf0b1d46b7e8";
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