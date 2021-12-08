with import (builtins.fetchGit {
  name = "nixpkgs-2021-12-08";
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "35ad3c79b6c264aa73bd8e7ca1dd0ffb67bd73b1";
}) { config = { allowUnfree = true; }; };


mkShell {
  name = "beste-shell";
  venvDir = "./_venv";
  # Add dependencies that pip can't fetch here (or that we don't want to
  # install using pip).
  buildInputs = (with pkgs.python39Packages; [ python venvShellHook wheel ])
    ++ (import ./system-dependencies.nix { inherit pkgs; });
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    pip install -r requirements.txt
  '';
}

