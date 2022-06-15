with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "35ad3c79b6c264aa73bd8e7ca1dd0ffb67bd73b1";
}) { config = { allowUnfree = true; }; };


mkShell {
  venvDir = "./_venv";
  # Add dependencies that pip can't fetch here (or that we don't want to
  # install using pip).
  buildInputs = (with pkgs.python39Packages; [ python39 venvShellHook wheel ])
     ++ (import ./system-dependencies.nix { inherit pkgs; });
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc]}:${pkgs.lib.makeLibraryPath [pkgs.zlib]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${python39.sitePackages}:$PYTHONPATH
  '';
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';
}

