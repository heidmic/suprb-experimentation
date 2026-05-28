{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = with pkgs; [
    gcc
    gccStdenv.cc.cc.lib
    libz
  ];
  languages = {
    python = {
      # If you don't need Python, comment this out:
      enable = true;

      # Choose your Python version:
      # package = pkgs.python312;
      version = "3.12.11";  # Use this only if you need a specific patch version, may build from source

      # Use the uv package manager:
      # uv = {
      #   enable = true;
      #   sync.enable = true;
      # };

      # Use venv and requirements.txt:
      venv = {
        enable = true;
        requirements = ./requirements.txt;  # Create this yourself
      };
    };
  };

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  enterShell=''
	export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DEVENV_ROOT"; # avoid issues with empty path
  '';
}

