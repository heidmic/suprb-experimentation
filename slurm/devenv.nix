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
        package = pkgs.python312;
        version = "3.12.11";  # Use this only if you need a specific patch version, may build from source

      # Use the uv package manager:
        uv = {
            enable = true;
            sync.enable = true;
        };

      # Use venv and requirements.txt:
        venv = {
          enable = true;
          requirements = ./requirements.txt;  # Create this yourself
        };
    };

    languages.rust = {
      # If you need Rust, comment this in:
      #   enable = true;
      #   channel = "stable"; # or "nixpkgs" or "nightly"
      #   version = "1.81.0"; # or "latest"
    };
  };
