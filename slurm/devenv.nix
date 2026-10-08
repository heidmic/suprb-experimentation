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
            
            version = "3.12.11"; # Use this only if you need a specific patch version, may build from source


            # Use venv and requirements.txt:
            venv = {
                enable = true;
                requirements = ../requirements.txt; # Create this yourself
            };
        };
    };

    services.postgres = {
    enable = true;
    package = pkgs.postgresql_16; # Definiert die Version
    initialDatabases = [
      { name = "optuna_db"; }
    ];
    # Erstellt einen lokalen Socket im Projektverzeichnis, um Netzwerkkonflikte zu vermeiden
    listen_addresses = "127.0.0.1"; 
  };


  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];
}
