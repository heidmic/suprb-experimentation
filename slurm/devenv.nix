{ pkgs, ... }:

let
  python = pkgs.python312;
in
{
  packages = [
    python
    pkgs.git
    pkgs.gcc
  ];

  languages.python = {
    enable = true;
    package = python;

    venv.enable = true;

    pip.enable = true;

    pip.install = ''
      pip install -r ../requirements.txt
    '';
  };

  env = {
    PYTHONPATH = "../src:..";

    MPLBACKEND = "Agg";

    MLFLOW_TRACKING_URI = "file:../mlruns";
  };
}