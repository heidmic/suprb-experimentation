{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: 
{
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
      #package = pkgs.python312; 
      version = "3.12.11";  # use this bc suprb requires == 3.12.11

      /* package = py.withPackages (ps: with ps; [
      suprb
      cmpbayes
      arviz
      click
      ipython
      matplotlib
      mlflow        # deckt mlflow-skinny mit ab
      optuna
      optuna-dashboard
      pandas
      pytest
      scikit-learn
      seaborn
      tabulate
      tqdm
    ]); */

      # Use the uv package manager:
      uv = {
        enable = true;
        sync.enable = true;
      };

      # Use venv and requirements.txt:
      # venv = {
      #   enable = true;
      #   requirements = ./requirements.txt;  # Create this yourself
      # };
    };
  };

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    #"/run/opengl-driver" # libcuda.so, libnvidia-*.so from host driver, only necessary for PyTorch/CUDA
  ];
}

