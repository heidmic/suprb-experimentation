{ pkgs, config, ... }: {
  languages.python = {
    enable = true;
    # Holt Python aus dem oben definierten Commit
    package = pkgs.python312;
    
    venv = {
      enable = true;
      requirements = ../requirements.txt;
    };
  };

  packages = with pkgs; [
    stdenv.cc.cc
    zlib
  ];

  env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH";
  };
}