{ pkgs, config, ... }: {
  
  # Python 3.12 aktivieren
  languages.python = {
    enable = true;
    version = "3.12";
    
    # Automatische Installation deiner requirements.txt in das venv
    venv = {
      enable = true;
      requirements = ../requirements.txt;
    };
  };

  # System-Abhängigkeiten (Ersatz für deine system-dependencies.nix und CC/zlib Hooks)
  packages = with pkgs; [
    stdenv.cc.cc
    zlib
  ];

  # Umgebungsvariablen setzen (Sauberer Ersatz für das alte postShellHook)
  env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH";
  };
}