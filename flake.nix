{
  inputs.nixpkgs.url = "nixpkgs/release-25.05";
  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    fhs = pkgs.buildFHSEnv {
      name = "fhs-shell";
      targetPkgs = pkgs: with pkgs; [zlib uv];
    };
  in {
    devShells.${system}.default = fhs.env;
  };
}
