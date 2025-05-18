include(default)

[settings]
compiler=clang
compiler.version=19
build_type=Release

[conf]
tools.build:compiler_executables={"cpp": "clang++", "c": "clang", "rc": "clang"}
tools.build:cflags=["-march=native"]
tools.build:cxxflags=["-march=native"]
