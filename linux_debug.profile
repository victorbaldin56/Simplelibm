include(default)

[settings]
compiler=clang
compiler.version=19
compiler.libcxx=libstdc++
compiler.cppstd=gnu17
build_type=Debug

[conf]
tools.build:compiler_executables={"cpp": "clang++", "c": "clang", "rc": "clang"}
tools.build:cflags=["-fsanitize=address,leak,undefined -march=native"]
tools.build:cxxflags=["-fsanitize=address,leak,undefined -march=native"]
tools.build:exelinkflags=["-fsanitize=address,leak,undefined"]
tools.build:sharedlinkflags=["-fsanitize=address,leak,undefined"]
