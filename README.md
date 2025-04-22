# Low Accuracy logf

Natural logarithm implementation with maximum error of 2.1 ULP.

## Build and test
```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..
python tests/test.py
```

## Plots of testing
For interval [1, 2]:

![Alt text](tests/graphs/lalogf.png?raw=true)
