# /bin/bash
g++ sim.cpp -o sim.exe\
    -std=c++17 \
    -O3 -march=native \
    -lemp-zk -lemp-tool -lssl -lcrypto -lpthread
