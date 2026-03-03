## VOLE-ZKP-based NN-FAIR Simulation

- `sim.cpp` is the implementation of Pi_norm
- To use, install emp-zk (following instructions from github repo); then compile and simulate:

```
./compile.sh
./run.sh
```

Note that:

- Some variables are initialized randomly, not exact as paper
    - finding unit vec in (mod pr) is difficult, so we use random vec
    - calc SVD in (mod pr) is more difficult, so in code we suppose a, u, v are calculated
- The above suggests that the experiment simulates the whole process of protocol, while not guaranteeing the correctness
