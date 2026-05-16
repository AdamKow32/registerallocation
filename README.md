# Register Allocation — Weighted Spill Problem

Implementation of metaheuristics for the register allocation problem, formalized as weighted graph coloring with spilling. The problem is **NP-hard**, Chaitin (1981) proved it is equivalent to k-coloring of the interference graph, which justifies the use of metaheuristics.

A compiler must assign program variables to a limited number of CPU registers. Two variables that are **live at the same time** cannot share a register. This relationship is modeled as an **interference graph** `G = (V, E)`.

If a variable cannot be assigned a register, it is **spilled** to RAM, generating a cost proportional to how often it is used (spill cost). The goal is to minimize total weighted spill cost.

## Project Structure
 
```
registerallocation/
├── src/
│   ├── instance.cpp       # instance loader + adjacency list
│   ├── logger.cpp         # CSV result logger
│   ├── chaitin.cpp        # Chaitin greedy heuristic (baseline)
│   └── generator.cpp      # synthetic instance generator
├── data/
│   └── instance.txt       # example instance
├── logs/                  # experiment results (auto-created)
├── main.cpp               # entry point
└── CMakeLists.txt
```

## Instance File Format
 
```
n K m              <- number of variables, registers, edges
w_1 w_2 ... w_n    <- spill costs (float)
u_1 v_1            <- edges (1-indexed)
...
```

### Graph types
 
| Type       | Description                                      | NP-hardness         |
|------------|--------------------------------------------------|---------------------|
| `erdos`    | each edge added with probability p (Erdos-Renyi) | full, unconditional |
| `interval` | edge when live ranges overlap                    | via weighted spill  |
| `chordal`  | every cycle >= 4 has a chord (SSA model)         | via weighted spill  |

## References
 
- Chaitin, G. J. (1981). *Register Allocation via Coloring*.
