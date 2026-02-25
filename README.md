# ConvexHull

# MPI Convex Hull (Tree Merge)
This project implements a high-performance parallel algorithm for calculating the **Convex Hull** of a large set of points ($N \approx 2.2 \text{ million}$) using the **MPI (Message Passing Interface)** library.

Beyond the standard MPI approach, this repository also includes optimized implementations for:
* **Hybrid MPI + Pthreads**
* **Hybrid MPI + OpenMP**

---

## Performance Benchmarks
Based on profiling $2.2 \times 10^6$ points:
* **MPI Implementation (4 Processes):** ~10 seconds.
* **OpenMP Implementation (Standalone):** ~30 seconds.

The MPI version demonstrates superior scalability due to the efficient **Binary Tree Merge** strategy compared to shared-memory overhead in large-scale sorting.

---

##  Technical Implementation

### 1. Workflow & Data Distribution
1.  **Generation & Global Sort:** Rank 0 generates points randomly and performs a global sort by $x$ (and $y$ for ties). This ensures that segments handled by different ranks are spatially separated.
2.  **Communication:**
    * `MPI_Bcast`: Sends metadata (`N_POINTS`) to all ranks.
    * `MPI_Scatterv`: Distributes point segments of varying sizes.
    * `MPI_PAIR`: A custom MPI data type (2 * int) created to handle the `Pair` structure efficiently.
3.  **Local Computation:** Each rank calculates its local hull using a **Divide & Conquer** method (switching to Brute Force below a specific threshold).

### 2. Parallel Binary Tree Merge
To consolidate local hulls into the final global hull, we use a reduction tree:
* In each `step`, **"even"** ranks receive a hull from a neighbor rank, perform a `merger()`, and continue.
* **"Odd"** ranks send their data and terminate their involvement in that stage.
* This logarithmic complexity $O(\log P)$ minimizes communication bottlenecks.



---

##  Development Roadmap

### Phase 1: Architecture & Migration
* Transposed sequential logic to the MPI parallel paradigm.
* Defined the communication skeleton and implemented the `MPI_PAIR` derived type.
* Established the workload distribution protocol for $N=2,200,000$ points.

### Phase 2: Master Process Control (Rank 0)
* Implemented optimized generation mechanisms and file integrity validation.
* Developed the global sorting stage.
* Finalized the distribution flow using `MPI_Scatterv`.

### Phase 3: Tree Merge Logic
* Developed the **Binary Tree Merge** structure.
* Integrated the local Divide & Conquer hulls into the global merging flow using `MPI_Send` and `MPI_Recv`.

### Phase 4: Performance Analysis
* Conducted scalability testing and generated speedup charts.
* Monitored performance across various core counts and compared MPI vs. Hybrid vs. OpenMP execution times.

---
