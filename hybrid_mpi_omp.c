#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

typedef struct {
    int first;
    int second;
} Pair;

/* --- thread-local midPoint for qsort comparator --- */
static Pair midPoint;
#pragma omp threadprivate(midPoint)

static int quad(Pair p) {
    if (p.first >= 0 && p.second >= 0) return 1;
    if (p.first <= 0 && p.second >= 0) return 2;
    if (p.first <= 0 && p.second <= 0) return 3;
    return 4;
}

static int orientation(Pair a, Pair b, Pair c) {
    long long res = (long long)(b.second - a.second) * (c.first - b.first)
                  - (long long)(c.second - b.second) * (b.first - a.first);
    if (res == 0) return 0;
    return (res > 0 ? 1 : -1);
}

static int cmp_by_x(const void *p1, const void *p2) {
    const Pair *a = (const Pair *)p1;
    const Pair *b = (const Pair *)p2;
    if (a->first != b->first)   return (a->first < b->first) ? -1 : 1;
    if (a->second != b->second) return (a->second < b->second) ? -1 : 1;
    return 0;
}

static int angle_compare(const void *p1, const void *q1) {
    const Pair *p = (const Pair *)p1;
    const Pair *q = (const Pair *)q1;

    Pair p_diff = {p->first - midPoint.first, p->second - midPoint.second};
    Pair q_diff = {q->first - midPoint.first, q->second - midPoint.second};

    int one = quad(p_diff);
    int two = quad(q_diff);
    if (one != two) return (one < two ? -1 : 1);

    long long cross = (long long)p_diff.second * q_diff.first
                    - (long long)q_diff.second * p_diff.first;

    return (cross < 0 ? -1 : 1);
}

static Pair *bruteHull(Pair *a, int n, int *ret_size) {
    Pair *ret = (Pair *)malloc((size_t)n * sizeof(Pair));
    int k = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int x1 = a[i].first, x2 = a[j].first;
            int y1 = a[i].second, y2 = a[j].second;

            int A = y1 - y2;
            int B = x2 - x1;
            long long C = (long long)x1 * y2 - (long long)y1 * x2;

            int pos = 0, neg = 0;
            for (int m = 0; m < n; m++) {
                long long val = (long long)A * a[m].first +
                                (long long)B * a[m].second + C;
                if (val <= 0) neg++;
                if (val >= 0) pos++;
            }

            if (pos == n || neg == n) {
                int unique = 1;
                for (int z = 0; z < k; z++)
                    if (ret[z].first == a[i].first && ret[z].second == a[i].second)
                        unique = 0;
                if (unique) ret[k++] = a[i];

                unique = 1;
                for (int z = 0; z < k; z++)
                    if (ret[z].first == a[j].first && ret[z].second == a[j].second)
                        unique = 0;
                if (unique) ret[k++] = a[j];
            }
        }
    }

    *ret_size = k;
    ret = (Pair *)realloc(ret, (size_t)(k > 0 ? k : 1) * sizeof(Pair));

    midPoint.first = 0;
    midPoint.second = 0;
    for (int i = 0; i < k; i++) {
        midPoint.first += ret[i].first;
        midPoint.second += ret[i].second;
    }
    if (k > 0) {
        midPoint.first /= k;
        midPoint.second /= k;
    }

    qsort(ret, (size_t)k, sizeof(Pair), angle_compare);
    return ret;
}

static Pair *merger(Pair *a, int n1, Pair *b, int n2, int *ret_size) {
    int ia = 0, ib = 0;
    for (int i = 1; i < n1; i++) if (a[i].first > a[ia].first) ia = i;
    for (int i = 1; i < n2; i++) if (b[i].first < b[ib].first) ib = i;

    int inda = ia, indb = ib, done = 0;

    while (!done) {
        done = 1;
        while (orientation(b[indb], a[inda], a[(inda + 1) % n1]) >= 0)
            inda = (inda + 1) % n1;

        while (orientation(a[inda], b[indb], b[(n2 + indb - 1) % n2]) <= 0) {
            indb = (n2 + indb - 1) % n2;
            done = 0;
        }
    }
    int uppera = inda, upperb = indb;

    inda = ia; indb = ib; done = 0;
    while (!done) {
        done = 1;
        while (orientation(a[inda], b[indb], b[(indb + 1) % n2]) >= 0)
            indb = (indb + 1) % n2;

        while (orientation(b[indb], a[inda], a[(n1 + inda - 1) % n1]) <= 0) {
            inda = (n1 + inda - 1) % n1;
            done = 0;
        }
    }
    int lowera = inda, lowerb = indb;

    Pair *ret = (Pair *)malloc((size_t)(n1 + n2) * sizeof(Pair));
    int k = 0;

    int idx = uppera;
    ret[k++] = a[idx];
    while (idx != lowera) {
        idx = (idx + 1) % n1;
        ret[k++] = a[idx];
    }

    idx = lowerb;
    ret[k++] = b[idx];
    while (idx != upperb) {
        idx = (idx + 1) % n2;
        ret[k++] = b[idx];
    }

    *ret_size = k;
    return ret;
}

#define BRUTE_THRESHOLD 150
#define MAX_PARALLEL_DEPTH 6

static Pair *divide_omp(Pair *a, int n, int depth, int *ret_size) {
    if (n <= BRUTE_THRESHOLD) {
        return bruteHull(a, n, ret_size);
    }

    int mid = n / 2;

    Pair *left_hull = NULL, *right_hull = NULL;
    int left_size = 0, right_size = 0;

    if (depth < MAX_PARALLEL_DEPTH) {
        #pragma omp task shared(left_hull, left_size) firstprivate(a, mid, depth)
        left_hull = divide_omp(a, mid, depth + 1, &left_size);

        #pragma omp task shared(right_hull, right_size) firstprivate(a, n, mid, depth)
        right_hull = divide_omp(a + mid, n - mid, depth + 1, &right_size);

        #pragma omp taskwait
    } else {
        left_hull  = divide_omp(a, mid, depth + 1, &left_size);
        right_hull = divide_omp(a + mid, n - mid, depth + 1, &right_size);
    }

    Pair *res = merger(left_hull, left_size, right_hull, right_size, ret_size);
    free(left_hull);
    free(right_hull);
    return res;
}

static void compute_chunk(int N, int world_size, int rank, int *local_n, int *start_index) {
    int base = N / world_size;
    int rem  = N % world_size;
    *local_n = base + (rank < rem ? 1 : 0);
    if (start_index) {
        *start_index = rank * base + (rank < rem ? rank : rem);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Datatype MPI_PAIR;
    MPI_Type_contiguous(2, MPI_INT, &MPI_PAIR);
    MPI_Type_commit(&MPI_PAIR);

    int N_POINTS = 0;
    Pair *all_points = NULL;

    if (rank == 0) {
        FILE *f = fopen("points.bin", "rb");
        if (!f) { perror("fopen points.bin"); MPI_Abort(MPI_COMM_WORLD, 1); }

        if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        long long fsize = ftell(f);
        if (fsize < 0) { perror("ftell"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        rewind(f);

        if (fsize % (long long)sizeof(Pair) != 0) {
            fprintf(stderr, "points.bin size not multiple of Pair\n");
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        N_POINTS = (int)(fsize / (long long)sizeof(Pair));
        all_points = (Pair *)malloc((size_t)N_POINTS * sizeof(Pair));
        if (!all_points) { perror("malloc all_points"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }

        size_t got = fread(all_points, sizeof(Pair), (size_t)N_POINTS, f);
        fclose(f);
        if (got != (size_t)N_POINTS) { fprintf(stderr, "fread failed\n"); free(all_points); MPI_Abort(MPI_COMM_WORLD, 1); }

        qsort(all_points, (size_t)N_POINTS, sizeof(Pair), cmp_by_x);
    }

    MPI_Bcast(&N_POINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n, start_idx;
    compute_chunk(N_POINTS, world_size, rank, &local_n, &start_idx);

    Pair *local_points = (Pair *)malloc((size_t)local_n * sizeof(Pair));
    if (!local_points) { perror("malloc local_points"); MPI_Abort(MPI_COMM_WORLD, 1); }

    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = (int *)malloc((size_t)world_size * sizeof(int));
        displs = (int *)malloc((size_t)world_size * sizeof(int));
        for (int r = 0; r < world_size; r++) {
            int ln, st;
            compute_chunk(N_POINTS, world_size, r, &ln, &st);
            counts[r] = ln;
            displs[r] = st;
        }
    }

    MPI_Scatterv(all_points, counts, displs, MPI_PAIR,
                 local_points, local_n, MPI_PAIR,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(all_points);
        free(counts);
        free(displs);
    }

    /* local hull  OpenMP */
    int local_hull_size = 0;
    Pair *local_hull = NULL;

    #pragma omp parallel
    {
        #pragma omp single
        {
            local_hull = divide_omp(local_points, local_n, 0, &local_hull_size);
        }
    }
    free(local_points);

    int step = 1;
    while (step < world_size) {
        if (rank % (2 * step) == 0) {
            int partner = rank + step;
            if (partner < world_size) {
                int other_size;
                MPI_Recv(&other_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Pair *other_hull = (Pair *)malloc((size_t)other_size * sizeof(Pair));
                MPI_Recv(other_hull, other_size, MPI_PAIR, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int merged_size;
                Pair *merged = merger(local_hull, local_hull_size, other_hull, other_size, &merged_size);

                free(local_hull);
                free(other_hull);
                local_hull = merged;
                local_hull_size = merged_size;
            }
        } else {
            int partner = rank - step;
            MPI_Send(&local_hull_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local_hull, local_hull_size, MPI_PAIR, partner, 0, MPI_COMM_WORLD);
            free(local_hull);
            local_hull = NULL;
            break;
        }
        step *= 2;
    }

    if (rank == 0) {
        printf("Global Convex Hull has %d points:\n", local_hull_size);
        for (int i = 0; i < local_hull_size; i++) {
            printf("%d %d\n", local_hull[i].first, local_hull[i].second);
        }
        free(local_hull);
    }

    MPI_Type_free(&MPI_PAIR);
    MPI_Finalize();
    return 0;
}
