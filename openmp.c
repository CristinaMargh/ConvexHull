#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
    int first;
    int second;
} Pair;

static _Thread_local Pair midPoint;

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
    Pair *ret = (Pair *)malloc(n * sizeof(Pair));
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
    ret = (Pair *)realloc(ret, (k > 0 ? k : 1) * sizeof(Pair)); 

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

    qsort(ret, k, sizeof(Pair), angle_compare);
    return ret;
}

static Pair *merger(Pair *a, int n1, Pair *b, int n2, int *ret_size) {
    int ia = 0, ib = 0;

    for (int i = 1; i < n1; i++)
        if (a[i].first > a[ia].first) ia = i;

    for (int i = 1; i < n2; i++)
        if (b[i].first < b[ib].first) ib = i;

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

    inda = ia;
    indb = ib;
    done = 0;

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

    Pair *ret = (Pair *)malloc((n1 + n2) * sizeof(Pair));
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
#define MAX_PARALLEL_DEPTH 4  

static Pair *divide_omp(Pair *a, int n, int depth, int *ret_size) {
    if (n <= BRUTE_THRESHOLD) {
        return bruteHull(a, n, ret_size);
    }

    int midIndex = n / 2;

    Pair *left_hull = NULL, *right_hull = NULL;
    int left_size = 0, right_size = 0;

    if (depth < MAX_PARALLEL_DEPTH) {
        #pragma omp task shared(left_hull, left_size) firstprivate(a, midIndex, depth)
        {
            left_hull = divide_omp(a, midIndex, depth + 1, &left_size);
        }

        #pragma omp task shared(right_hull, right_size) firstprivate(a, n, midIndex, depth)
        {
            right_hull = divide_omp(a + midIndex, n - midIndex, depth + 1, &right_size);
        }

        #pragma omp taskwait
    } else {
        left_hull  = divide_omp(a, midIndex, depth + 1, &left_size);
        right_hull = divide_omp(a + midIndex, n - midIndex, depth + 1, &right_size);
    }

    Pair *res = merger(left_hull, left_size, right_hull, right_size, ret_size);

    free(left_hull);
    free(right_hull);
    return res;
}

int main(void) {
    const int N_POINTS = 2200000;

    Pair *points = (Pair *)malloc((size_t)N_POINTS * sizeof(Pair));
    if (!points) { perror("malloc points"); return 1; }

    FILE *f = fopen("points.bin", "rb");
    if (!f) { perror("Failed to open file"); return 1; }

    size_t read = fread(points, sizeof(Pair), N_POINTS, f);
    if (read != (size_t)N_POINTS) {
        fprintf(stderr, "Failed to read all points! Only read %zu\n", read);
        return 1;
    }
    fclose(f);

    qsort(points, N_POINTS, sizeof(Pair), cmp_by_x);

    int ret_size = 0;
    Pair *hull = NULL;

    #pragma omp parallel
    {
        #pragma omp single
        {
            hull = divide_omp(points, N_POINTS, 0, &ret_size);
        }
    }

    printf("Convex Hull has %d points:\n", ret_size);
    for (int i = 0; i < ret_size; i++) {
        printf("%d %d\n", hull[i].first, hull[i].second);
    }

    free(points);
    free(hull);
    return 0;
}
