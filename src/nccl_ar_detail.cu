/*
 * Minimal NCCL all_reduce benchmark with per-iteration CUDA event timing.
 * Outputs JSON with a "detail" array: per-round time, size, count, type, redop, oop/ip.
 *
 * Build (on remote):
 *   nvcc -o nccl_ar_detail nccl_ar_detail.cu \
 *     -I$NCCL_HOME/include -I$MPI_HOME/include \
 *     -L$NCCL_HOME/lib -L$MPI_HOME/lib -lnccl -lmpi -lcudart
 *
 * Usage (MPMD, same as before):
 *   mpirun -np 1 -H 112:1 /usr/bin/env ... ./nccl_ar_detail \
 *     -b 8M -e 256M -f 2 -n 5000 -w 10 -J /tmp/out.json \
 *   : -np 1 -H 113:1 ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <sys/time.h>

/* ---- JSON helpers (minimal, no dependencies) ---- */
static FILE *json_fp = NULL;
static int json_first_key = 1;
static int json_first_elem = 1;
static int json_in_object = 0;

static void json_open(const char *path) {
    json_fp = fopen(path, "w");
    if (!json_fp) { fprintf(stderr, "cannot open %s\n", path); return; }
    fprintf(json_fp, "{");
    json_first_key = 1;
}

static void json_key(const char *k) {
    if (!json_fp) return;
    if (!json_first_key) fprintf(json_fp, ",");
    fprintf(json_fp, "\"%s\":", k);
    json_first_key = 0;
}

static void json_int(const char *k, long long v) {
    if (!json_fp) return;
    json_key(k);
    fprintf(json_fp, "%lld", v);
}

static void json_double(const char *k, double v) {
    if (!json_fp) return;
    json_key(k);
    fprintf(json_fp, "%.6f", v);
}

static void json_str(const char *k, const char *v) {
    if (!json_fp) return;
    json_key(k);
    fprintf(json_fp, "\"%s\"", v);
}

static void json_bool(const char *k, int v) {
    if (!json_fp) return;
    json_key(k);
    fprintf(json_fp, v ? "true" : "false");
}

static void json_obj_start(const char *k) {
    if (!json_fp) return;
    if (k) { json_key(k); }
    else   { if (!json_first_elem) fprintf(json_fp, ","); json_first_elem = 0; }
    fprintf(json_fp, "{");
    json_first_key = 1;
}

static void json_obj_end() {
    if (!json_fp) return;
    fprintf(json_fp, "}");
    json_first_key = 0;
    json_first_elem = 0;
}

static void json_arr_start(const char *k) {
    if (!json_fp) return;
    if (k) { json_key(k); }
    else   { if (!json_first_elem) fprintf(json_fp, ","); json_first_elem = 0; }
    fprintf(json_fp, "[");
    json_first_elem = 1;
}

static void json_arr_end() {
    if (!json_fp) return;
    fprintf(json_fp, "]");
    json_first_key = 0;
}

static void json_arr_int(long long v) {
    if (!json_fp) return;
    if (!json_first_elem) fprintf(json_fp, ",");
    fprintf(json_fp, "%lld", v);
    json_first_elem = 0;
}

static void json_arr_double(double v) {
    if (!json_fp) return;
    if (!json_first_elem) fprintf(json_fp, ",");
    fprintf(json_fp, "%.6f", v);
    json_first_elem = 0;
}

static void json_close() {
    if (!json_fp) return;
    fprintf(json_fp, "}\n");
    fclose(json_fp);
    json_fp = NULL;
}

static void json_obj_int(const char *k, long long v) {
    json_obj_start(k); json_int("v", v); json_obj_end();
}

/* ---- Main ---- */

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                            \
    if (e != cudaSuccess) {                         \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",  \
            __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                    \
    }                                               \
} while(0)

#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                           \
    if (r != ncclSuccess) {                         \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",  \
            __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1);                                    \
    }                                               \
} while(0)

static double now_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static size_t parse_size(const char *s) {
    char *end;
    size_t v = strtoull(s, &end, 0);
    if (*end == 'K' || *end == 'k') v *= 1024;
    else if (*end == 'M' || *end == 'm') v *= 1024*1024;
    else if (*end == 'G' || *end == 'g') v *= 1024*1024*1024;
    return v;
}

int main(int argc, char *argv[]) {
    /* ---- defaults ---- */
    size_t min_bytes = 8*1024*1024;
    size_t max_bytes = 8*1024*1024;
    int    step_factor = 2;
    int    n_iters = 20;
    int    warmup_iters = 1;
    const char *json_path = NULL;
    int    verbose = 0;

    /* ---- parse args ---- */
    int opt;
    while ((opt = getopt(argc, argv, "b:e:f:n:w:J:v")) != -1) {
        switch (opt) {
        case 'b': min_bytes = parse_size(optarg); break;
        case 'e': max_bytes = parse_size(optarg); break;
        case 'f': step_factor = atoi(optarg); break;
        case 'n': n_iters = atoi(optarg); break;
        case 'w': warmup_iters = atoi(optarg); break;
        case 'J': json_path = optarg; break;
        case 'v': verbose = 1; break;
        default:
            fprintf(stderr, "Usage: %s [-b min] [-e max] [-f factor] [-n iters] [-w warmup] [-J json] [-v]\n", argv[0]);
            return 1;
        }
    }

    /* ---- MPI init ---- */
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    /* ---- CUDA init ---- */
    int n_devices;
    CUDACHECK(cudaGetDeviceCount(&n_devices));
    int dev = mpi_rank % n_devices;
    CUDACHECK(cudaSetDevice(dev));

    /* ---- NCCL init ---- */
    ncclUniqueId nccl_id;
    if (mpi_rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank));

    /* ---- allocate buffers (max size) ---- */
    float *d_send, *d_recv;
    CUDACHECK(cudaMalloc(&d_send, max_bytes));
    CUDACHECK(cudaMalloc(&d_recv, max_bytes));

    /* ---- CUDA events ---- */
    cudaEvent_t ev_start, ev_stop;
    CUDACHECK(cudaEventCreate(&ev_start));
    CUDACHECK(cudaEventCreate(&ev_stop));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    /* ---- JSON open (rank 0 only) ---- */
    if (mpi_rank == 0 && json_path) json_open(json_path);

    /* ignore warmup: do some simple all_reduce to init connections */
    {
        size_t wsize = min_bytes;
        size_t count = wsize / sizeof(float);
        for (int i = 0; i < warmup_iters; i++) {
            NCCLCHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    /* ---- benchmark loop ---- */
    int n_sizes = 0;
    for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) n_sizes++;

    /* collect timings: [size_idx * n_iters * 2 + iter * 2 + {0=oop,1=ip}] */
    int total_slots = n_sizes * n_iters * 2;
    double *times_us = (double *)calloc(total_slots, sizeof(double));

    int slot = 0;
    for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) {
        size_t count = sz / sizeof(float);

        /* warmup for this size */
        for (int i = 0; i < 3; i++) {
            NCCLCHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDACHECK(cudaStreamSynchronize(stream));

        /* ---- out-of-place timing ---- */
        for (int iter = 0; iter < n_iters; iter++) {
            CUDACHECK(cudaEventRecord(ev_start, stream));
            NCCLCHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
            CUDACHECK(cudaEventRecord(ev_stop, stream));
            CUDACHECK(cudaEventSynchronize(ev_stop));
            float ms;
            CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            times_us[slot + iter] = ms * 1000.0;
        }

        /* ---- in-place timing (recv == send) ---- */
        for (int iter = 0; iter < n_iters; iter++) {
            CUDACHECK(cudaEventRecord(ev_start, stream));
            NCCLCHECK(ncclAllReduce(d_recv, d_recv, count, ncclFloat, ncclSum, comm, stream));
            CUDACHECK(cudaEventRecord(ev_stop, stream));
            CUDACHECK(cudaEventSynchronize(ev_stop));
            float ms;
            CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            times_us[slot + n_iters + iter] = ms * 1000.0;
        }

        slot += n_iters * 2;
    }

    /* ---- gather timings to rank 0 (max across ranks per slot) ---- */
    double *max_times = NULL;
    double t_start = 0;
    if (mpi_rank == 0) {
        max_times = (double *)calloc(total_slots, sizeof(double));
        t_start = now_us();
    }

    MPI_Reduce(times_us, max_times, total_slots, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double t_end = now_us();

    /* ---- output JSON (rank 0) ---- */
    if (mpi_rank == 0 && json_path && json_fp) {
        json_int("nccl_version", NCCL_VERSION_CODE);
        json_int("n_ranks", mpi_size);
        json_int("min_bytes", (long long)min_bytes);
        json_int("max_bytes", (long long)max_bytes);
        json_int("step_factor", step_factor);
        json_int("n_iters", n_iters);
        json_int("warmup_iters", warmup_iters);
        json_double("wall_time_us", t_end - t_start);
        json_bool("data_ok", 1);

        json_arr_start("results");
        int sidx = 0;
        for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) {
            size_t count = sz / sizeof(float);

            json_obj_start(NULL);

            json_int("size", (long long)sz);
            json_int("count", (long long)count);
            json_str("type", "float");
            json_str("redop", "sum");

            /* out-of-place summary */
            double sum_oop = 0, min_oop = 1e18, max_oop = 0;
            json_arr_start("out_of_place_us");
            int base = sidx * n_iters * 2;
            for (int i = 0; i < n_iters; i++) {
                double t = max_times[base + i];
                json_arr_double(t);
                sum_oop += t;
                if (t < min_oop) min_oop = t;
                if (t > max_oop) max_oop = t;
            }
            json_arr_end();

            json_obj_start("out_of_place_stats");
            json_double("avg_us", sum_oop / n_iters);
            json_double("min_us", min_oop);
            json_double("max_us", max_oop);
            json_obj_end();

            double bw_oop = (double)sz / 1e9 / (sum_oop / n_iters * 1e-6) * 2 * (mpi_size - 1) / mpi_size;
            json_double("out_of_place_bus_bw", bw_oop);

            /* in-place summary */
            double sum_ip = 0, min_ip = 1e18, max_ip = 0;
            json_arr_start("in_place_us");
            for (int i = 0; i < n_iters; i++) {
                double t = max_times[base + n_iters + i];
                json_arr_double(t);
                sum_ip += t;
                if (t < min_ip) min_ip = t;
                if (t > max_ip) max_ip = t;
            }
            json_arr_end();

            json_obj_start("in_place_stats");
            json_double("avg_us", sum_ip / n_iters);
            json_double("min_us", min_ip);
            json_double("max_us", max_ip);
            json_obj_end();

            double bw_ip = (double)sz / 1e9 / (sum_ip / n_iters * 1e-6) * 2 * (mpi_size - 1) / mpi_size;
            json_double("in_place_bus_bw", bw_ip);

            json_obj_end(); /* result object */
            sidx++;
        }
        json_arr_end();
        json_close();
    }

    /* print summary to stdout */
    if (mpi_rank == 0) {
        printf("# size         count    oop_avg_us  oop_bw    ip_avg_us   ip_bw\n");
        int sidx = 0;
        for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) {
            size_t count = sz / sizeof(float);
            int base = sidx * n_iters * 2;
            double sum_oop = 0, sum_ip = 0;
            for (int i = 0; i < n_iters; i++) {
                sum_oop += max_times[base + i];
                sum_ip += max_times[base + n_iters + i];
            }
            double bw_oop = (double)sz / 1e9 / (sum_oop / n_iters * 1e-6) * 2 * (mpi_size - 1) / mpi_size;
            double bw_ip = (double)sz / 1e9 / (sum_ip / n_iters * 1e-6) * 2 * (mpi_size - 1) / mpi_size;
            printf("  %10zu %10zu  %10.1f  %6.2f  %10.1f  %6.2f\n",
                   sz, count, sum_oop/n_iters, bw_oop, sum_ip/n_iters, bw_ip);
            sidx++;
        }
    }

    /* ---- cleanup ---- */
    free(times_us);
    free(max_times);
    CUDACHECK(cudaEventDestroy(ev_start));
    CUDACHECK(cudaEventDestroy(ev_stop));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_send));
    CUDACHECK(cudaFree(d_recv));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
