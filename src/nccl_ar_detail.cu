/*
 * NCCL collective benchmark with per-iteration CUDA event timing.
 * Supports: all_reduce, all_gather, reduce_scatter, broadcast, reduce, alltoall.
 * Outputs JSON detail array: {round, size, count, op, type, oop_time_us, ip_time_us}.
 * Validates every measured collective on the GPU and reports data_ok; exits 2 on mismatch.
 *
 * Build:
 *   nvcc -o nccl_ar_detail nccl_ar_detail.cu \
 *     -I$NCCL_HOME/include -I<mpi_include> -L$NCCL_HOME/lib -L<mpi_lib> -lnccl -lmpi -lcudart
 *
 * Usage (MPMD):
 *   mpirun --oversubscribe -np N ... ./nccl_ar_detail -o all_reduce -b 8M -e 256M -f 2 -n 5000 -w 10 -J out.json
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

static void json_open(const char *path) {
    json_fp = fopen(path, "w");
    if (!json_fp) { fprintf(stderr, "cannot open %s\n", path); return; }
    setbuf(json_fp, NULL);
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
    json_key(k); fprintf(json_fp, "%lld", v);
}

static void json_double(const char *k, double v) {
    json_key(k); fprintf(json_fp, "%.6f", v);
}

static void json_str(const char *k, const char *v) {
    json_key(k); fprintf(json_fp, "\"%s\"", v);
}

static void json_bool(const char *k, int v) {
    json_key(k); fprintf(json_fp, v ? "true" : "false");
}

static void json_obj_start(const char *k) {
    if (k) { json_key(k); }
    else   { if (!json_first_elem) fprintf(json_fp, ","); json_first_elem = 0; }
    fprintf(json_fp, "{");
    json_first_key = 1;
}

static void json_obj_end() {
    fprintf(json_fp, "}");
    json_first_key = 0;
    json_first_elem = 0;
}

static void json_arr_start(const char *k) {
    json_key(k);
    fprintf(json_fp, "[");
    json_first_elem = 1;
}

static void json_arr_end() {
    fprintf(json_fp, "]");
    json_first_key = 0;
    json_first_elem = 0;
}

static void json_obj_int(const char *k, long long v) {
    json_key(k); fprintf(json_fp, "%lld", v);
}

static void json_obj_double(const char *k, double v) {
    json_key(k); fprintf(json_fp, "%.6f", v);
}

static void json_obj_str(const char *k, const char *v) {
    json_key(k); fprintf(json_fp, "\"%s\"", v);
}

static void json_close() {
    fprintf(json_fp, "}\n");
    fclose(json_fp);
    json_fp = NULL;
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

enum { OP_ALL_REDUCE, OP_ALL_GATHER, OP_REDUCE_SCATTER, OP_BROADCAST, OP_REDUCE, OP_N };

static const char *op_names[] = {
    "all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce"
};

static int has_in_place(int op) {
    return op == OP_ALL_REDUCE || op == OP_REDUCE;
}

static ncclResult_t run_collective(int op, const float *send, float *recv,
                                   size_t count, ncclComm_t comm, cudaStream_t stream) {
    switch (op) {
    case OP_ALL_REDUCE:     return ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream);
    case OP_ALL_GATHER:     return ncclAllGather(send, recv, count, ncclFloat, comm, stream);
    case OP_REDUCE_SCATTER: return ncclReduceScatter(send, recv, count, ncclFloat, ncclSum, comm, stream);
    case OP_BROADCAST:      return ncclBroadcast(send, recv, count, ncclFloat, 0, comm, stream);
    case OP_REDUCE:         return ncclReduce(send, recv, count, ncclFloat, ncclSum, 0, comm, stream);
    default:                return ncclInvalidArgument;
    }
}

__global__ static void fill_buffer(float *data, size_t count, float value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += blockDim.x * gridDim.x) {
        data[i] = value;
    }
}

__global__ static void check_uniform(const float *data, size_t count, float expected, int *error) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += blockDim.x * gridDim.x) {
        if (data[i] != expected) atomicExch(error, 1);
    }
}

__global__ static void check_all_gather(const float *data, size_t count, int nranks, int *error) {
    size_t total = count * (size_t)nranks;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += blockDim.x * gridDim.x) {
        float expected = (float)(i / count + 1);
        if (data[i] != expected) atomicExch(error, 1);
    }
}

static int blocks_for(size_t count) {
    size_t blocks = (count + 255) / 256;
    return (int)(blocks < 65535 ? blocks : 65535);
}

static void prepare_input(int op, float *send, float *recv, size_t count,
                          int rank, int nranks, cudaStream_t stream) {
    size_t send_count = op == OP_REDUCE_SCATTER ? count * (size_t)nranks : count;
    size_t recv_count = op == OP_ALL_GATHER ? count * (size_t)nranks : count;
    fill_buffer<<<blocks_for(send_count), 256, 0, stream>>>(send, send_count, (float)(rank + 1));
    CUDACHECK(cudaGetLastError());
    fill_buffer<<<blocks_for(recv_count), 256, 0, stream>>>(recv, recv_count, -1.0f);
    CUDACHECK(cudaGetLastError());
}

static void validate_output(int op, const float *recv, size_t count, int rank,
                            int nranks, int *error, cudaStream_t stream) {
    float sum = (float)(nranks * (nranks + 1) / 2);
    if (op == OP_REDUCE && rank != 0) return;
    if (op == OP_ALL_GATHER) {
        size_t total = count * (size_t)nranks;
        check_all_gather<<<blocks_for(total), 256, 0, stream>>>(recv, count, nranks, error);
        CUDACHECK(cudaGetLastError());
    } else {
        float expected = op == OP_BROADCAST ? 1.0f : sum;
        check_uniform<<<blocks_for(count), 256, 0, stream>>>(recv, count, expected, error);
        CUDACHECK(cudaGetLastError());
    }
}

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
    size_t min_bytes = 8*1024*1024;
    size_t max_bytes = 8*1024*1024;
    int    step_factor = 2;
    int    n_iters = 20;
    int    warmup_iters = 1;
    const char *json_path = NULL;
    int    verbose = 0;
    int    op = OP_ALL_REDUCE;

    int opt;
    while ((opt = getopt(argc, argv, "b:e:f:n:w:J:o:v")) != -1) {
        switch (opt) {
        case 'b': min_bytes = parse_size(optarg); break;
        case 'e': max_bytes = parse_size(optarg); break;
        case 'f': step_factor = atoi(optarg); break;
        case 'n': n_iters = atoi(optarg); break;
        case 'w': warmup_iters = atoi(optarg); break;
        case 'J': json_path = optarg; break;
        case 'v': verbose = 1; break;
        case 'o':
            op = -1;
            for (int i = 0; i < OP_N; i++)
                if (strcmp(optarg, op_names[i]) == 0) { op = i; break; }
            if (op < 0) {
                fprintf(stderr, "Unknown op '%s'. Valid:", optarg);
                for (int i = 0; i < OP_N; i++) fprintf(stderr, " %s", op_names[i]);
                fprintf(stderr, "\n"); return 1;
            }
            break;
        default:
            fprintf(stderr, "Usage: %s [-b min] [-e max] [-f factor] [-n iters] [-w warmup] [-o op] [-J json]\n", argv[0]);
            fprintf(stderr, "  ops:");
            for (int i = 0; i < OP_N; i++) fprintf(stderr, " %s", op_names[i]);
            fprintf(stderr, "\n");
            return 1;
        }
    }

    if (min_bytes == 0 || max_bytes < min_bytes || min_bytes % sizeof(float) != 0 ||
        max_bytes % sizeof(float) != 0 || step_factor < 2 || n_iters < 1 || warmup_iters < 0) {
        fprintf(stderr, "Invalid arguments: require 0 < min <= max, float-aligned sizes, factor >= 2, "
                        "iters >= 1, and warmup >= 0\n");
        return 1;
    }

    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int n_devices;
    CUDACHECK(cudaGetDeviceCount(&n_devices));
    CUDACHECK(cudaSetDevice(mpi_rank % n_devices));

    ncclUniqueId nccl_id;
    if (mpi_rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank));

    /* max buffer = max_bytes * (mpi_size for all_gather/alltoall, else 1) */
    size_t buf_mult = (op == OP_ALL_GATHER || op == OP_REDUCE_SCATTER) ? mpi_size : 1;
    size_t max_buf = max_bytes * buf_mult;
    float *d_send, *d_recv;
    int *d_data_error;
    CUDACHECK(cudaMalloc(&d_send, max_buf));
    CUDACHECK(cudaMalloc(&d_recv, max_buf));
    CUDACHECK(cudaMalloc(&d_data_error, sizeof(*d_data_error)));
    CUDACHECK(cudaMemset(d_data_error, 0, sizeof(*d_data_error)));

    cudaEvent_t ev_start, ev_stop;
    CUDACHECK(cudaEventCreate(&ev_start));
    CUDACHECK(cudaEventCreate(&ev_stop));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    /* Warm up the selected collective so fault injection exercises that operation. */
    {
        size_t wc = min_bytes / sizeof(float);
        prepare_input(op, d_send, d_recv, wc, mpi_rank, mpi_size, stream);
        for (int i = 0; i < warmup_iters; i++)
            NCCLCHECK(run_collective(op, d_send, d_recv, wc, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    int n_sizes = 0;
    for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) n_sizes++;

    int has_ip = has_in_place(op);
    int slots_per_iter = has_ip ? 2 : 1;
    int total_slots = n_sizes * n_iters * slots_per_iter;
    double *times_us = (double *)calloc(total_slots, sizeof(double));

    int slot = 0;
    for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) {
        size_t count = sz / sizeof(float);

        /* per-size warmup */
        prepare_input(op, d_send, d_recv, count, mpi_rank, mpi_size, stream);
        for (int i = 0; i < 3; i++) {
            NCCLCHECK(run_collective(op, d_send, d_recv, count, comm, stream));
        }
        CUDACHECK(cudaStreamSynchronize(stream));

        /* ---- out-of-place ---- */
        prepare_input(op, d_send, d_recv, count, mpi_rank, mpi_size, stream);
        for (int iter = 0; iter < n_iters; iter++) {
            CUDACHECK(cudaEventRecord(ev_start, stream));
            NCCLCHECK(run_collective(op, d_send, d_recv, count, comm, stream));
            CUDACHECK(cudaEventRecord(ev_stop, stream));
            validate_output(op, d_recv, count, mpi_rank, mpi_size, d_data_error, stream);
            CUDACHECK(cudaEventSynchronize(ev_stop));
            float ms;
            CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            times_us[slot + iter] = ms * 1000.0;
        }

        /* ---- in-place (only for ops that support it) ---- */
        if (has_ip) {
            for (int iter = 0; iter < n_iters; iter++) {
                fill_buffer<<<blocks_for(count), 256, 0, stream>>>(d_recv, count, (float)(mpi_rank + 1));
                CUDACHECK(cudaGetLastError());
                CUDACHECK(cudaEventRecord(ev_start, stream));
                NCCLCHECK(run_collective(op, d_recv, d_recv, count, comm, stream));
                CUDACHECK(cudaEventRecord(ev_stop, stream));
                validate_output(op, d_recv, count, mpi_rank, mpi_size, d_data_error, stream);
                CUDACHECK(cudaEventSynchronize(ev_stop));
                float ms;
                CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                times_us[slot + n_iters + iter] = ms * 1000.0;
            }
        }

        slot += n_iters * slots_per_iter;
    }

    CUDACHECK(cudaStreamSynchronize(stream));
    int local_data_error = 0;
    CUDACHECK(cudaMemcpy(&local_data_error, d_data_error, sizeof(local_data_error), cudaMemcpyDeviceToHost));
    int local_data_ok = local_data_error == 0;
    int data_ok = 0;
    MPI_Allreduce(&local_data_ok, &data_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    /* gather timings to rank 0 */
    double *max_times = NULL;
    double t_start = 0;
    if (mpi_rank == 0) {
        max_times = (double *)calloc(total_slots, sizeof(double));
        t_start = now_us();
    }
    MPI_Reduce(times_us, max_times, total_slots, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double t_end = now_us();

    /* ---- JSON ---- */
    if (mpi_rank == 0 && json_path) json_open(json_path);
    if (mpi_rank == 0 && json_fp) {
        json_int("nccl_version", NCCL_VERSION_CODE);
        json_int("n_ranks", mpi_size);
        json_str("op", op_names[op]);
        json_int("min_bytes", (long long)min_bytes);
        json_int("max_bytes", (long long)max_bytes);
        json_int("step_factor", step_factor);
        json_int("n_iters", n_iters);
        json_int("warmup_iters", warmup_iters);
        json_double("wall_time_us", t_end - t_start);
        json_bool("data_ok", data_ok);

        json_arr_start("detail");
        int round = 0, sidx = 0;
        for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) {
            size_t count = sz / sizeof(float);
            int base = sidx * n_iters * slots_per_iter;

            for (int iter = 0; iter < n_iters; iter++) {
                double t_oop = max_times[base + iter];
                double t_ip  = has_ip ? max_times[base + n_iters + iter] : 0.0;

                json_obj_start(NULL);
                json_obj_int("round", round++);
                json_obj_int("size", (long long)sz);
                json_obj_int("count", (long long)count);
                json_obj_str("op", op_names[op]);
                json_obj_str("type", "float");
                json_obj_double("oop_time_us", t_oop);
                json_obj_double("ip_time_us", t_ip);
                json_obj_end();
            }
            sidx++;
        }
        json_arr_end();
        json_close();
    }

    /* ---- summary table ---- */
    if (mpi_rank == 0 && verbose) {
        printf("# %s  size         count    oop_avg_us  oop_bw", op_names[op]);
        if (has_ip) printf("    ip_avg_us   ip_bw");
        printf("\n");

        int sidx = 0;
        for (size_t sz = min_bytes; sz <= max_bytes; sz *= step_factor) {
            size_t count = sz / sizeof(float);
            int base = sidx * n_iters * slots_per_iter;

            double sum_oop = 0, sum_ip = 0;
            for (int i = 0; i < n_iters; i++) sum_oop += max_times[base + i];
            if (has_ip) for (int i = 0; i < n_iters; i++) sum_ip += max_times[base + n_iters + i];

            double data_moved = (double)sz;
            if (op == OP_ALL_GATHER || op == OP_REDUCE_SCATTER)
                data_moved *= mpi_size;
            if (op == OP_ALL_REDUCE) data_moved *= 2;
            double bw_oop = data_moved / 1e9 / (sum_oop / n_iters * 1e-6) * (mpi_size - 1.0) / mpi_size;

            printf("  %12zu %10zu  %10.1f  %6.2f", sz, count, sum_oop/n_iters, bw_oop);
            if (has_ip) {
                double bw_ip = data_moved / 1e9 / (sum_ip / n_iters * 1e-6) * (mpi_size - 1.0) / mpi_size;
                printf("  %10.1f  %6.2f", sum_ip/n_iters, bw_ip);
            }
            printf("\n");
            sidx++;
        }
    }

    free(times_us);
    free(max_times);
    CUDACHECK(cudaEventDestroy(ev_start));
    CUDACHECK(cudaEventDestroy(ev_stop));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_send));
    CUDACHECK(cudaFree(d_recv));
    CUDACHECK(cudaFree(d_data_error));
    ncclCommDestroy(comm);
    MPI_Finalize();
    if (!data_ok && mpi_rank == 0) fprintf(stderr, "Data validation failed\n");
    return data_ok ? 0 : 2;
}
