#define _POSIX_C_SOURCE 200809L

#include "projet.h"
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <math.h>

unsigned long long int t_node_searched = 0;
#pragma omp threadprivate(t_node_searched)
int task_depth;

typedef struct timespec Time;
void get_time(Time* t) {
    clock_gettime(CLOCK_MONOTONIC, t);
}

double seconds_from(const Time* t) {
    Time now;
    get_time(&now);
    return (double)(now.tv_sec - t->tv_sec) +
        (double)(now.tv_nsec - t->tv_nsec) / 1000000000.;
}

const int TAG_DATA = 0;
const int TAG_TASK = 1;
const int TAG_STOP = 2;

typedef struct {
    int rank;
    int procs;
    MPI_Datatype tree_type;
    MPI_Datatype result_type;
} Env;

void evaluate_master(const Env* env, tree_t* T, result_t* result) {
    t_node_searched++;

    move_t moves[MAX_MOVES];
    int n_moves;

    result->score = -MAX_SCORE - 1;
    result->pv_length = 0;

    if (test_draw_or_victory(T, result)) {
        return;
    }

    if (TRANSPOSITION_TABLE && tt_lookup(T, result)) {
        return;
    }

    compute_attack_squares(T);

    if (T->depth == 0) {
        result->score = (2 * T->side - 1) * heuristic_evaluation(T);
        return;
    }

    n_moves = generate_legal_moves(T, moves);

    if (n_moves == 0) {
        result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
        return;
    }

    if (ALPHA_BETA_PRUNING)
        sort_moves(T, n_moves, moves);

    int m = 0;
    {
        tree_t child;
        result_t child_result;

        play_move(T, moves[m], &child);
        evaluate_master(env, &child, &child_result);

        int child_score = -child_result.score;

        if (child_score > result->score) {
            result->score = child_score;
            result->best_move = moves[m];
            result->pv_length = child_result.pv_length + 1;
            for (int j = 0; j < child_result.pv_length; j++)
                result->PV[j + 1] = child_result.PV[j];
            result->PV[0] = moves[m];
        }

        if (ALPHA_BETA_PRUNING && child_score >= T->beta)
            return;

        T->alpha = MAX(T->alpha, child_score);
        m++;
    }
    int seq_end = m;

    int workers = env->procs - 1;
    int worker_moves = n_moves - seq_end;
    if (worker_moves < workers) {
        workers = worker_moves;
    }

    move_t* pmoves = malloc(workers * sizeof(move_t));

    for (int p = 1; p <= workers; p++) {
        tree_t child;
        play_move(T, moves[m], &child);
        MPI_Send(&child, 1, env->tree_type, p, TAG_TASK, MPI_COMM_WORLD);
        pmoves[p - 1] = moves[m];
        m++;
    }

    for (int i = 0; i < worker_moves; i++) {
        result_t child_result;
        MPI_Status status;
        MPI_Recv(&child_result, 1, env->result_type,
                 MPI_ANY_SOURCE, TAG_DATA, MPI_COMM_WORLD, &status);

        int child_score = -child_result.score;
        move_t* move = &pmoves[status.MPI_SOURCE - 1];

        if (child_score > result->score) {
            result->score = child_score;
            result->best_move = *move;
            result->pv_length = child_result.pv_length + 1;
            for (int j = 0; j < child_result.pv_length; j++)
                result->PV[j + 1] = child_result.PV[j];
            result->PV[0] = *move;
        }

        if (ALPHA_BETA_PRUNING && result->score >= T->beta) {
            continue;
        }

        T->alpha = MAX(T->alpha, child_score);

        if (m < n_moves) {
            tree_t child;
            play_move(T, moves[m], &child);
            MPI_Send(&child, 1, env->tree_type, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
            *move = moves[m];
            m++;
        }
    }

    free(pmoves);
}

int assimilate(tree_t* T, result_t* result,
               move_t move, result_t* child_result) {
    int child_score = -child_result->score;

    if (child_score > result->score) {
        result->score = child_score;
        result->best_move = move;
        result->pv_length = child_result->pv_length + 1;
        for (int j = 0; j < child_result->pv_length; j++)
            result->PV[j + 1] = child_result->PV[j];
        result->PV[0] = move;
    }

    int alpha = MAX(T->alpha, child_score);
#pragma omp atomic write
    T->alpha = alpha;
    return ALPHA_BETA_PRUNING && alpha >= T->beta;
}

void evaluate(tree_t* T, result_t* result) {
    t_node_searched++;

    move_t moves[MAX_MOVES];
    int n_moves;

    result->score = -MAX_SCORE - 1;
    result->pv_length = 0;

    if (test_draw_or_victory(T, result))
        return;

    if (TRANSPOSITION_TABLE) {
        int available;
#pragma omp critical
        available = tt_lookup(T, result);
        if (available) return;
    }
        
    compute_attack_squares(T);

    if (T->depth == 0) {
        result->score = (2 * T->side - 1) * heuristic_evaluation(T);
        return;
    }

    n_moves = generate_legal_moves(T, &moves[0]);

    if (n_moves == 0) {
        result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
        return;
    }

    if (ALPHA_BETA_PRUNING)
        sort_moves(T, n_moves, moves);

    if (T->height <= task_depth) {
        tree_t elder;
        result_t elder_result;
        play_move(T, moves[0], &elder);
        evaluate(&elder, &elder_result);

        if (!assimilate(T, result, moves[0], &elder_result)) {
            omp_lock_t lock;
            omp_init_lock(&lock);
            for (int i = 1; i < n_moves; i++) {
#pragma omp task
                {
                    tree_t child;
                    result_t child_result;
                    play_move(T, moves[i], &child);
                    evaluate(&child, &child_result);

                    omp_set_lock(&lock);
                    assimilate(T, result, moves[i], &child_result);
                    omp_unset_lock(&lock);
                }
            }
#pragma omp taskwait
            omp_destroy_lock(&lock);
        }
    } else {
        for (int i = 0; i < n_moves; i++) {
            tree_t child;
            result_t child_result;
            play_move(T, moves[i], &child);
            evaluate(&child, &child_result);

            if (assimilate(T, result, moves[i], &child_result))
                break;    
        }
    }

    if (TRANSPOSITION_TABLE) {
#pragma omp critical
        tt_store(T, result);
    }
}

void decide_master(Env* env, tree_t* T, result_t* result) {
    for (int depth = 1;; depth++) {
        T->depth = depth;
        T->height = 0;
        T->alpha_start = T->alpha = -MAX_SCORE - 1;
        T->beta = MAX_SCORE + 1;

        printf("=====================================\n");
        evaluate_master(env, T, result);

        printf("depth: %d / score: %.2f / best_move : ", T->depth, 0.01 * result->score);
        print_pv(T, result);

        if (DEFINITIVE(result->score))
            break;
    }

    for (int p = 1; p < env->procs; p++) {
        MPI_Send(NULL, 0, env->tree_type, p, TAG_STOP, MPI_COMM_WORLD);
    }
}

void commit_result_datatype(MPI_Datatype* type) {
#define field_count 4

    int blocklengths[field_count] = {
        1,
        1,
        1,
        MAX_DEPTH
    };
    MPI_Aint displacements[field_count] = {
        offsetof(result_t, score),
        offsetof(result_t, best_move),
        offsetof(result_t, pv_length),
        offsetof(result_t, PV),
    };
    MPI_Datatype types[field_count] = {
        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_INT
    };

    MPI_Type_create_struct(
        field_count,
        blocklengths,
        displacements,
        types,
        type
        );
    MPI_Type_commit(type);

#undef field_count
}

void commit_tree_datatype(MPI_Datatype* type) {
#define field_count 14

    int blocklengths[field_count] = {
        128,
        128,
        1,

        1,
        1,
        1,
        1,
        1,

        2,
        2,
        128,

        1,
        1,
        MAX_DEPTH,
    };
    MPI_Aint displacements[field_count] = {
        offsetof(tree_t, pieces),
        offsetof(tree_t, colors),
        offsetof(tree_t, side),

        offsetof(tree_t, depth),
        offsetof(tree_t, height),
        offsetof(tree_t, alpha),
        offsetof(tree_t, beta),
        offsetof(tree_t, alpha_start),

        offsetof(tree_t, king),
        offsetof(tree_t, pawns),
        offsetof(tree_t, attack),

        offsetof(tree_t, suggested_move),
        offsetof(tree_t, hash),
        offsetof(tree_t, history),
    };
    MPI_Datatype types[field_count] = {
        MPI_INT,
        MPI_INT,
        MPI_INT,

        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_INT,

        MPI_INT,
        MPI_INT,
        MPI_INT,

        MPI_INT,
        MPI_UNSIGNED_LONG_LONG,
        MPI_UNSIGNED_LONG_LONG,
    };

    MPI_Type_create_struct(
        field_count,
        blocklengths,
        displacements,
        types,
        type
        );
    MPI_Type_commit(type);

#undef field_count
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    Env e;
    MPI_Comm_rank(MPI_COMM_WORLD, &e.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &e.procs);
    commit_tree_datatype(&e.tree_type);
    commit_result_datatype(&e.result_type);

    Time start;
    get_time(&start);

    if (TRANSPOSITION_TABLE)
        init_tt();

    int branching = 16;
    if (ALPHA_BETA_PRUNING) {
        if (TRANSPOSITION_TABLE) {
            branching = 8;
        } else {
            branching = 12;
        }
    }
    int task_depth_relative = log(3500 * omp_get_max_threads()) / log(branching);
    if (argc >= 3) {
        task_depth_relative = atoi(argv[2]);
    }
   
    if (e.rank == 0) {
        if (argc < 2) {
            printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
            exit(1);
        }

        if (ALPHA_BETA_PRUNING)
            printf("Alpha-beta pruning ENABLED\n");

        if (TRANSPOSITION_TABLE)
            printf("Transposition table ENABLED\n");

        tree_t root;
        parse_FEN(argv[1], &root);
        print_position(&root);

        result_t result;
        decide_master(&e, &root, &result);

        printf("\nDécision de la position: ");
        switch (result.score * (2 * root.side - 1)) {
            case MAX_SCORE:
                printf("blanc gagne\n");
                break;
            case CERTAIN_DRAW:
                printf("partie nulle\n");
                break;
            case -MAX_SCORE:
                printf("noir gagne\n");
                break;
            default:
                printf("BUG\n");
        }

        printf("master down, searched %llu nodes\n", t_node_searched);
        printf("execution time: %lf\n", seconds_from(&start));
    } else {
        MPI_Status status;
        tree_t tree;
        result_t result;
        printf("worker %i up\n", e.rank);
        Time idle_start;
        get_time(&idle_start);
        double idle_time = 0;
        while (1) {
            MPI_Recv(&tree, 1, e.tree_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            idle_time += seconds_from(&idle_start);
            if (status.MPI_TAG == TAG_STOP) { break; }
            task_depth = task_depth_relative + tree.height;
#pragma omp parallel
#pragma omp single
            evaluate(&tree, &result);
            get_time(&idle_start);
            MPI_Send(&result, 1, e.result_type, 0, TAG_DATA, MPI_COMM_WORLD);
        }

        double execution_time = seconds_from(&start);
        double work_time = execution_time - idle_time;

        unsigned long long int node_searched = 0;
#pragma omp parallel reduction(+:node_searched)
        node_searched += t_node_searched;

        double speed = (double)(node_searched) / work_time;
        printf("worker %i down, searched %llu nodes, %lf execution time (%lf work + %lf idle), speed: %lf node/s\n",
               e.rank, node_searched, execution_time, work_time, idle_time, speed);

#pragma omp parallel
        {
            printf("worker %i node searched: %llu\n", e.rank, t_node_searched);
        }
    }

    MPI_Finalize();
    if (TRANSPOSITION_TABLE)
        free_tt();
    return EXIT_SUCCESS;
}
