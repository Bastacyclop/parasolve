#include "projet.h"
#include <mpi.h>
#include <time.h>

unsigned long long int node_searched = 0;

#define TASK_DEPTH 3

const int TAG_DATA = 0;
const int TAG_TASK = 1;
const int TAG_STOP = 2;

typedef struct {
    int rank;
    int procs;
    MPI_Datatype tree_type;
    MPI_Datatype result_type;
} Env;

void evaluate(const Env* env, tree_t *T, result_t *result) {
#pragma omp atomic
    node_searched++;

    move_t moves[MAX_MOVES];
    int n_moves;

    result->score = -MAX_SCORE - 1;
    result->pv_length = 0;

    if (test_draw_or_victory(T, result))
        return;

    if (TRANSPOSITION_TABLE && tt_lookup(T, result)) /* la réponse est-elle déjà connue ? */
        return;

    compute_attack_squares(T);

    /* profondeur max atteinte ? si oui, évaluation heuristique */
    if (T->depth == 0) {
        result->score = (2 * T->side - 1) * heuristic_evaluation(T);
        return;
    }

    n_moves = generate_legal_moves(T, &moves[0]);

    /* absence de coups légaux : pat ou mat */
    if (n_moves == 0) {
        result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
        return;
    }

    if (ALPHA_BETA_PRUNING)
        sort_moves(T, n_moves, moves);

    if (env->rank == 0 && n_moves >= (env->procs - 1)) {
        printf("delegating %i moves\n", n_moves);
        move_t* pmoves = malloc((env->procs - 1) * sizeof(move_t));

        for (int p = 1; p < env->procs; p++) {
            int m = p - 1;
            tree_t child;
            play_move(T, moves[m], &child);
            MPI_Send(&child, 1, env->tree_type, p, TAG_TASK, MPI_COMM_WORLD);
            pmoves[m] = moves[m];
        }

        for (int i = 0; i < n_moves; i++) {
            result_t child_result;
            MPI_Status status;
            MPI_Recv(&child_result, 1, env->result_type, MPI_ANY_SOURCE, TAG_DATA, MPI_COMM_WORLD, &status);

            int child_score = -child_result.score;
            move_t* pm = &pmoves[status.MPI_SOURCE - 1];

            if (child_score > result->score) {
                result->score = child_score;
                result->best_move = *pm;
                result->pv_length = child_result.pv_length + 1;
                for (int j = 0; j < child_result.pv_length; j++)
                    result->PV[j + 1] = child_result.PV[j];
                result->PV[0] = *pm;
            }

            T->alpha = MAX(T->alpha, child_score);

            if (i <= (n_moves - env->procs)) {
                int m = i + env->procs - 1;
                tree_t child;
                play_move(T, moves[m], &child);
                MPI_Send(&child, 1, env->tree_type, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                *pm = moves[m];
            }
        }
    } else {
        for (int i = 0; i < n_moves; i++) {
            tree_t child;
            result_t child_result;

            play_move(T, moves[i], &child);

            evaluate(env, &child, &child_result);

            int child_score = -child_result.score;

            if (child_score > result->score) {
                result->score = child_score;
                result->best_move = moves[i];
                result->pv_length = child_result.pv_length + 1;
                for (int j = 0; j < child_result.pv_length; j++)
                    result->PV[j + 1] = child_result.PV[j];
                result->PV[0] = moves[i];
            }

            if (ALPHA_BETA_PRUNING && child_score >= T->beta)
                break;

            T->alpha = MAX(T->alpha, child_score);
        }
    }

    if (TRANSPOSITION_TABLE)
        tt_store(T, result);
}

void decide(const Env* env, tree_t *T, result_t *result) {
    for (int depth = 1;; depth++) {
        T->depth = depth;
        T->height = 0;
        T->alpha_start = T->alpha = -MAX_SCORE - 1;
        T->beta = MAX_SCORE + 1;

        printf("=====================================\n");
        evaluate(env, T, result);

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

    time_t marker = time(NULL);

    if (e.rank == 0) {
        if (argc < 2) {
            printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
            exit(1);
        }

        if (ALPHA_BETA_PRUNING)
            printf("Alpha-beta pruning ENABLED\n");

        if (TRANSPOSITION_TABLE) {
            printf("Transposition table ENABLED\n");
            init_tt();
        }

        tree_t root;
        parse_FEN(argv[1], &root);
        print_position(&root);

        result_t result;
        decide(&e, &root, &result);

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

        printf("Node searched: %llu\n", node_searched);

        if (TRANSPOSITION_TABLE)
            free_tt();
    } else {
        MPI_Status status;
        tree_t tree;
        result_t result;
        printf("worker %i up\n", e.rank);
        while (1) {
            MPI_Recv(&tree, 1, e.tree_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP) { break; }
            evaluate(&e, &tree, &result);
            MPI_Send(&result, 1, e.result_type, 0, TAG_DATA, MPI_COMM_WORLD);
        }
        printf("worker %i down, searched %llu nodes\n", e.rank, node_searched);
    }

    printf("execution time (%i): %li\n", e.rank, time(NULL) - marker);
  
    MPI_Finalize();
    return EXIT_SUCCESS;
}
