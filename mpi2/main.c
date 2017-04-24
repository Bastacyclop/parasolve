#include "projet.h"
#include <mpi.h>
#include <time.h>

unsigned long long int node_searched = 0;

const int TAG_DATA = 0;
const int TAG_TASK = 1;
const int TAG_STOP = 2;

typedef struct Task Task;
typedef struct Node Node;

typedef struct {
    int rank;
    int procs;
    MPI_Datatype tree_type;
    MPI_Datatype result_type;
    // Master only:
    int idle_workers;
    Task* worker_tasks;
    int depth;
} Env;

struct Task {
    move_t move;
    Node* parent;
};

struct Node {
    tree_t T;
    result_t result;
    int countdown;
    move_t move; // undefined if root
    Node* parent; // null if root
};

void transmit(Node* node, move_t move, result_t* result) {
    node->countdown--;
    int score = -result->score;
    if (score > node->result.score) {
        node->result.score = score;
        node->result.best_move = move;
        node->result.pv_length = result->pv_length + 1;
        for (int j = 0; j < result->pv_length; j++)
            node->result.PV[j + 1] = result->PV[j];
        result->PV[0] = move;
    }

    // TODO: alpha-beta?

    if (node->countdown == 0 && node->parent) {
        // TODO: transposition table?
        transmit(node->parent, node->move, &node->result);
        free(node);
    }
}

int next_worker(Env* env) {
    int p;
    if (env->idle_workers > 0) {
        p = env->procs - env->idle_workers;
        env->idle_workers--;
    } else {
        result_t result;
        MPI_Status status;
        MPI_Recv(&result, 1, env->result_type, MPI_ANY_SOURCE,
                 TAG_DATA, MPI_COMM_WORLD, &status);
        p = status.MPI_SOURCE;
        Task task = env->worker_tasks[p - 1];
        transmit(task.parent, task.move, &result);
    }
    return p;
}

void workers_barrier(Env* env) {
    int workers = (env->procs - 1);
    for (int w = env->idle_workers; w < workers; w++) {
        next_worker(env);
    }
    env->idle_workers = workers;
}

void evaluate_master(Env* env, Node* node) {
    node_searched++;

    tree_t* T = &node->T;
    result_t* result = &node->result;
    move_t moves[MAX_MOVES];
    int n_moves;

    result->score = -MAX_SCORE - 1;
    result->pv_length = 0;
    node->countdown = 0;

    if (test_draw_or_victory(T, result)) {
        if (node->parent) transmit(node->parent, node->move, &node->result);
        return;
    }

    if (TRANSPOSITION_TABLE && tt_lookup(T, result)) {
        if (node->parent) transmit(node->parent, node->move, &node->result);
        return;
    }

    compute_attack_squares(T);

    if (T->depth == 0) {
        result->score = (2 * T->side - 1) * heuristic_evaluation(T);
        if (node->parent) transmit(node->parent, node->move, &node->result);
        return;
    }

    n_moves = generate_legal_moves(T, &moves[0]);
    node->countdown = n_moves;

    if (n_moves == 0) {
        result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
        if (node->parent) transmit(node->parent, node->move, &node->result);
        return;
    }

    if (ALPHA_BETA_PRUNING)
        sort_moves(T, n_moves, moves);

    if (T->height <= env->depth) {
        for (int i = 0; i < n_moves; i++) {
            Node* child = malloc(sizeof(Node));
            child->move = moves[i];
            child->parent = node;
            play_move(T, moves[i], &child->T);
            evaluate_master(env, child);
        }
    } else {
        /* TODO: somewhere ?
        if (ALPHA_BETA_PRUNING && result->score >= T->beta)
            continue;
        */

        for (int i = 0; i < n_moves; i++) {
            tree_t child;
            int p = next_worker(env);
            play_move(T, moves[i], &child);
            MPI_Send(&child, 1, env->tree_type, p, TAG_TASK, MPI_COMM_WORLD);
            Task* task = &env->worker_tasks[p - 1];
            task->move = moves[i];
            task->parent = node;
        }
    }
}

void evaluate(tree_t *T, result_t *result) {
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

    for (int i = 0; i < n_moves; i++) {
        tree_t child;
        result_t child_result;

        play_move(T, moves[i], &child);
        evaluate(&child, &child_result);

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

    if (TRANSPOSITION_TABLE)
        tt_store(T, result);
}

void decide_master(Env* env, Node* root) {
    tree_t* T = &root->T;
    result_t* result = &root->result;

    for (int depth = 1;; depth++) {
        T->depth = depth;
        T->height = 0;
        T->alpha_start = T->alpha = -MAX_SCORE - 1;
        T->beta = MAX_SCORE + 1;

        printf("=====================================\n");
        // TODO only if > env->depth?
        evaluate_master(env, root);
        workers_barrier(env);
        if (root->countdown != 0) {
            printf("COUNTDOWN :/\n");
            exit(EXIT_FAILURE);
        }

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

    if (TRANSPOSITION_TABLE)
        init_tt();

    if (e.rank == 0) {
        e.idle_workers = (e.procs - 1);
        e.worker_tasks = malloc(e.idle_workers * sizeof(Task));
        int tasks_per_worker = 4;
        e.depth = 3;

        if (argc < 2) {
            printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
            exit(1);
        }

        if (ALPHA_BETA_PRUNING)
            printf("Alpha-beta pruning ENABLED\n");

        if (TRANSPOSITION_TABLE)
            printf("Transposition table ENABLED\n");

        Node root;
        root.parent = NULL;
        parse_FEN(argv[1], &root.T);
        print_position(&root.T);

        decide_master(&e, &root);

        printf("\nDécision de la position: ");
        switch (root.result.score * (2 * root.T.side - 1)) {
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

        printf("master down, searched %llu nodes\n", node_searched);
        free(e.worker_tasks);
    } else {
        MPI_Status status;
        tree_t tree;
        result_t result;
        printf("worker %i up\n", e.rank);
        while (1) {
            MPI_Recv(&tree, 1, e.tree_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP) { break; }
            evaluate(&tree, &result);
            MPI_Send(&result, 1, e.result_type, 0, TAG_DATA, MPI_COMM_WORLD);
        }
        printf("worker %i down, searched %llu nodes\n", e.rank, node_searched);
    }

    printf("execution time (%i): %li\n", e.rank, time(NULL) - marker);

    MPI_Finalize();
    if (TRANSPOSITION_TABLE)
        free_tt();
    return EXIT_SUCCESS;
}
