#define _POSIX_C_SOURCE 200809L

#include "projet.h"
#include <mpi.h>
#include <time.h>
#include <math.h>

unsigned long long int node_searched = 0;

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

typedef struct Task Task;
typedef struct Node Node;

typedef struct {
    int rank;
    int procs;
    MPI_Datatype tree_type;
    MPI_Datatype result_type;
    // Master only:
    int active_workers;
    int next_worker;
    int task_count;
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
    move_t moves[MAX_MOVES];
    int n_moves;
    int countdown;
    move_t move; // undefined if root
    Node* parent; // null if root
};

void transmit(Env* env, Node* node, move_t move, result_t* result);
void send_task(Env* env, Node* parent, move_t move, tree_t* child);
void evaluate_master(Env* env, Node* node);

int may_drop_node(Env* env, Node* node) {
    if (node->countdown == 0 && node->parent) {
        // TODO: transposition table?
        transmit(env, node->parent, node->move, &node->result);
        free(node);
        return 1;
    }
    return 0;
}

void transmit(Env* env, Node* node, move_t move, result_t* result) {
    int is_elder = node->countdown == node->n_moves &&
        node->T.height < env->depth;
    node->countdown--;
    int score = -result->score;
    if (score > node->result.score) {
        node->result.score = score;
        node->result.best_move = move;
        node->result.pv_length = result->pv_length + 1;
        for (int j = 0; j < result->pv_length; j++)
            node->result.PV[j + 1] = result->PV[j];
        node->result.PV[0] = move;
    }

    node->T.alpha = MAX(node->T.alpha, score);

    if (is_elder) {
        if (may_drop_node(env, node)) return;
        if (ALPHA_BETA_PRUNING && node->T.alpha >= node->T.beta) {
            node->countdown = 0;
            may_drop_node(env, node);
        } else {
            for (int i = 1; i < node->n_moves; i++) {
                Node* child = malloc(sizeof(Node));
                child->move = node->moves[i];
                child->parent = node;
                play_move(&node->T, node->moves[i], &child->T);
                evaluate_master(env, child);
            }
        }
    } else {
        may_drop_node(env, node);
    }
}

void wait_task(Env* env) {
    result_t result;
    MPI_Status status;
    MPI_Recv(&result, 1, env->result_type, MPI_ANY_SOURCE,
             TAG_DATA, MPI_COMM_WORLD, &status);
    Task task = env->worker_tasks[status.MPI_SOURCE - 1];
    env->task_count--;
    // printf(" #%i from %i\n", env->task_count, status.MPI_SOURCE);
    env->next_worker = status.MPI_SOURCE;
    transmit(env, task.parent, task.move, &result);
}

void send_task(Env* env, Node* parent, move_t move, tree_t* child) {
    int p;
    if (env->next_worker != 0) {
        p = env->next_worker;
        env->next_worker = 0;
    } else if (env->active_workers < (env->procs - 1)) {
        env->active_workers++;
        p = env->active_workers;
    } else {
        wait_task(env);
        return send_task(env, parent, move, child);
    }

    MPI_Send(child, 1, env->tree_type, p, TAG_TASK, MPI_COMM_WORLD);
    Task* task = &env->worker_tasks[p - 1];
    task->move = move;
    task->parent = parent;
    env->task_count++;
    // printf("SENT #%i to %i\n", env->task_count, p);
}

void task_barrier(Env* env) {
    while (env->task_count > 0) {
        wait_task(env);
    }
    env->active_workers = 0;
    env->next_worker = 0;
}

void evaluate_master(Env* env, Node* node) {
    node_searched++;

    tree_t* T = &node->T;
    result_t* result = &node->result;

    result->score = -MAX_SCORE - 1;
    result->pv_length = 0;
    node->countdown = 0;

    if (test_draw_or_victory(T, result)) {
        if (node->parent) transmit(env, node->parent, node->move, &node->result);
        return;
    }

    if (TRANSPOSITION_TABLE && tt_lookup(T, result)) {
        if (node->parent) transmit(env, node->parent, node->move, &node->result);
        return;
    }

    compute_attack_squares(T);

    if (T->depth == 0) {
        result->score = (2 * T->side - 1) * heuristic_evaluation(T);
        if (node->parent) transmit(env, node->parent, node->move, &node->result);
        return;
    }

    node->n_moves = generate_legal_moves(T, node->moves);
    node->countdown = node->n_moves;

    if (node->n_moves == 0) {
        result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
        if (node->parent) transmit(env, node->parent, node->move, &node->result);
        return;
    }

    if (ALPHA_BETA_PRUNING)
        sort_moves(T, node->n_moves, node->moves);

    if (T->height < env->depth) {
        Node* child = malloc(sizeof(Node));
        child->move = node->moves[0];
        child->parent = node;
        play_move(T, node->moves[0], &child->T);
        evaluate_master(env, child);
    } else {
        for (int i = 0; i < node->n_moves; i++) {
            tree_t child;
            play_move(T, node->moves[i], &child);
            send_task(env, node, node->moves[i], &child);
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

    for (int depth = env->depth;; depth++) {
        T->depth = depth;
        T->height = 0;
        T->alpha_start = T->alpha = -MAX_SCORE - 1;
        T->beta = MAX_SCORE + 1;

        printf("=====================================\n");
        // TODO only if > env->depth?
        evaluate_master(env, root);
        task_barrier(env);
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

    Time start;
    get_time(&start);

    if (TRANSPOSITION_TABLE)
        init_tt();

    if (e.rank == 0) {
        e.active_workers = 0;
        e.next_worker = 0;
        e.task_count = 0;
        e.worker_tasks = malloc((e.procs - 1) * sizeof(Task));
        e.depth = 2;//log(10 * (e.procs - 1)) / log(16);

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

        printf("master down, task depth = %i, searched %llu nodes\n", e.depth, node_searched);
        printf("execution time: %lf\n", seconds_from(&start));
        free(e.worker_tasks);
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
            evaluate(&tree, &result);
            get_time(&idle_start);
            MPI_Send(&result, 1, e.result_type, 0, TAG_DATA, MPI_COMM_WORLD);
        }

        double execution_time = seconds_from(&start);
        double work_time = execution_time - idle_time;
        double speed = (double)(node_searched) / work_time;
        printf("worker %i down, searched %llu nodes, %lf execution time (%lf work + %lf idle), speed: %lf node/s\n",
               e.rank, node_searched, execution_time, work_time, idle_time, speed);
    }

    MPI_Finalize();
    if (TRANSPOSITION_TABLE)
        free_tt();
    return EXIT_SUCCESS;
}
