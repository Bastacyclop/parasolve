#define _POSIX_C_SOURCE 200809L

#include "projet.h"
#include <time.h>
#include <omp.h>
#include <math.h>

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

unsigned long long int node_searched = 0;
unsigned long int task_spawned = 0;
int task_depth;

void evaluate(tree_t * T, result_t *result)
{
#pragma omp atomic
    node_searched++;

    move_t moves[MAX_MOVES];
    int n_moves;

    result->score = -MAX_SCORE - 1;
    result->pv_length = 0;

    if (test_draw_or_victory(T, result))
        return;

    if (TRANSPOSITION_TABLE && tt_lookup(T, result))
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

    if (T->height <= task_depth) {
#pragma omp atomic
        task_spawned += n_moves;
        omp_lock_t lock;
        omp_init_lock(&lock);
        for (int i = 0; i < n_moves; i++) {
#pragma omp task
            {
                tree_t child;
                result_t child_result;

                play_move(T, moves[i], &child);
                evaluate(&child, &child_result);

                int child_score = -child_result.score;

                omp_set_lock(&lock);
                if (child_score > result->score) {
                    result->score = child_score;
                    result->best_move = moves[i];
                    result->pv_length = child_result.pv_length + 1;
                    for(int j = 0; j < child_result.pv_length; j++)
                        result->PV[j+1] = child_result.PV[j];
                    result->PV[0] = moves[i];
                }

                /* not possible
                if (ALPHA_BETA_PRUNING && child_score >= T->beta)
                    break;
                */

                T->alpha = MAX(T->alpha, child_score);
                omp_unset_lock(&lock);
            }
        }
#pragma omp taskwait
        omp_destroy_lock(&lock);
    } else {
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
                for(int j = 0; j < child_result.pv_length; j++)
                    result->PV[j+1] = child_result.PV[j];
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


void decide(tree_t * T, result_t *result)
{
    for (int depth = 1;; depth++) {
        T->depth = depth;
        T->height = 0;
        T->alpha_start = T->alpha = -MAX_SCORE - 1;
        T->beta = MAX_SCORE + 1;

        printf("=====================================\n");
#pragma omp parallel
#pragma omp single
        evaluate(T, result);

        printf("depth: %d / score: %.2f / best_move : ", T->depth, 0.01 * result->score);
        print_pv(T, result);

        if (DEFINITIVE(result->score))
            break;
    }
}

int main(int argc, char **argv) {
    tree_t root;
    result_t result;

    Time start;
    get_time(&start);

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

    parse_FEN(argv[1], &root);
    print_position(&root);

    task_depth = log(360 * omp_get_max_threads()) / log(6);
    if (argc >= 3) {
        task_depth = atoi(argv[2]);
    }

    decide(&root, &result);

    printf("\nDécision de la position: ");
    switch(result.score * (2*root.side - 1)) {
        case MAX_SCORE: printf("blanc gagne\n"); break;
        case CERTAIN_DRAW: printf("partie nulle\n"); break;
        case -MAX_SCORE: printf("noir gagne\n"); break;
        default: printf("BUG\n");
    }

    printf("Node searched: %llu\n", node_searched);

    if (TRANSPOSITION_TABLE)
        free_tt();

    printf("task depth = %i, spawned = %lu\n", task_depth, task_spawned);
    printf("execution time: %lf\n", seconds_from(&start));
    return 0;
}
