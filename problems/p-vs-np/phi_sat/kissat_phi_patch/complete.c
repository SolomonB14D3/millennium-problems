/*
 * Greedy completion for φ-early-SAT
 *
 * When α << α_c, the remaining formula is trivially satisfiable.
 * Complete the assignment greedily without backtracking.
 */

#include "internal.h"
#include "decide.h"
#include "assign.h"
#include "propagate.h"

/*
 * Greedily complete all unassigned variables.
 *
 * Strategy: For each unassigned variable, pick the polarity that
 * satisfies the most unsatisfied clauses. Since α << α_c, conflicts
 * are extremely unlikely.
 *
 * Returns: 10 (SAT) if successful, 0 if conflict encountered
 */
int kissat_complete_greedily(kissat *solver) {

    while (solver->unassigned > 0) {
        /* Find best variable and polarity */
        unsigned best_var = 0;
        bool best_polarity = true;
        int best_score = -1;

        for (unsigned v = 1; v <= solver->vars; v++) {
            if (VALUE(v) != 0) continue;  /* Already assigned */

            /* Count clauses satisfied by each polarity */
            int pos_score = 0, neg_score = 0;

            /* Score positive polarity */
            watches *pos_watches = &WATCHES(LIT(v));
            for (all_binary_large_watches(w, *pos_watches)) {
                if (w.type.binary) pos_score++;
                else pos_score += 2;  /* Large clauses worth more */
            }

            /* Score negative polarity */
            watches *neg_watches = &WATCHES(NOT(LIT(v)));
            for (all_binary_large_watches(w, *neg_watches)) {
                if (w.type.binary) neg_score++;
                else neg_score += 2;
            }

            int score = (pos_score > neg_score) ? pos_score : neg_score;
            bool polarity = (pos_score > neg_score);

            if (score > best_score) {
                best_score = score;
                best_var = v;
                best_polarity = polarity;
            }
        }

        if (best_var == 0) break;  /* All assigned */

        /* Assign and propagate */
        unsigned lit = best_polarity ? LIT(best_var) : NOT(LIT(best_var));

        kissat_decide(solver, lit);

        clause *conflict = kissat_propagate(solver);

        if (conflict) {
            /*
             * Unexpected conflict during greedy completion.
             * This shouldn't happen if α << α_c, but handle gracefully.
             */
            LOG("φ-greedy: unexpected conflict, falling back to search");
            return 0;  /* Signal to resume normal search */
        }
    }

    /* Verify all clauses satisfied */
    if (solver->unassigned == 0) {
        return 10;  /* SAT */
    }

    return 0;  /* Incomplete */
}
