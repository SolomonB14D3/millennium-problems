/*
 * φ-Guided Restart Policy Implementation
 *
 * Phase transition model for random 3-SAT.
 */

#include "phi.h"
#include <math.h>

/* Global configuration */
phi_config PHI_CONFIG;

/* Empirically measured α_c(n) values for random 3-SAT */
static const struct {
    unsigned n;
    double alpha_c;
} ALPHA_C_TABLE[] = {
    {    20, 2.50 },
    {    50, 3.00 },
    {   100, 3.20 },
    {   200, 3.40 },
    {   500, 3.57 },
    {  1000, 3.90 },
    {  2000, 4.10 },
    {  4000, 4.20 },
    {  8000, 4.25 },
    { 16000, 4.26 },
    {     0, 4.267 }  /* Asymptotic value, sentinel */
};

void phi_init(void) {
    PHI_CONFIG.enabled = true;
    PHI_CONFIG.early_sat_threshold = -0.5;
    PHI_CONFIG.danger_zone = 0.15;
    PHI_CONFIG.restart_multiplier = 0.5;  /* Restart 2x faster in danger zone */
    PHI_CONFIG.min_progress = 0.3;
}

double phi_predict_alpha_c(unsigned n) {
    if (n == 0) return 4.267;

    /* Find bracketing entries */
    unsigned i = 0;
    while (ALPHA_C_TABLE[i + 1].n != 0 && ALPHA_C_TABLE[i + 1].n < n) {
        i++;
    }

    /* Handle edge cases */
    if (n <= ALPHA_C_TABLE[0].n) {
        return ALPHA_C_TABLE[0].alpha_c;
    }
    if (ALPHA_C_TABLE[i + 1].n == 0) {
        return ALPHA_C_TABLE[i].alpha_c;
    }

    /* Log-linear interpolation between known points */
    unsigned n1 = ALPHA_C_TABLE[i].n;
    unsigned n2 = ALPHA_C_TABLE[i + 1].n;
    double a1 = ALPHA_C_TABLE[i].alpha_c;
    double a2 = ALPHA_C_TABLE[i + 1].alpha_c;

    double t = (log((double)n) - log((double)n1)) / (log((double)n2) - log((double)n1));
    return a1 + t * (a2 - a1);
}

double phi_distance(unsigned remaining_vars, unsigned remaining_clauses) {
    if (remaining_vars == 0) {
        return (remaining_clauses > 0) ? 1000.0 : -1000.0;
    }

    double alpha = (double)remaining_clauses / (double)remaining_vars;
    double alpha_c = phi_predict_alpha_c(remaining_vars);

    return (alpha - alpha_c) / alpha_c;
}

bool phi_should_restart(
    unsigned remaining_vars,
    unsigned remaining_clauses,
    uint64_t conflicts,
    uint64_t base_interval
) {
    if (!PHI_CONFIG.enabled) {
        return conflicts >= base_interval;
    }

    double distance = phi_distance(remaining_vars, remaining_clauses);

    /*
     * φ-guided restart policy:
     *
     * Near phase transition (|distance| < danger_zone):
     *   - This is the hard region
     *   - Restart more aggressively to escape bad search paths
     *   - Heavy-tailed runtime means fresh starts often find easier paths
     *
     * Far from transition:
     *   - Problem is "easy" (either trivially SAT or UNSAT)
     *   - Use normal restart policy, don't interrupt progress
     */

    if (fabs(distance) < PHI_CONFIG.danger_zone) {
        /* In danger zone: accelerate restarts */
        uint64_t adjusted_interval = (uint64_t)(base_interval * PHI_CONFIG.restart_multiplier);
        return conflicts >= adjusted_interval;
    }

    /* Outside danger zone: normal policy */
    return conflicts >= base_interval;
}

bool phi_early_sat_possible(
    unsigned remaining_vars,
    unsigned remaining_clauses,
    unsigned original_vars,
    double progress
) {
    if (!PHI_CONFIG.enabled) {
        return false;
    }

    /* Need sufficient progress before early termination */
    if (progress < PHI_CONFIG.min_progress) {
        return false;
    }

    double distance = phi_distance(remaining_vars, remaining_clauses);

    /*
     * Early SAT detection:
     *
     * When α << α_c, the remaining formula is severely under-constrained.
     * Almost any assignment will satisfy it. We can:
     * 1. Switch to greedy completion
     * 2. Or just declare SAT (risky, but fast)
     *
     * This works because random 3-SAT with low α is trivially satisfiable
     * with probability approaching 1.
     */

    return distance < PHI_CONFIG.early_sat_threshold;
}
