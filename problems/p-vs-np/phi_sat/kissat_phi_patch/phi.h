/*
 * φ-Guided Restart Policy for Kissat
 *
 * Adds phase transition awareness to restart decisions.
 */

#ifndef _phi_h_INCLUDED
#define _phi_h_INCLUDED

#include <stdbool.h>
#include <stdint.h>

/* Phase transition prediction for random 3-SAT */

/* Predict critical clause density α_c for n variables */
double phi_predict_alpha_c(unsigned n);

/* Compute relative distance from phase transition: (α - α_c) / α_c */
double phi_distance(unsigned remaining_vars, unsigned remaining_clauses);

/* Check if we should restart based on φ-analysis */
bool phi_should_restart(
    unsigned remaining_vars,
    unsigned remaining_clauses,
    uint64_t conflicts,
    uint64_t base_interval
);

/* Check if formula is trivially SAT (α << α_c) */
bool phi_early_sat_possible(
    unsigned remaining_vars,
    unsigned remaining_clauses,
    unsigned original_vars,
    double progress  /* fraction of variables assigned */
);

/* Configuration */
typedef struct phi_config {
    bool enabled;              /* Master switch */
    double early_sat_threshold;    /* Distance below α_c for early SAT (-0.5) */
    double danger_zone;            /* Distance from α_c considered hard (0.15) */
    double restart_multiplier;     /* How much to accelerate restarts in danger zone */
    double min_progress;           /* Minimum progress before early SAT (0.3) */
} phi_config;

extern phi_config PHI_CONFIG;

/* Initialize with defaults */
void phi_init(void);

#endif
