-- Set this to the highest trial number you want to KEEP.
-- Example: keep trials 0 through 29, delete 30 and above.
CREATE TEMP TABLE _trial_cutoff AS
SELECT 29 AS max_trial_number;

BEGIN TRANSACTION;

CREATE TEMP TABLE _trials_to_delete AS
SELECT trial_id
FROM trials
WHERE number > (SELECT max_trial_number FROM _trial_cutoff);

DELETE FROM trial_heartbeats
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DELETE FROM trial_intermediate_values
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DELETE FROM trial_params
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DELETE FROM trial_system_attributes
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DELETE FROM trial_user_attributes
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DELETE FROM trial_values
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DELETE FROM trials
WHERE trial_id IN (SELECT trial_id FROM _trials_to_delete);

DROP TABLE _trials_to_delete;
DROP TABLE _trial_cutoff;

COMMIT;

VACUUM;