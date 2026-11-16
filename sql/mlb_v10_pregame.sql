-- One row per completed 2026 game: the V10 feature row AS THE MODEL SAW IT, the label, and prod's own probability.
-- Leakage rules (do not loosen without thinking):
--   * only predictions made BEFORE first pitch (backfills after the game are contaminated). Measured 2026-09-29 of 2,427
--     completed regular-season games: 2,028 kept, 374 dropped (only post-first-pitch predictions, mostly Mar 27-Apr 17),
--     13 dropped for a NULL game_time_utc, 12 never predicted.
--   * features = the latest computed_at at or before that prediction (game_v10_features holds several per game)
--   * games can hold a skeleton row plus a final row; take the latest synced_at, Final/Game Over only, no ties
WITH pred AS (
  SELECT game_pk, model_version, home_win_probability AS prod_prob, predicted_at, game_time_utc
  FROM `hankstank.mlb_2026_season.game_predictions`
  WHERE predicted_at < game_time_utc
  QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) = 1
),
final AS (
  SELECT game_pk, home_score, away_score, CAST(home_score > away_score AS INT64) AS home_win
  FROM `hankstank.mlb_2026_season.games`
  WHERE status IN ('Final', 'Game Over') AND home_score IS NOT NULL AND away_score IS NOT NULL
    AND home_score != away_score
  QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY synced_at DESC NULLS LAST) = 1
),
feat AS (
  SELECT f.*
  FROM `hankstank.mlb_2026_season.game_v10_features` f
  JOIN pred p USING (game_pk)
  WHERE f.computed_at <= p.predicted_at
  QUALIFY ROW_NUMBER() OVER (PARTITION BY f.game_pk ORDER BY f.computed_at DESC) = 1
)
SELECT feat.*, pred.prod_prob, pred.model_version AS prod_model_version, pred.predicted_at,
       final.home_score, final.away_score, final.home_win
FROM feat
JOIN pred USING (game_pk)
JOIN final USING (game_pk)
ORDER BY feat.game_date, feat.game_pk
