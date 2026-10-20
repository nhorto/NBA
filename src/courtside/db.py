"""SQLite access: connect(), append-only migrations, small upsert helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config

MIGRATIONS: list[str] = [
    # 1: v1 tables
    """
    CREATE TABLE games (
        game_id TEXT PRIMARY KEY,
        espn_id TEXT,
        season TEXT,
        nba_day TEXT,
        tipoff_utc TEXT,
        home TEXT,
        away TEXT,
        status TEXT NOT NULL DEFAULT 'scheduled'
            CHECK (status IN ('scheduled','live','final','postponed')),
        home_score INTEGER,
        away_score INTEGER,
        updated_at TEXT
    );
    CREATE INDEX idx_games_nba_day ON games(nba_day);

    CREATE TABLE odds_snapshots (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        source TEXT NOT NULL,
        provider TEXT,
        home_spread REAL,
        total REAL,
        home_ml REAL,
        away_ml REAL,
        raw_path TEXT
    );
    CREATE INDEX idx_odds_game_captured ON odds_snapshots(game_id, captured_at);

    CREATE TABLE injury_snapshots (
        id INTEGER PRIMARY KEY,
        captured_at TEXT NOT NULL,
        source TEXT NOT NULL,
        team TEXT NOT NULL,
        player_name TEXT NOT NULL,
        player_id TEXT,
        status TEXT,
        reason TEXT,
        raw_path TEXT
    );
    CREATE INDEX idx_injury_captured ON injury_snapshots(captured_at);

    CREATE TABLE collect_runs (
        id INTEGER PRIMARY KEY,
        job TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        ok INTEGER,
        detail TEXT
    );
    """,
    # 2: historical game logs (stats.nba.com)
    """
    CREATE TABLE team_games (
        game_id TEXT NOT NULL, team TEXT NOT NULL, season TEXT, season_type TEXT,
        game_date TEXT, opp TEXT, is_home INTEGER, wl TEXT, min REAL,
        fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER, fta INTEGER,
        oreb INTEGER, dreb INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER,
        tov INTEGER, pf INTEGER, pts INTEGER, plus_minus REAL,
        PRIMARY KEY (game_id, team)
    );
    CREATE INDEX idx_team_games_season ON team_games(season);
    CREATE INDEX idx_team_games_date ON team_games(game_date);
    CREATE INDEX idx_team_games_team_date ON team_games(team, game_date);

    CREATE TABLE player_games (
        game_id TEXT NOT NULL, player_id INTEGER NOT NULL, player_name TEXT, team TEXT,
        season TEXT, season_type TEXT, game_date TEXT, min REAL,
        fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER, fta INTEGER,
        oreb INTEGER, dreb INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER,
        tov INTEGER, pf INTEGER, pts INTEGER, plus_minus REAL,
        PRIMARY KEY (game_id, player_id)
    );
    CREATE INDEX idx_player_games_season ON player_games(season);
    CREATE INDEX idx_player_games_date ON player_games(game_date);
    CREATE INDEX idx_player_games_player_date ON player_games(player_id, game_date);

    CREATE TABLE team_games_adv (
        game_id TEXT NOT NULL, team TEXT NOT NULL,
        off_rating REAL, def_rating REAL, net_rating REAL, pace REAL, poss REAL,
        efg_pct REAL, ts_pct REAL, tov_pct REAL, oreb_pct REAL, dreb_pct REAL,
        ast_pct REAL, pie REAL,
        PRIMARY KEY (game_id, team)
    );

    CREATE TABLE player_games_adv (
        game_id TEXT NOT NULL, player_id INTEGER NOT NULL,
        off_rating REAL, def_rating REAL, net_rating REAL, usg_pct REAL, ts_pct REAL,
        efg_pct REAL, ast_pct REAL, reb_pct REAL, pace REAL, poss REAL, pie REAL,
        PRIMARY KEY (game_id, player_id)
    );
    """,
    # 3: historical closing lines (Kaggle SBR/ESPN BET, ESPN core API open/close)
    """
    CREATE TABLE IF NOT EXISTS closing_lines (
        game_id TEXT NOT NULL,
        source TEXT NOT NULL,
        provider TEXT,
        home_spread_open REAL, home_spread_close REAL,
        total_open REAL, total_close REAL,
        home_ml_open REAL, home_ml_close REAL,
        away_ml_open REAL, away_ml_close REAL,
        PRIMARY KEY (game_id, source)
    );
    """,
    # 4: per-game analytics facts, prediction history, built Briefs
    """
    CREATE TABLE game_facts (
        game_id TEXT PRIMARY KEY,
        nba_day TEXT,
        facts_json TEXT NOT NULL,
        computed_at TEXT NOT NULL
    );
    CREATE INDEX idx_game_facts_day ON game_facts(nba_day);

    -- Append-only history: every run of a model inserts new rows (never updates). nba_day,
    -- home and away are denormalised so past days with no `games` row (demo days built from
    -- team_games) can still be joined to results.
    CREATE TABLE predictions (
        game_id TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        home_win_prob REAL,
        home_margin REAL,
        total REAL,
        market_home_spread REAL,
        market_total REAL,
        market_home_prob REAL,
        inputs_json TEXT,
        nba_day TEXT,
        home TEXT,
        away TEXT,
        PRIMARY KEY (game_id, model, created_at)
    );
    CREATE INDEX idx_predictions_day ON predictions(nba_day);

    -- The prediction a reader should score: per (game, model) the newest row made at or
    -- before tip-off (any row when tip-off is unknown, e.g. games with no `games` row).
    CREATE VIEW latest_pretip_predictions AS
    SELECT p.* FROM predictions p
    WHERE p.created_at = (
        SELECT MAX(p2.created_at) FROM predictions p2
        LEFT JOIN games g ON g.game_id = p2.game_id
        WHERE p2.game_id = p.game_id AND p2.model = p.model
          AND (g.tipoff_utc IS NULL OR p2.created_at <= g.tipoff_utc)
    );

    CREATE TABLE briefs (
        nba_day TEXT PRIMARY KEY,
        brief_json TEXT NOT NULL,
        built_at TEXT NOT NULL,
        version TEXT
    );
    """,
    # 5: MLOps - model registry + lineage, monitoring metrics, data checks, drift, alerts.
    # See docs/MLOPS.md.
    """
    -- One row per model *spec* (code + hyperparameters + feature set). Daily refits of the
    -- same spec on new data are not new versions. status: candidate | shadow | champion |
    -- retired. At most one champion per model_name.
    CREATE TABLE model_versions (
        version_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        spec_hash TEXT NOT NULL,
        code_sha TEXT,
        params_json TEXT,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        promoted_at TEXT,
        retired_at TEXT,
        benchmark_json TEXT,
        mlflow_run_id TEXT,
        notes TEXT
    );
    CREATE INDEX idx_model_versions_name ON model_versions(model_name, status);

    -- Append-only audit trail: register | shadow | promote | rollback | retire | reject.
    CREATE TABLE registry_events (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        version_id TEXT NOT NULL,
        action TEXT NOT NULL,
        reason TEXT,
        details_json TEXT
    );

    -- Lineage: which registered version produced a prediction, and in what role
    -- (champion = shown to readers, shadow = logged for evaluation only).
    ALTER TABLE predictions ADD COLUMN model_version TEXT;
    ALTER TABLE predictions ADD COLUMN role TEXT;

    -- Resolved-prediction performance per model and window, computed daily.
    -- window: 7d | 30d | last100 | last300 | season. mkt_* are on the same games.
    CREATE TABLE metric_snapshots (
        as_of TEXT NOT NULL,
        model TEXT NOT NULL,
        model_version TEXT,
        window TEXT NOT NULL,
        n INTEGER,
        log_loss REAL,
        brier REAL,
        accuracy REAL,
        ece REAL,
        home_pred REAL,
        home_actual REAL,
        margin_bias REAL,
        margin_mae REAL,
        total_bias REAL,
        total_mae REAL,
        mkt_n INTEGER,
        mkt_log_loss REAL,
        mkt_brier REAL,
        ll_diff_vs_mkt REAL,
        ll_diff_se REAL,
        details_json TEXT,
        PRIMARY KEY (as_of, model, window)
    );

    -- Data-quality / pipeline-health results. status: ok | warn | fail | skip.
    CREATE TABLE data_checks (
        run_at TEXT NOT NULL,
        check_name TEXT NOT NULL,
        status TEXT NOT NULL,
        value REAL,
        threshold REAL,
        detail TEXT,
        PRIMARY KEY (run_at, check_name)
    );

    -- Feature drift: population stability index of live features vs the training reference.
    CREATE TABLE drift_snapshots (
        as_of TEXT NOT NULL,
        feature TEXT NOT NULL,
        psi REAL,
        ref_label TEXT,
        cur_n INTEGER,
        PRIMARY KEY (as_of, feature)
    );

    -- Deduplicated alerts: one open row per `key`; re-firing updates it, clearing resolves it.
    CREATE TABLE alerts (
        id INTEGER PRIMARY KEY,
        key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        severity TEXT NOT NULL,
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        resolved_at TEXT
    );
    CREATE INDEX idx_alerts_open ON alerts(status, key);
    """,
    # 6: lineage-safe predictions key. The v4 PRIMARY KEY (game_id, model, created_at) let a
    # row of another version with the same model name and created_at silently replace the
    # champion's (INSERT OR REPLACE). Rebuilt with a surrogate id and a UNIQUE key that includes
    # the version; existing rows copied unchanged. latest_pretip_predictions (readers: Brief,
    # scorecard) now sees only champion/legacy (NULL role) rows and breaks created_at ties by
    # id; latest_pretip_predictions_all is the same per (game, model, role, version) for
    # evaluation (shadows, monitoring).
    """
    DROP VIEW IF EXISTS latest_pretip_predictions;
    ALTER TABLE predictions RENAME TO predictions_v5;
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        home_win_prob REAL,
        home_margin REAL,
        total REAL,
        market_home_spread REAL,
        market_total REAL,
        market_home_prob REAL,
        inputs_json TEXT,
        nba_day TEXT,
        home TEXT,
        away TEXT,
        model_version TEXT,
        role TEXT CHECK (role IS NULL OR role IN ('champion', 'shadow'))
    );
    INSERT INTO predictions (game_id, model, created_at, home_win_prob, home_margin, total,
        market_home_spread, market_total, market_home_prob, inputs_json, nba_day, home, away,
        model_version, role)
    SELECT game_id, model, created_at, home_win_prob, home_margin, total, market_home_spread,
        market_total, market_home_prob, inputs_json, nba_day, home, away, model_version, role
    FROM predictions_v5 ORDER BY created_at, rowid;
    DROP TABLE predictions_v5;
    CREATE UNIQUE INDEX ux_predictions_key
        ON predictions(game_id, model, created_at, COALESCE(model_version, ''));
    CREATE INDEX idx_predictions_day ON predictions(nba_day);
    CREATE INDEX idx_predictions_game_model ON predictions(game_id, model, created_at);

    CREATE VIEW latest_pretip_predictions AS
    SELECT id, game_id, model, created_at, home_win_prob, home_margin, total,
           market_home_spread, market_total, market_home_prob, inputs_json, nba_day, home, away,
           model_version, role
    FROM (
        SELECT p.*, ROW_NUMBER() OVER (PARTITION BY p.game_id, p.model
                                       ORDER BY p.created_at DESC, p.id DESC) AS rn
        FROM predictions p LEFT JOIN games g ON g.game_id = p.game_id
        WHERE (g.tipoff_utc IS NULL OR p.created_at <= g.tipoff_utc)
          AND (p.role IS NULL OR p.role = 'champion')
    ) WHERE rn = 1;

    CREATE VIEW latest_pretip_predictions_all AS
    SELECT id, game_id, model, created_at, home_win_prob, home_margin, total,
           market_home_spread, market_total, market_home_prob, inputs_json, nba_day, home, away,
           model_version, role
    FROM (
        SELECT p.*, ROW_NUMBER() OVER (PARTITION BY p.game_id, p.model, COALESCE(p.role, ''),
                                                    COALESCE(p.model_version, '')
                                       ORDER BY p.created_at DESC, p.id DESC) AS rn
        FROM predictions p LEFT JOIN games g ON g.game_id = p.game_id
        WHERE (g.tipoff_utc IS NULL OR p.created_at <= g.tipoff_utc)
    ) WHERE rn = 1;
    """,
    # players backfill (stats.nba.com player/matchup/tracking data; see backfill_players.py).
    # Season tables keep key numeric columns typed; unmapped headers land in extra_json.
    # player_matchups orientation: off_player_id is the offensive player, def_player_id the
    # defender who guarded him; stats are the offensive player's output against that defender.
    """
    CREATE TABLE IF NOT EXISTS player_index (
        player_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, slug TEXT,
        team_id INTEGER, team TEXT, jersey TEXT, position TEXT, height TEXT, height_in INTEGER,
        weight INTEGER, college TEXT, country TEXT, draft_year INTEGER, draft_round INTEGER,
        draft_number INTEGER, from_year INTEGER, to_year INTEGER, roster_status REAL,
        is_defunct INTEGER, snapshot_season TEXT
    );

    CREATE TABLE IF NOT EXISTS team_rosters (
        season TEXT NOT NULL, team_id INTEGER NOT NULL, team TEXT, player_id INTEGER NOT NULL,
        player TEXT, jersey TEXT, position TEXT, height TEXT, height_in INTEGER, weight INTEGER,
        birth_date TEXT, age REAL, exp INTEGER, school TEXT, how_acquired TEXT,
        PRIMARY KEY (season, team_id, player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_team_rosters_player ON team_rosters(player_id, season);

    CREATE TABLE IF NOT EXISTS team_coaches (
        season TEXT NOT NULL, team_id INTEGER NOT NULL, team TEXT, coach_id INTEGER NOT NULL,
        coach_name TEXT, coach_type TEXT, is_assistant INTEGER, sort_sequence INTEGER,
        PRIMARY KEY (season, team_id, coach_id)
    );

    CREATE TABLE IF NOT EXISTS matchup_rollup (
        season TEXT NOT NULL, season_type TEXT NOT NULL, def_player_id INTEGER NOT NULL,
        position TEXT NOT NULL, def_player_name TEXT, pct_of_time REAL, gp INTEGER,
        matchup_min REAL, partial_poss REAL, pts REAL, team_pts REAL, ast REAL, tov REAL,
        blk REAL, fgm REAL, fga REAL, fg3m REAL, fg3a REAL, ftm REAL, fta REAL, sfl REAL,
        extra_json TEXT,
        PRIMARY KEY (season, season_type, def_player_id, position)
    );

    CREATE TABLE IF NOT EXISTS season_matchups (
        season TEXT NOT NULL, season_type TEXT NOT NULL, off_player_id INTEGER NOT NULL,
        def_player_id INTEGER NOT NULL, gp INTEGER, matchup_min REAL, matchup_sec REAL,
        partial_poss REAL, pts REAL, team_pts REAL, ast REAL, tov REAL, blk REAL, fgm REAL,
        fga REAL, fg3m REAL, fg3a REAL, help_blk REAL, help_fgm REAL, help_fga REAL,
        ftm REAL, fta REAL, sfl REAL, extra_json TEXT,
        PRIMARY KEY (season, season_type, off_player_id, def_player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_season_matchups_def
        ON season_matchups(def_player_id, season);

    CREATE TABLE IF NOT EXISTS pt_defend (
        season TEXT NOT NULL, season_type TEXT NOT NULL, category TEXT NOT NULL,
        player_id INTEGER NOT NULL, player_name TEXT, team_id INTEGER, team TEXT, position TEXT,
        age REAL, gp INTEGER, g INTEGER, freq REAL, d_fgm REAL, d_fga REAL, d_fg_pct REAL,
        normal_fg_pct REAL, pct_plusminus REAL, extra_json TEXT,
        PRIMARY KEY (season, season_type, category, player_id)
    );

    CREATE TABLE IF NOT EXISTS pt_shot_defdist (
        season TEXT NOT NULL, season_type TEXT NOT NULL, def_dist TEXT NOT NULL,
        player_id INTEGER NOT NULL, player_name TEXT, team_id INTEGER, team TEXT, age REAL,
        gp INTEGER, g INTEGER, fga_freq REAL, fgm REAL, fga REAL, fg_pct REAL, efg_pct REAL,
        fg2a_freq REAL, fg2m REAL, fg2a REAL, fg2_pct REAL, fg3a_freq REAL, fg3m REAL,
        fg3a REAL, fg3_pct REAL, extra_json TEXT,
        PRIMARY KEY (season, season_type, def_dist, player_id)
    );

    -- player_id = 0 on team rows (entity 'T'); grouping is Offensive | Defensive.
    CREATE TABLE IF NOT EXISTS synergy_playtypes (
        season TEXT NOT NULL, season_type TEXT NOT NULL, entity TEXT NOT NULL,
        grouping TEXT NOT NULL, play_type TEXT NOT NULL, player_id INTEGER NOT NULL,
        team_id INTEGER NOT NULL, player_name TEXT, team TEXT, percentile REAL, gp INTEGER,
        poss REAL, poss_pct REAL, ppp REAL, pts REAL, fgm REAL, fga REAL, fgmx REAL,
        fg_pct REAL, efg_pct REAL, ft_poss_pct REAL, tov_poss_pct REAL, sf_poss_pct REAL,
        plusone_poss_pct REAL, score_poss_pct REAL, extra_json TEXT,
        PRIMARY KEY (season, season_type, entity, grouping, play_type, player_id, team_id)
    );

    CREATE TABLE IF NOT EXISTS hustle_season (
        season TEXT NOT NULL, season_type TEXT NOT NULL, player_id INTEGER NOT NULL,
        player_name TEXT, team_id INTEGER, team TEXT, age REAL, gp INTEGER, min REAL,
        contested_shots REAL, contested_2pt REAL, contested_3pt REAL, deflections REAL,
        charges_drawn REAL, screen_assists REAL, screen_ast_pts REAL, loose_off REAL,
        loose_def REAL, loose_total REAL, boxouts_off REAL, boxouts_def REAL,
        boxout_team_rebs REAL, boxout_player_rebs REAL, box_outs REAL, extra_json TEXT,
        PRIMARY KEY (season, season_type, player_id, team_id)
    );

    CREATE TABLE IF NOT EXISTS player_matchups (
        game_id TEXT NOT NULL, off_player_id INTEGER NOT NULL, def_player_id INTEGER NOT NULL,
        off_team TEXT, def_team TEXT, matchup_min REAL, partial_poss REAL, pct_def_time REAL,
        pct_off_time REAL, pct_both_time REAL, switches INTEGER, pts INTEGER, team_pts INTEGER,
        ast INTEGER, pot_ast INTEGER, tov INTEGER, blk INTEGER, fgm INTEGER, fga INTEGER,
        fg3m INTEGER, fg3a INTEGER, help_blk INTEGER, help_fgm INTEGER, help_fga INTEGER,
        ftm INTEGER, fta INTEGER, sfl INTEGER,
        PRIMARY KEY (game_id, off_player_id, def_player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_player_matchups_off ON player_matchups(off_player_id);
    CREATE INDEX IF NOT EXISTS idx_player_matchups_def ON player_matchups(def_player_id);

    CREATE TABLE IF NOT EXISTS player_tracking (
        game_id TEXT NOT NULL, player_id INTEGER NOT NULL, team TEXT, minutes REAL,
        speed REAL, distance REAL, reb_chances_off INTEGER, reb_chances_def INTEGER,
        reb_chances_total INTEGER, touches INTEGER, secondary_ast INTEGER, ft_ast INTEGER,
        passes INTEGER, ast INTEGER, cont_fgm INTEGER, cont_fga INTEGER, uncont_fgm INTEGER,
        uncont_fga INTEGER, def_rim_fgm INTEGER, def_rim_fga INTEGER,
        PRIMARY KEY (game_id, player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_player_tracking_player ON player_tracking(player_id);

    CREATE TABLE IF NOT EXISTS player_hustle (
        game_id TEXT NOT NULL, player_id INTEGER NOT NULL, team TEXT, minutes REAL,
        points INTEGER, contested_shots INTEGER, contested_2pt INTEGER, contested_3pt INTEGER,
        deflections INTEGER, charges_drawn INTEGER, screen_assists INTEGER,
        screen_ast_pts INTEGER, loose_off INTEGER, loose_def INTEGER, loose_total INTEGER,
        boxouts_off INTEGER, boxouts_def INTEGER, boxout_team_rebs INTEGER,
        boxout_player_rebs INTEGER, box_outs INTEGER,
        PRIMARY KEY (game_id, player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_player_hustle_player ON player_hustle(player_id);
    """,
    # pre-tip archive sources that vanish after tip-off: ESPN/DraftKings player props, official
    # referee assignments, Action Network per-book lines, DraftKings DFS salaries.
    #  * prop_snapshots is a CHANGE LOG: a row is written when a market's line/prices differ from
    #    the latest stored state (raw pages keep every poll). State at time T = latest row per
    #    (game_id, provider, espn_athlete_id, espn_team_id, market_id, kind[, line for
    #    milestones]) with captured_at <= T. kind: ou | yesno | milestone | single; over_odds
    #    is the over / yes / N-or-more price (first item of ESPN's pair, see sources/espn_props).
    #  * referee_assignments: one row per official and role; a new set of rows is written only
    #    when the posted crews change (game_id NULL for replay-center rows / unmatched games).
    #  * dfs_salaries: one row per draftable (player x roster slot), full snapshot per run.
    """
    CREATE TABLE IF NOT EXISTS prop_snapshots (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        provider TEXT NOT NULL,
        provider_name TEXT,
        espn_athlete_id TEXT,
        espn_team_id TEXT,
        player_name TEXT,
        subject TEXT,
        market_id TEXT NOT NULL,
        market TEXT NOT NULL,
        kind TEXT NOT NULL,
        line REAL,
        over_odds INTEGER,
        under_odds INTEGER,
        open_line REAL,
        open_over_odds INTEGER,
        open_under_odds INTEGER,
        last_updated TEXT,
        side_basis TEXT,
        raw_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_props_game_captured ON prop_snapshots(game_id, captured_at);
    CREATE INDEX IF NOT EXISTS idx_props_athlete_market
        ON prop_snapshots(espn_athlete_id, market_id, captured_at);

    CREATE TABLE IF NOT EXISTS referee_assignments (
        id INTEGER PRIMARY KEY,
        nba_day TEXT NOT NULL,
        game_id TEXT,
        game_label TEXT,
        home TEXT,
        away TEXT,
        official_name TEXT NOT NULL,
        official_number INTEGER,
        role TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        raw_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_refs_day_game ON referee_assignments(nba_day, game_id);
    CREATE INDEX IF NOT EXISTS idx_refs_official ON referee_assignments(official_name);

    CREATE TABLE IF NOT EXISTS line_snapshots (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        source TEXT NOT NULL,
        book TEXT NOT NULL,
        book_id INTEGER,
        is_open INTEGER NOT NULL DEFAULT 0,
        home_spread REAL,
        home_spread_price INTEGER,
        away_spread_price INTEGER,
        total REAL,
        over_price INTEGER,
        under_price INTEGER,
        home_ml INTEGER,
        away_ml INTEGER,
        home_team_total REAL,
        away_team_total REAL,
        line_updated_at TEXT,
        spread_home_bets_pct REAL,
        spread_home_money_pct REAL,
        total_over_bets_pct REAL,
        total_over_money_pct REAL,
        ml_home_bets_pct REAL,
        ml_home_money_pct REAL,
        num_bets INTEGER,
        source_game_id TEXT,
        raw_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_lines_game_captured ON line_snapshots(game_id, captured_at);
    CREATE INDEX IF NOT EXISTS idx_lines_book ON line_snapshots(book, captured_at);

    CREATE TABLE IF NOT EXISTS dfs_salaries (
        id INTEGER PRIMARY KEY,
        captured_at TEXT NOT NULL,
        site TEXT NOT NULL,
        slate_id TEXT NOT NULL,
        game_type TEXT NOT NULL,
        game_type_id INTEGER,
        draftable_id TEXT,
        player_name TEXT NOT NULL,
        site_player_id TEXT,
        site_player_dk_id TEXT,
        team TEXT,
        positions TEXT,
        roster_slot_id INTEGER,
        roster_slot TEXT,
        salary INTEGER,
        status TEXT,
        news_status TEXT,
        fppg REAL,
        disabled INTEGER NOT NULL DEFAULT 0,
        competition_id TEXT,
        game_id TEXT,
        game_label TEXT,
        home TEXT,
        away TEXT,
        start_time TEXT,
        raw_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_slate_captured ON dfs_salaries(slate_id, captured_at);
    CREATE INDEX IF NOT EXISTS idx_dfs_player ON dfs_salaries(site_player_id, captured_at);
    CREATE INDEX IF NOT EXISTS idx_dfs_game ON dfs_salaries(game_id, captured_at);
    """,
    # 9: player projections + DFS optimizer (docs/PLAYERS.md v2 §6). Historical salaries
    # (RotoGuru), cross-source player ids, slates/contests/payouts, projections with quantiles,
    # optimizer runs and lineups, manual contest uploads, prop comparison, post-game why v0,
    # stints/on-court tallies and the nightly empirical-Bayes rate state.
    """
    ALTER TABLE dfs_salaries ADD COLUMN starter INTEGER;
    ALTER TABLE dfs_salaries ADD COLUMN swappable INTEGER;

    CREATE TABLE IF NOT EXISTS player_ids (
        source TEXT NOT NULL,
        source_id TEXT NOT NULL,
        player_id INTEGER,
        name TEXT,
        team TEXT,
        first_seen TEXT,
        last_seen TEXT,
        how TEXT,
        confidence REAL,
        PRIMARY KEY (source, source_id)
    );
    CREATE INDEX IF NOT EXISTS idx_player_ids_player ON player_ids(player_id);

    CREATE TABLE IF NOT EXISTS player_aliases (
        source TEXT NOT NULL,
        name_norm TEXT NOT NULL,
        team TEXT NOT NULL DEFAULT '',
        player_id INTEGER NOT NULL,
        note TEXT,
        PRIMARY KEY (source, name_norm, team)
    );

    CREATE TABLE IF NOT EXISTS dfs_hist_salaries (
        game_date TEXT NOT NULL,
        rg_id TEXT NOT NULL,
        player_id INTEGER,
        name TEXT NOT NULL,
        pos TEXT,
        team TEXT,
        opp TEXT,
        home INTEGER,
        starter INTEGER,
        salary INTEGER,
        dk_pts REAL,
        minutes REAL,
        raw_stats TEXT,
        game_id TEXT,
        on_slate INTEGER,
        raw_path TEXT,
        PRIMARY KEY (game_date, rg_id)
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_hist_player ON dfs_hist_salaries(player_id, game_date);

    CREATE TABLE IF NOT EXISTS dfs_slates (
        slate_id TEXT PRIMARY KEY,
        site TEXT NOT NULL DEFAULT 'dk',
        nba_day TEXT NOT NULL,
        game_type_id INTEGER,
        tag TEXT,
        game_ids_json TEXT,
        lock_utc TEXT,
        rules_json TEXT,
        captured_at TEXT,
        raw_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_slates_day ON dfs_slates(nba_day);

    CREATE TABLE IF NOT EXISTS dfs_contests (
        contest_id TEXT PRIMARY KEY,
        slate_id TEXT,
        name TEXT,
        entry_fee REAL,
        field_size INTEGER,
        max_per_user INTEGER,
        guaranteed INTEGER,
        total_prizes REAL,
        payout_json TEXT,
        attrs_json TEXT,
        captured_at TEXT NOT NULL,
        raw_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_contests_slate ON dfs_contests(slate_id);

    CREATE TABLE IF NOT EXISTS player_projections (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        nba_day TEXT,
        slate_id TEXT,
        player_id INTEGER NOT NULL,
        team TEXT,
        run_id TEXT,
        created_at TEXT NOT NULL,
        model_version TEXT,
        role TEXT,
        p_play REAL,
        minutes_json TEXT,
        inputs_json TEXT,
        UNIQUE (game_id, player_id, run_id, model_version)
    );
    CREATE INDEX IF NOT EXISTS idx_proj_day ON player_projections(nba_day, player_id);
    CREATE INDEX IF NOT EXISTS idx_proj_run ON player_projections(run_id);

    CREATE TABLE IF NOT EXISTS player_projection_stats (
        projection_id INTEGER NOT NULL REFERENCES player_projections(id),
        stat TEXT NOT NULL,
        cond TEXT NOT NULL,
        mean REAL, sd REAL,
        q05 REAL, q10 REAL, q25 REAL, q50 REAL, q75 REAL, q90 REAL, q95 REAL,
        PRIMARY KEY (projection_id, stat, cond)
    );

    CREATE TABLE IF NOT EXISTS dfs_runs (
        run_id TEXT PRIMARY KEY,
        slate_id TEXT,
        kind TEXT NOT NULL,
        created_at TEXT NOT NULL,
        model_version TEXT,
        sim_path TEXT,
        path TEXT,
        late_news_json TEXT,
        seconds REAL,
        ok INTEGER NOT NULL DEFAULT 1,
        detail TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_runs_slate ON dfs_runs(slate_id, created_at);

    CREATE TABLE IF NOT EXISTS dfs_ownership_proxy (
        run_id TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        own REAL,
        band TEXT,
        version TEXT,
        PRIMARY KEY (run_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS dfs_lineups (
        id INTEGER PRIMARY KEY,
        run_id TEXT NOT NULL,
        mode TEXT NOT NULL,
        contest_id TEXT,
        rank INTEGER,
        slots_json TEXT NOT NULL,
        salary INTEGER,
        mean REAL, p10 REAL, p50 REAL, p90 REAL,
        p_cash REAL, p_top1 REAL, e_payout REAL, dupes REAL,
        why_json TEXT,
        actual REAL,
        pct_field_proxy REAL,
        ratio_hindsight REAL
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_lineups_run ON dfs_lineups(run_id, mode, rank);

    CREATE TABLE IF NOT EXISTS dfs_contest_results (
        contest_id TEXT NOT NULL,
        rank INTEGER NOT NULL,
        entry_hash TEXT NOT NULL,
        points REAL,
        lineup_text TEXT,
        payout REAL,
        uploaded_at TEXT NOT NULL,
        PRIMARY KEY (contest_id, rank, entry_hash)
    );

    CREATE TABLE IF NOT EXISTS dfs_contest_ownership (
        contest_id TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        name TEXT,
        pct_drafted REAL,
        fpts REAL,
        PRIMARY KEY (contest_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS prop_compare (
        projection_id INTEGER NOT NULL,
        market TEXT NOT NULL,
        line REAL NOT NULL,
        captured_at TEXT NOT NULL,
        mkt_p_above REAL,
        our_p_above REAL,
        PRIMARY KEY (projection_id, market, line, captured_at)
    );

    CREATE TABLE IF NOT EXISTS postgame_why_v0 (
        game_id TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        expected REAL,
        actual REAL,
        terms_json TEXT,
        text TEXT,
        built_at TEXT,
        PRIMARY KEY (game_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS stints (
        game_id TEXT NOT NULL,
        stint_no INTEGER NOT NULL,
        period INTEGER NOT NULL,
        start_sec REAL NOT NULL,
        end_sec REAL NOT NULL,
        home5 TEXT NOT NULL,
        away5 TEXT NOT NULL,
        home_pts INTEGER, away_pts INTEGER,
        home_poss REAL, away_poss REAL,
        PRIMARY KEY (game_id, stint_no)
    );

    CREATE TABLE IF NOT EXISTS player_game_oncourt (
        game_id TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        team TEXT,
        started INTEGER,
        sec_q13 REAL, sec_q4 REAL, sec_ot REAL, sec_garbage REAL,
        on_poss REAL, opp_on_poss REAL,
        team_fga_on INTEGER, team_fgm_on INTEGER, team_misses_on INTEGER,
        opp_misses_on INTEGER, tm_fgm_on INTEGER, opp_tov_on INTEGER, opp_2pa_on INTEGER,
        PRIMARY KEY (game_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS rate_state (
        as_of TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        stat TEXT NOT NULL,
        rate REAL,
        n_eff REAL,
        prior REAL,
        kappa REAL,
        PRIMARY KEY (as_of, player_id, stat)
    );

    CREATE TABLE IF NOT EXISTS player_metric_snapshots (
        as_of TEXT NOT NULL,
        model TEXT NOT NULL,
        metric TEXT NOT NULL,
        window TEXT NOT NULL,
        n INTEGER, value REAL, se REAL, details_json TEXT,
        PRIMARY KEY (as_of, model, metric, window)
    );

    CREATE TABLE IF NOT EXISTS dfs_metric_snapshots (
        as_of TEXT NOT NULL,
        model TEXT NOT NULL,
        metric TEXT NOT NULL,
        window TEXT NOT NULL,
        n INTEGER, value REAL, se REAL, details_json TEXT,
        PRIMARY KEY (as_of, model, metric, window)
    );
    """,
    # 10: DraftKings request log for the light-capture daily budget (sources/dfs_capture.py).
    """
    CREATE TABLE IF NOT EXISTS dk_requests (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        et_day TEXT NOT NULL,
        kind TEXT NOT NULL,
        ref TEXT,
        ok INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_dk_requests_day ON dk_requests(et_day, kind);
    """,
    # 11: starters per game (5 rows per team-game). Source: sportsdataverse lineups (first
    # period-1 lineup, 1997+), cross-checked with cdn boxscores and RotoGuru's starter flag.
    """
    CREATE TABLE IF NOT EXISTS player_starts (
        game_id TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        team TEXT,
        source TEXT NOT NULL,
        PRIMARY KEY (game_id, player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_player_starts_player ON player_starts(player_id, game_id);
    """,
    # 12: DFS tracking (owner request 2026-10-19: keep everything, grade every build, real
    # results). The owner's own entries, uploaded DK standings files, and per-lineup /
    # per-player grades written by the nightly DFS grading job.
    """
    CREATE TABLE IF NOT EXISTS dfs_uploads (
        upload_id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        contest_id TEXT,
        filename TEXT,
        sha256 TEXT NOT NULL UNIQUE,
        raw_path TEXT NOT NULL,
        rows INTEGER,
        uploaded_at TEXT NOT NULL,
        detail TEXT
    );

    CREATE TABLE IF NOT EXISTS dfs_entries (
        entry_id INTEGER PRIMARY KEY,
        contest_id TEXT,
        slate_id TEXT,
        dk_entry_id TEXT,
        lineup_id INTEGER,
        slots_json TEXT,
        entry_fee REAL,
        points REAL,
        rank INTEGER,
        field_size INTEGER,
        payout REAL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        note TEXT,
        UNIQUE (contest_id, dk_entry_id)
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_entries_slate ON dfs_entries(slate_id);

    CREATE TABLE IF NOT EXISTS dfs_lineup_grades (
        lineup_id INTEGER PRIMARY KEY,
        run_id TEXT NOT NULL,
        slate_id TEXT,
        graded_at TEXT NOT NULL,
        actual REAL,
        projected REAL,
        hindsight REAL,
        ratio_hindsight REAL,
        pct_random REAL,
        cash_line REAL,
        cash_line_source TEXT,
        would_cash INTEGER,
        est_rank INTEGER,
        est_payout REAL,
        detail_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_lineup_grades_slate ON dfs_lineup_grades(slate_id);

    CREATE TABLE IF NOT EXISTS player_projection_grades (
        projection_id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL,
        player_id INTEGER NOT NULL,
        run_id TEXT,
        graded_at TEXT NOT NULL,
        played INTEGER,
        actual_min REAL,
        actual_dk REAL,
        proj_min REAL,
        proj_dk REAL,
        pit REAL,
        in80 INTEGER,
        in50 INTEGER,
        crps REAL,
        dk_fppg REAL,
        salary INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_ppg_game ON player_projection_grades(game_id);
    """,
    # 13: contest-builder portfolios (proj/dfs/portfolio.py, T2.12 / T2.16). One row per build
    # (status building | ok | error); its lineups are dfs_lineups rows with run_id =
    # portfolio_id and mode 'gpp'. Kept out of dfs_runs (not a projection run).
    """
    CREATE TABLE IF NOT EXISTS dfs_portfolios (
        portfolio_id TEXT PRIMARY KEY,
        slate_id TEXT NOT NULL,
        contest_id TEXT,
        run_id TEXT,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        n_requested INTEGER,
        n_built INTEGER,
        settings_json TEXT,
        summary_json TEXT,
        seconds REAL,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dfs_portfolios_slate ON dfs_portfolios(slate_id, created_at);
    """,
]

GAME_COLUMNS = ("espn_id", "season", "nba_day", "tipoff_utc", "home", "away",
                "status", "home_score", "away_score")
ODDS_COLUMNS = ("game_id", "captured_at", "source", "provider", "home_spread",
                "total", "home_ml", "away_ml", "raw_path")
INJURY_COLUMNS = ("captured_at", "source", "team", "player_name", "player_id",
                  "status", "reason", "raw_path")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; return the current schema version."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version "
                 "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    current = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
    for version, sql in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        conn.executescript("BEGIN;\n" + sql + "\nCOMMIT;")
        conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                     (version, _now()))
        conn.commit()
    return max(current, len(MIGRATIONS))


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    if path is None:
        config.ensure_dirs()
        path = config.db_path()
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _check(fields: dict, allowed: tuple[str, ...], extra: tuple[str, ...] = ()) -> None:
    bad = set(fields) - set(allowed) - set(extra)
    if bad:
        raise TypeError(f"unknown column(s): {sorted(bad)}")


def upsert_game(conn: sqlite3.Connection, **fields) -> None:
    """Insert or update a game by game_id; None-valued fields never overwrite existing data."""
    game_id = fields.pop("game_id", None)
    if not game_id:
        raise ValueError("game_id is required")
    _check(fields, GAME_COLUMNS, ("updated_at",))
    fields.pop("updated_at", None)
    given = {k: v for k, v in fields.items() if v is not None}
    cols = ["game_id", *given, "updated_at"]
    vals = [game_id, *given.values(), _now()]
    sets = ", ".join(f"{c}=excluded.{c}" for c in [*given, "updated_at"])
    conn.execute(
        f"INSERT INTO games ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
        f"ON CONFLICT(game_id) DO UPDATE SET {sets}", vals)
    conn.commit()


def insert_odds_snapshot(conn: sqlite3.Connection, **fields) -> int:
    _check(fields, ODDS_COLUMNS)
    cols = list(fields)
    cur = conn.execute(
        f"INSERT INTO odds_snapshots ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        list(fields.values()))
    conn.commit()
    return cur.lastrowid


def insert_injury_snapshots(conn: sqlite3.Connection, rows: list[dict]) -> int:
    for r in rows:
        _check(r, INJURY_COLUMNS)
    conn.executemany(
        f"INSERT INTO injury_snapshots ({', '.join(INJURY_COLUMNS)}) "
        f"VALUES ({', '.join(':' + c for c in INJURY_COLUMNS)})",
        [{c: r.get(c) for c in INJURY_COLUMNS} for r in rows])
    conn.commit()
    return len(rows)


def upsert_rows(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """INSERT OR REPLACE dicts into `table`. Keys must be existing columns; missing keys -> NULL."""
    if not rows:
        return 0
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if not cols:
        raise ValueError(f"unknown table {table!r}")
    for r in rows:
        _check(r, tuple(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)})",
        [{c: r.get(c) for c in cols} for r in rows])
    conn.commit()
    return len(rows)


def insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """Plain INSERT of dicts into an append-only snapshot `table` (auto `id`). Keys must be
    existing columns (others raise TypeError); missing keys -> NULL."""
    if not rows:
        return 0
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[1] != "id"]
    if not cols:
        raise ValueError(f"unknown table {table!r}")
    for r in rows:
        _check(r, tuple(cols))
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
        [{c: r.get(c) for c in cols} for r in rows])
    conn.commit()
    return len(rows)


def start_run(conn: sqlite3.Connection, job: str) -> int:
    cur = conn.execute("INSERT INTO collect_runs (job, started_at) VALUES (?, ?)", (job, _now()))
    conn.commit()
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, ok: bool, detail: str | None = None) -> None:
    conn.execute("UPDATE collect_runs SET finished_at=?, ok=?, detail=? WHERE id=?",
                 (_now(), int(ok), detail, run_id))
    conn.commit()
