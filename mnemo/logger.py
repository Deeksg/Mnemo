# mnemo/logger.py
#
# The Interaction Logger — permanently records every pipeline run
# into a SQLite database (mnemo.db).
#
# WHY SQLITE:
# Built into Python, no installation needed, no separate server
# to run. Stores everything in one file on disk. Perfect for a
# single-user local project like this — Weaviate (Phase 5) will
# be the proper multi-user database later.
#
# TWO TABLES:
#
# 1. interactions — one row per pipeline RUN.
#    Holds the full nested result as a JSON blob (full_record),
#    plus a few commonly-needed fields (task, final_output) as
#    real columns for fast access without parsing JSON every time.
#
# 2. stage_results — one row PER STAGE within a pipeline run.
#    research, writing, and fact_checking all share this ONE
#    table — a stage_name column tells rows apart. This is what
#    makes queries like "which agent wins research most often"
#    possible with real SQL instead of manually looping through
#    JSON in Python every time we want an answer.
#
# This is the foundation for:
# - Phase 4 (GNN) querying interaction history to train on
# - Phase 5 (Weaviate) using these records as retrievable memory
# - Phase 6 (Blockchain) hashing these records into a Merkle tree

import sqlite3
import json
from datetime import datetime


class InteractionLogger:
    """
    Handles all permanent storage of pipeline interaction records.

    This is a CLASS (not just functions) because every method
    needs to know the same db_path. Wrapping db_path in self
    means we set it once when creating the logger, and every
    method automatically knows which database file to use —
    instead of passing db_path as an argument to every single
    function call.

    Creates and manages two SQLite tables on first use.
    """

    def __init__(self, db_path: str = "mnemo.db"):
        # Store the path so every method below can reach it
        # via self.db_path without needing it passed in again
        self.db_path = db_path

        # Create both tables if they don't already exist
        # Safe to call every time the logger starts
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Creates both tables if they don't already exist.
        Called automatically every time InteractionLogger is created.

        CREATE TABLE IF NOT EXISTS only creates the table the
        FIRST time this ever runs. On every later run, this line
        does nothing because the table already exists. This means
        it's always safe to call, even every single time the
        program starts — no risk of wiping existing data.
        """

        # Connect to the database file.
        # Creates mnemo.db if it doesn't exist yet.
        connection = sqlite3.connect(self.db_path)

        # Cursor is what actually executes SQL commands —
        # think of connection as "the file is open" and
        # cursor as "your hands actually typing inside it"
        cursor = connection.cursor()

        # ── TABLE 1: interactions ──────────────────────────
        # One row per full pipeline run.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task TEXT NOT NULL,
                final_output TEXT,
                exploration_mode INTEGER,
                full_record TEXT NOT NULL
            )
        """)
        # id                 — auto-numbered by SQLite, never repeats
        # timestamp          — when this interaction happened
        # task               — the original task text, real column
        #                      for fast searching without JSON parsing
        # final_output       — winning output of the LAST stage,
        #                      duplicated here from full_record
        #                      purely for fast browsing access
        # exploration_mode   — 1 or 0 (SQLite has no true boolean,
        #                      so we store True/False as 1/0)
        # full_record        — the ENTIRE result dictionary,
        #                      converted to one long JSON text
        #                      string. This is the "blob" column —
        #                      holds everything, searchable by
        #                      nothing directly, but flexible
        #                      for any nested structure

        # ── TABLE 2: stage_results ─────────────────────────
        # One row per STAGE within a pipeline run.
        # research, writing, fact_checking ALL share this table —
        # the stage_name column is what tells rows apart.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stage_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                stage_name TEXT NOT NULL,
                winner_agent TEXT NOT NULL,
                quality_score INTEGER,
                relevance_score INTEGER,
                completeness_score INTEGER,
                overall_score INTEGER,
                used_fallback INTEGER,
                parse_error INTEGER,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        """)
        # interaction_id     — links this stage row back to its
        #                      parent row in the interactions table.
        #                      This is called a FOREIGN KEY —
        #                      it's how two separate tables stay
        #                      connected to each other. One
        #                      interaction can have many
        #                      stage_results rows (one per stage).
        # stage_name         — "research", "writing", "fact_checking"
        #                      or whatever a client names their stages
        # winner_agent       — which agent won THIS specific stage
        # quality/relevance/completeness/overall_score
        #                    — the winner's scores from the judge,
        #                      pulled out as real numeric columns
        #                      so SQL can AVG() and COUNT() them
        # used_fallback      — was this evaluation done by the
        #                      backup judge (Gemini) instead of
        #                      the primary (Llama 70B)?
        # parse_error        — did the judge's JSON response fail
        #                      to parse, forcing neutral 5/10 scores?


        # ── TABLE 3: agent_scores ──────────────────────────
        # One row per AGENT per STAGE per interaction.
        # Unlike stage_results which only records the winner,
        # this table records EVERY agent's scores from every
        # stage they competed in. This gives the GNN rich
        # signal — an agent that consistently scores 8 but
        # loses to a 9 is still valuable information.
        # Also used to track how many times each agent has
        # competed (total_appearances) for exploration mode.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                stage_name TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                quality_score INTEGER,
                relevance_score INTEGER,
                completeness_score INTEGER,
                overall_score INTEGER,
                won INTEGER,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        """)
        # won — 1 if this agent won this stage, 0 if they lost
        # Allows filtering wins without a separate table

        # Save all 3 table creations permanently to disk
        connection.commit()

        # Close this connection — we open a fresh one for every
        # operation rather than keeping one open the whole time.
        # Safer for SQLite, especially once multiple parts of
        # Mnemo might read/write the database.
        connection.close()

    def log_interaction(self, result: dict) -> int:
        """
        Saves one complete pipeline result permanently.
        Inserts ONE row into interactions, and ONE row PER STAGE
        into stage_results.

        Parameters
        ----------
        result : dict
            The complete result dictionary returned by
            MnemoCore.run() — contains task, stages, final_output,
            exploration_mode, interaction_counts.

        Returns
        -------
        int
            The id assigned to this interaction by the database.
            Useful for later retrieval — e.g. the memory layer
            in Phase 5 linking a memory back to its source
            interaction.
        """

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # Build the values for our "real" columns
        timestamp = datetime.now().isoformat()
        task = result.get("task", "")
        final_output = result.get("final_output", "")

        # exploration_mode is True/False in Python but SQLite has
        # no true boolean type — it stores booleans as 0 or 1.
        # We convert explicitly here so it's intentional and clear
        # rather than relying on automatic conversion.
        exploration_mode = 1 if result.get("exploration_mode") else 0

        # json.dumps() converts the ENTIRE result dictionary
        # (every nested stage, output, score) into one long
        # JSON text string — this is the "blob" column value
        full_record = json.dumps(result)

        # ── INSERT INTO interactions ───────────────────────
        # Notice the ? placeholders instead of writing values
        # directly into the SQL string. This is called a
        # PARAMETERIZED QUERY and it matters for two reasons:
        #
        # 1. Security — if inserted text contains special SQL
        #    characters (like a quote mark), directly embedding
        #    it could corrupt the command or allow SQL injection.
        #    ? placeholders let SQLite handle escaping safely.
        #
        # 2. Correctness — task descriptions could contain
        #    apostrophes or quotes that would break a manually
        #    built SQL string.
        cursor.execute("""
            INSERT INTO interactions
            (timestamp, task, final_output, exploration_mode, full_record)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, task, final_output, exploration_mode, full_record))

        # cursor.lastrowid gives us the id SQLite automatically
        # assigned to the row we just inserted (AUTOINCREMENT).
        # We need this BEFORE inserting stage_results rows,
        # since each of them needs to reference this id.
        interaction_id = cursor.lastrowid

        # ── INSERT INTO stage_results — one row per stage ──
        # Loop through every stage in the result. result["stages"]
        # is a dict like {"research": {...}, "writing": {...}, ...}
        # .items() gives us both the stage name and its data together.
        for stage_name, stage_data in result.get("stages", {}).items():

            evaluation = stage_data.get("evaluation", {})
            winner = evaluation.get("winner", "")

            # Pull out just the WINNER's individual scores from
            # the nested scores dict inside the evaluation.
            # evaluation["scores"] looks like:
            # {"agent_a": {...}, "agent_b": {...}, "agent_c": {...}}
            # We only want the winner's scores for this row.
            winner_scores = evaluation.get("scores", {}).get(winner, {})

            used_fallback = 1 if evaluation.get("used_fallback") else 0
            parse_error = 1 if evaluation.get("parse_error") else 0

            cursor.execute("""
                INSERT INTO stage_results
                (interaction_id, stage_name, winner_agent,
                 quality_score, relevance_score, completeness_score,
                 overall_score, used_fallback, parse_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,       # links back to parent row
                stage_name,           # "research", "writing", etc.
                winner,                # winning agent's name
                winner_scores.get("quality"),
                winner_scores.get("relevance"),
                winner_scores.get("completeness"),
                winner_scores.get("overall"),
                used_fallback,
                parse_error
            ))

        # ── INSERT INTO agent_scores — one row per agent ───
        # Store EVERY agent's scores, not just the winner.
        # This is what makes meaningful reputation calculation
        # possible — we need to know how everyone performed,
        # not just who came first.
        for stage_name, stage_data in result.get("stages", {}).items():
            evaluation = stage_data.get("evaluation", {})
            winner = evaluation.get("winner", "")

            for agent_name, scores in evaluation.get("scores", {}).items():
                cursor.execute("""
                    INSERT INTO agent_scores
                    (interaction_id, stage_name, agent_name,
                     quality_score, relevance_score,
                     completeness_score, overall_score, won)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction_id,
                    stage_name,
                    agent_name,
                    scores.get("quality"),
                    scores.get("relevance"),
                    scores.get("completeness"),
                    scores.get("overall"),
                    1 if agent_name == winner else 0
                ))
                
        
        # Save everything permanently — the interactions row
        # AND all the stage_results rows, and all agent_scores rows at once
        connection.commit()
        connection.close()

        print(f"[Logger] Interaction #{interaction_id} saved "
              f"({len(result.get('stages', {}))} stage results)")

        return interaction_id

    def get_interaction(self, interaction_id: int) -> dict:
        """
        Retrieves one specific interaction by its id,
        including the full nested record.

        Parameters
        ----------
        interaction_id : int
            The id of the interaction to retrieve.

        Returns
        -------
        dict or None
            The full interaction record (as a real Python dict,
            not a JSON string), or None if not found.
        """

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # WHERE id = ? filters to just the one row we want.
        # Again using ? placeholder instead of embedding
        # interaction_id directly into the string.
        cursor.execute(
            "SELECT id, timestamp, task, final_output, "
            "exploration_mode, full_record "
            "FROM interactions WHERE id = ?",
            (interaction_id,)
        )

        # fetchone() actually RETRIEVES the single matching row
        # as a tuple. execute() only RUNS the search — fetchone()
        # is the separate step that hands back the actual result,
        # like a waiter bringing your order after you've placed it.
        # Returns None if no row matches this id.
        row = cursor.fetchone()

        connection.close()

        if row is None:
            return None

        # Tuple unpacking — splits the 6-value tuple into
        # 6 separate named variables in one line, in the exact
        # order we SELECTed them
        row_id, timestamp, task, final_output, exploration_mode, full_record = row

        # full_record is currently just TEXT (a JSON string).
        # json.loads() converts that text back into a real,
        # usable Python dictionary. We name this new dictionary
        # "record" because it now represents the fully restored
        # interaction record, ready to use in Python.
        record = json.loads(full_record)

        # Attach database metadata onto the record so whoever
        # receives it also knows which database row it came from
        record["_db_id"] = row_id
        record["_db_timestamp"] = timestamp

        return record

    def get_all_interactions(self, limit: int = 50) -> list:
        """
        Retrieves recent interaction SUMMARIES, newest first.
        Returns lightweight dicts (not the full nested record) —
        fast for browsing many interactions at once.
        Use get_interaction() for full detail on one specific one.

        Parameters
        ----------
        limit : int
            Maximum number of interactions to return.
            Default 50 — prevents accidentally loading
            thousands of records into memory at once.

        Returns
        -------
        list of dict
            Lightweight summaries, ordered newest first.
        """

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # ORDER BY id DESC — newest interactions first
        # (id auto-increments, so higher id = more recent)
        # LIMIT ? — caps how many rows come back.
        # LIMIT is SQL syntax (not Python) — it means
        # "only give me this many rows." The Python variable
        # `limit` fills in the ? placeholder with a number,
        # same placeholder pattern used everywhere else.
        cursor.execute(
            "SELECT id, timestamp, task, final_output, exploration_mode "
            "FROM interactions ORDER BY id DESC LIMIT ?",
            (limit,)
        )

        # fetchall() gets ALL matching rows at once,
        # as a list of tuples
        rows = cursor.fetchall()

        connection.close()

        # Build one dictionary per row using a list comprehension.
        # For each tuple in rows, unpack it into named variables,
        # then build a dictionary from those names.
        # bool(exploration_mode) converts the stored 1/0 back
        # into a real Python True/False.
        return [
            {
                "id": row_id,
                "timestamp": timestamp,
                "task": task,
                "final_output": final_output,
                "exploration_mode": bool(exploration_mode)
            }
            for row_id, timestamp, task, final_output, exploration_mode in rows
        ]

    def get_agent_history(self, agent_name: str, limit: int = 50) -> list:
        """
        Retrieves past interactions involving a specific agent.

        This will become important in Phase 5 — when an agent
        runs, Mnemo retrieves its own past interactions as
        memory context. This method is the foundation for that.

        Note: this does a Python-side filter rather than a SQL
        WHERE clause, because agent names live inside the JSON
        blob (full_record), not as their own column. At our scale
        (hundreds, not millions of interactions) this is completely
        fine performance-wise.

        Parameters
        ----------
        agent_name : str
            The agent's name, e.g. "researcher-gemini"

        limit : int
            Maximum number of interactions to return

        Returns
        -------
        list of dict
            Full interactions where this agent participated,
            newest first.
        """

        # Fetch full records (not just summaries) for recent
        # interactions, then filter for this specific agent
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute(
            "SELECT id, timestamp, task, final_output, "
            "exploration_mode, full_record "
            "FROM interactions ORDER BY id DESC LIMIT 200",
        )
        rows = cursor.fetchall()
        connection.close()

        matching = []
        for row_id, timestamp, task, final_output, exploration_mode, full_record in rows:
            record = json.loads(full_record)
            record["_db_id"] = row_id
            record["_db_timestamp"] = timestamp

            # Check if this agent appears in ANY stage's outputs
            # for this interaction
            for stage_name, stage_data in record.get("stages", {}).items():
                if agent_name in stage_data.get("outputs", {}):
                    matching.append(record)
                    break  # found it, no need to check other stages

            if len(matching) >= limit:
                break

        return matching

    def get_stage_winners(self, stage_name: str = None) -> list:
        """
        Returns how many times each agent won a given stage,
        using a real SQL query — possible ONLY because of the
        stage_results table existing separately from the JSON blob.

        Parameters
        ----------
        stage_name : str, optional
            Filter to one specific stage e.g. "research".
            If None, returns win counts across ALL stages combined.

        Returns
        -------
        list of dict
            e.g. [
                {"agent": "researcher-gemini", "wins": 12},
                {"agent": "researcher-llama", "wins": 5},
                {"agent": "researcher-qwen", "wins": 3}
            ]
            Ordered by most wins first.
        """

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        if stage_name:
            # GROUP BY winner_agent groups all rows that share
            # the same winner together into one bucket each.
            # COUNT(*) counts how many rows fell into each bucket
            # — i.e. how many times that agent won.
            cursor.execute("""
                SELECT winner_agent, COUNT(*) as wins
                FROM stage_results
                WHERE stage_name = ?
                GROUP BY winner_agent
                ORDER BY wins DESC
            """, (stage_name,))
        else:
            # No stage filter — count wins across every stage combined
            cursor.execute("""
                SELECT winner_agent, COUNT(*) as wins
                FROM stage_results
                GROUP BY winner_agent
                ORDER BY wins DESC
            """)

        rows = cursor.fetchall()
        connection.close()

        return [
            {"agent": agent, "wins": wins}
            for agent, wins in rows
        ]


    def count_interactions(self) -> int:
        """
        Returns the total number of interactions logged so far.
        Useful for quick stats and for future exploration vs
        exploitation decisions.
        """

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # COUNT(*) is a SQL function that counts matching rows.
        # No WHERE clause means it counts EVERY row in the table.
        cursor.execute("SELECT COUNT(*) FROM interactions")

        # fetchone() returns a tuple even for a single value,
        # e.g. (47,) — so we take [0] to get just the number itself
        count = cursor.fetchone()[0]

        connection.close()

        return count

    def get_agent_appearance_count(self, agent_name: str) -> int:
        """
        Returns how many times this agent has competed in any stage.
        Used by the pipeline to track exploration vs exploitation.
        Simple count of appearances — not wins, not scores.
        An agent competes in every stage of every pipeline run,
        so after 3 full runs every agent has 9 appearances
        (3 stages x 3 runs).
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM agent_scores
            WHERE agent_name = ?
        """, (agent_name,))

        count = cursor.fetchone()[0]
        connection.close()
        return count

    def get_agent_average_scores(self, agent_name: str) -> dict:
        """
        Returns this agent's average scores across ALL competitions
        — including stages they lost. This is a more honest
        reputation signal than averaging only winning scores.

        Returns
        -------
        dict:
            {
                "agent": "researcher-gemini",
                "total_appearances": 9,
                "total_wins": 3,
                "win_rate": 33.3,
                "avg_quality": 7.8,
                "avg_relevance": 8.1,
                "avg_completeness": 7.5,
                "avg_overall": 7.8
            }
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute("""
            SELECT COUNT(*) as total_appearances,
                   SUM(won) as total_wins,
                   AVG(quality_score) as avg_quality,
                   AVG(relevance_score) as avg_relevance,
                   AVG(completeness_score) as avg_completeness,
                   AVG(overall_score) as avg_overall
            FROM agent_scores
            WHERE agent_name = ?
        """, (agent_name,))

        row = cursor.fetchone()
        connection.close()

        total_appearances, total_wins, avg_quality, avg_relevance, avg_completeness, avg_overall = row

        win_rate = 0.0
        if total_appearances:
            win_rate = round((total_wins / total_appearances) * 100, 1)

        return {
            "agent": agent_name,
            "total_appearances": total_appearances or 0,
            "total_wins": int(total_wins) if total_wins else 0,
            "win_rate": win_rate,
            "avg_quality": round(avg_quality, 2) if avg_quality else 0,
            "avg_relevance": round(avg_relevance, 2) if avg_relevance else 0,
            "avg_completeness": round(avg_completeness, 2) if avg_completeness else 0,
            "avg_overall": round(avg_overall, 2) if avg_overall else 0
        }