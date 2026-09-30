"""SQLite source ownership and durable, independent event delivery.

The JSONL recorder is the producer. A cursor and an immutable HTTP payload are
committed together, so restarting this service cannot skip an imported event.
This module deliberately has no inference or training dependencies.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import ssl
import stat
import sys
import threading
import time
from typing import Any, Iterator
from urllib.parse import urlsplit
import uuid


DEFAULT_DB = "/data/Eating_behavior/events/events.sqlite3"
DEFAULT_ENDPOINT = "https://app.hzzootech.com/api/open/eatingBehavior/events"
MAX_LINE = 1024 * 1024
ACTIVE = ("pending", "retry", "inflight", "blocked_auth", "blocked_contract")
POSTURES = {
    "eating": "cat_eating", "lying": "cat_lying",
    "standing": "cat_standing", "crouching": "cat_crouching",
}


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(field + " must be a nonempty string of at most 128 characters")
    if value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(field + " contains whitespace or control characters")
    return value


def event_type(record: dict[str, Any]) -> str | None:
    if record.get("event_type") == "cat_passed":
        return "cat_appeared"
    if record.get("event_type") == "cat_posture_changed":
        details = record.get("details")
        if not isinstance(details, dict) or details.get("posture") not in POSTURES:
            raise ValueError("unsupported_posture")
        return POSTURES[details["posture"]]
    # Completed eating intervals and bowl placeholders are not backend alerts.
    return None


def source_url(value: str) -> str:
    p = urlsplit(value)
    if p.scheme not in {"rtmp", "rtmps"} or not p.hostname or p.fragment:
        raise ValueError("source must be an RTMP/RTMPS URL")
    if any(c.isspace() for c in value):
        raise ValueError("source URL contains whitespace")
    return value


def anchor(handle: Any, offset: int) -> str:
    handle.seek(max(0, offset - 256))
    return hashlib.sha256(handle.read(min(256, offset))).hexdigest()


def file_position(path: Path, *, beginning: bool = False) -> tuple[int, int, int, str]:
    try:
        with path.open("rb") as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("event file must be regular")
            offset = 0
            if not beginning:
                f.seek(max(0, info.st_size - MAX_LINE))
                tail = f.read()
                last = tail.rfind(b"\n")
                if last < 0 and info.st_size >= MAX_LINE:
                    raise ValueError("event line exceeds size limit")
                offset = info.st_size - len(tail) + last + 1
            return info.st_dev, info.st_ino, offset, anchor(f, offset)
    except FileNotFoundError:
        return 0, 0, 0, hashlib.sha256(b"").hexdigest()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY CHECK(user_id > 0),
    username TEXT NOT NULL UNIQUE CHECK(length(username) > 0), created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(user_id),
    rtmp_url TEXT NOT NULL UNIQUE, event_file TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    revision INTEGER NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cursors (
    source_id TEXT PRIMARY KEY REFERENCES sources(source_id),
    device INTEGER NOT NULL, inode INTEGER NOT NULL, offset INTEGER NOT NULL,
    anchor TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL REFERENCES sources(source_id), user_id INTEGER NOT NULL CHECK(typeof(user_id)='integer' AND user_id > 0),
    revision INTEGER NOT NULL, event_type TEXT NOT NULL, payload TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    next_at REAL NOT NULL, lease_until REAL, lease_token TEXT,
    http_status INTEGER, error_code TEXT, created_at REAL NOT NULL,
    event_time REAL, sent_at REAL
);
CREATE INDEX IF NOT EXISTS outbox_due ON outbox(status, next_at);
CREATE INDEX IF NOT EXISTS outbox_source ON outbox(source_id, seq, status);
CREATE TABLE IF NOT EXISTS ingest_errors (
    source_id TEXT NOT NULL, offset INTEGER NOT NULL,
    reason TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY(source_id, offset)
);
PRAGMA user_version=2;
"""


def backend_user_id(value: Any) -> int:
    if type(value) is not int or not 0 < value <= 9223372036854775807:
        raise ValueError("user_id must be a positive integer within SQLite range")
    return value


class EventStore:
    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists() and not create:
            raise FileNotFoundError("event database does not exist; run init first")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 1:
                raise ValueError("legacy user IDs require explicit migrate-users mapping")
            if version not in (0, 2):
                raise ValueError("unsupported event database version")
            db.execute("PRAGMA journal_mode=WAL")
            for statement in SCHEMA_SQL.split(";"):
                if statement.strip():
                    db.execute(statement)
        self.path.chmod(0o600)

    @staticmethod
    def migrate_users(path: str | Path, mappings: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Explicit v1 identity correction, with backup and no cursor reset.

        Only never-attempted or explicitly unauthorized active events can have
        their payload corrected. Unknown delivery outcomes require reconciliation.
        Terminal payloads remain unchanged as historical evidence.
        """
        path = Path(path).resolve(strict=True)
        if not isinstance(mappings, dict) or not mappings:
            raise ValueError("explicit legacy user mapping is required")
        for old, account in mappings.items():
            identifier(old, "legacy_user_id")
            if not isinstance(account, dict) or set(account) != {"user_id", "username"}:
                raise ValueError("mapping must contain user_id and username")
            backend_user_id(account["user_id"])
            identifier(account["username"], "username")
        if len({a["user_id"] for a in mappings.values()}) != len(mappings) or len({a["username"] for a in mappings.values()}) != len(mappings):
            raise ValueError("duplicate mapped user IDs or usernames")
        with path.with_suffix(".sender.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("stop the sender before user migration") from None
            db = sqlite3.connect(path, timeout=5)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=OFF")
            try:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise ValueError("migration requires database schema version 1")
                users = db.execute("SELECT * FROM users").fetchall()
                if {r["user_id"] for r in users} != set(mappings):
                    raise ValueError("mapping must cover exactly the existing users")
                if db.execute("""SELECT 1 FROM outbox WHERE status IN
                    ('pending','retry','inflight','blocked_auth','blocked_contract')
                    AND NOT COALESCE((status='pending' AND attempts=0) OR
                      (status='blocked_auth' AND
                       (http_status IN (401,403) OR error_code IN ('backend_code_401','backend_code_403'))),0)""").fetchone():
                    raise ValueError("reconcile uncertain delivery outcomes before identity correction")
                # A separate read connection sees the committed snapshot while
                # this write transaction prevents another writer changing it.
                backup = path.with_name(path.stem + ".v1-backup-" + uuid.uuid4().hex + ".sqlite3")
                backup.touch(mode=0o600, exist_ok=False)
                source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
                target = sqlite3.connect(backup)
                try:
                    source.backup(target)
                finally:
                    target.close()
                    source.close()
                for table in ("users", "sources", "outbox"):
                    prefix = "CREATE TABLE IF NOT EXISTS " + table + " ("
                    statement = next(s.strip() for s in SCHEMA_SQL.split(";") if s.strip().startswith(prefix))
                    db.execute(statement.replace(prefix, "CREATE TABLE " + table + "_v2 (", 1))
                for row in users:
                    account = mappings[row["user_id"]]
                    db.execute("INSERT INTO users_v2 VALUES (?,?,?)", (account["user_id"], account["username"], row["created_at"]))
                for row in db.execute("SELECT * FROM sources"):
                    values = dict(row)
                    if row["user_id"] is not None:
                        values["user_id"] = mappings[row["user_id"]]["user_id"]
                    db.execute("INSERT INTO sources_v2 VALUES (?,?,?,?,?,?,?)", tuple(values.values()))
                corrected = 0
                for row in db.execute("SELECT * FROM outbox"):
                    values = dict(row)
                    values["user_id"] = mappings[row["user_id"]]["user_id"]
                    if row["status"] in ACTIVE:
                        payload = json.loads(row["payload"])
                        if not isinstance(payload, dict) or payload != {"event_id": row["event_id"], "user_id": row["user_id"], "event_type": row["event_type"]}:
                            raise ValueError("stored event identity conflicts with mapping")
                        payload["user_id"] = values["user_id"]
                        values["payload"] = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                        corrected += 1
                    placeholders = ",".join("?" for _ in values)
                    db.execute("INSERT INTO outbox_v2 VALUES (" + placeholders + ")", tuple(values.values()))
                for table in ("outbox", "sources", "users"):
                    db.execute("DROP TABLE " + table)
                for table in ("users", "sources", "outbox"):
                    db.execute("ALTER TABLE " + table + "_v2 RENAME TO " + table)
                for statement in SCHEMA_SQL.split(";"):
                    if statement.strip():
                        db.execute(statement)
                if db.execute("PRAGMA foreign_key_check").fetchone():
                    raise ValueError("migration foreign key check failed")
                db.commit()
                return {"schema_version": 2, "users": len(users), "corrected_pending": corrected, "backup": str(backup)}
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def endpoint(self, endpoint: str) -> None:
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO settings VALUES ('endpoint', ?)", (endpoint,))
            if db.execute("SELECT value FROM settings WHERE key='endpoint'").fetchone()[0] != endpoint:
                raise ValueError("endpoint differs from database; use a separate database per environment")

    def bind(self, source: str, user: int, rtmp: str, event_file: str | Path, *, username: str) -> None:
        identifier(source, "source_id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", source):
            raise ValueError("invalid source_id")
        backend_user_id(user)
        identifier(username, "username")
        source_url(rtmp)
        path = Path(event_file)
        if not path.is_absolute() or path.suffix != ".jsonl":
            raise ValueError("event file must be an absolute JSONL path")
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM sources WHERE source_id=?", (source,)).fetchone()
            db.execute("""INSERT INTO users (user_id,username,created_at) VALUES (?,?,?)
                ON CONFLICT(user_id) DO UPDATE SET username=excluded.username""", (user, username, now))
            if previous and (previous["user_id"], previous["rtmp_url"], previous["event_file"], previous["enabled"]) == (user, rtmp, str(path), 1):
                return
            if db.execute("SELECT 1 FROM outbox WHERE source_id=? AND status='inflight' AND lease_until>?", (source, now)).fetchone():
                raise ValueError("source has an in-flight request; retry binding after it finishes")
            revision = previous["revision"] + 1 if previous else 1
            db.execute("""INSERT INTO sources VALUES (?,?,?,?,1,?,?)
                ON CONFLICT(source_id) DO UPDATE SET user_id=excluded.user_id,
                rtmp_url=excluded.rtmp_url,event_file=excluded.event_file,enabled=1,
                revision=excluded.revision,updated_at=excluded.updated_at""",
                (source, user, rtmp, str(path), revision, now))
            db.execute("UPDATE outbox SET status='cancelled',error_code='mapping_changed' WHERE source_id=? AND status IN (?,?,?,?,?)", (source, *ACTIVE))
            # Registration starts at the current complete-line boundary: no old
            # footage or previous-owner events are silently sent to a new user.
            position = file_position(path)
            db.execute("INSERT OR REPLACE INTO cursors VALUES (?,?,?,?,?)", (source, *position))

    def disable(self, source: str) -> None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM outbox WHERE source_id=? AND status='inflight' AND lease_until>?", (source, time.time())).fetchone():
                raise ValueError("source has an in-flight request")
            if not db.execute("UPDATE sources SET enabled=0,revision=revision+1 WHERE source_id=?", (source,)).rowcount:
                raise ValueError("source not registered")
            db.execute("UPDATE outbox SET status='cancelled',error_code='source_disabled' WHERE source_id=? AND status IN (?,?,?,?,?)", (source, *ACTIVE))

    def stream(self, source: str) -> str:
        return self.binding(source)["rtmp_url"]

    def binding(self, source: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT s.*,u.username FROM sources s JOIN users u ON u.user_id=s.user_id WHERE s.source_id=? AND s.enabled=1", (source,)).fetchone()
            if row is None:
                raise ValueError("source has no enabled user/RTMP mapping")
            return dict(row)

    def sources(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT s.source_id,s.user_id,u.username,s.enabled,s.revision FROM sources s LEFT JOIN users u ON u.user_id=s.user_id ORDER BY s.source_id")]

    def _enqueue(self, db: sqlite3.Connection, source: sqlite3.Row, record: dict[str, Any], now: float) -> bool:
        if not isinstance(record, dict) or record.get("source") != source["source_id"]:
            raise ValueError("record_source_mismatch")
        mapped = event_type(record)
        if mapped is None:
            return False
        eid = identifier(record.get("record_id"), "record_id")
        payload = json.dumps({"event_id": eid, "user_id": source["user_id"], "event_type": mapped}, separators=(",", ":"), ensure_ascii=False)
        existing = db.execute("SELECT payload,source_id FROM outbox WHERE event_id=?", (eid,)).fetchone()
        if existing:
            if existing["payload"] != payload or existing["source_id"] != source["source_id"]:
                raise ValueError("event_id_conflict")
            return False
        event_time = record.get("created_at_unix")
        if isinstance(event_time, bool) or not isinstance(event_time, (int, float)) or not math.isfinite(event_time):
            event_time = None
        db.execute("""INSERT INTO outbox
            (event_id,source_id,user_id,revision,event_type,payload,status,next_at,created_at,event_time)
            VALUES (?,?,?,?,?,?,'pending',?,?,?)""",
            (eid, source["source_id"], source["user_id"], source["revision"], mapped, payload, now, now, event_time))
        return True

    def queue_record(self, source_id: str, record: dict[str, Any]) -> bool:
        """Explicit, audited one-record import; never used for automatic replay."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = db.execute("SELECT * FROM sources WHERE source_id=? AND enabled=1", (source_id,)).fetchone()
            if source is None:
                raise ValueError("source not enabled")
            return self._enqueue(db, source, record, time.time())

    def ingest(self, source_id: str, *, limit: int = 100, max_pending: int = 100000) -> int:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = db.execute("SELECT * FROM sources WHERE source_id=? AND enabled=1", (source_id,)).fetchone()
            if source is None:
                return 0
            # The producer must have loaded this exact registry revision. This
            # prevents an old worker's stream being assigned to a newly bound
            # owner before that worker has restarted with the new RTMP URL.
            status_path = Path(source["event_file"]).parent.parent / "status.json"
            try:
                producer = json.loads(status_path.read_bytes())
            except (OSError, ValueError):
                raise ValueError("producer registry status unavailable") from None
            if not isinstance(producer, dict) or (
                producer.get("source_id"), producer.get("source"),
                producer.get("source_registry_revision")
            ) != (source_id, source["rtmp_url"], source["revision"]):
                raise ValueError("producer registry mismatch; restart the assigned worker")
            pending = db.execute("SELECT count(*) FROM outbox WHERE status IN (?,?,?,?,?)", ACTIVE).fetchone()[0]
            if pending >= max_pending:
                raise BufferError("pending event limit reached; cursor retained")
            cursor = db.execute("SELECT * FROM cursors WHERE source_id=?", (source_id,)).fetchone()
            try:
                f = Path(source["event_file"]).open("rb")
            except FileNotFoundError:
                return 0
            with f:
                info = os.fstat(f.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("event file must be regular")
                if cursor["inode"] and (cursor["device"], cursor["inode"]) != (info.st_dev, info.st_ino):
                    raise ValueError("event_file_replaced; restore file or explicitly rebind")
                if info.st_size < cursor["offset"] or anchor(f, cursor["offset"]) != cursor["anchor"]:
                    raise ValueError("event_file_truncated_or_changed; cursor retained")
                f.seek(cursor["offset"])
                offset = cursor["offset"]
                added = 0
                for _ in range(min(limit, max_pending - pending)):
                    line = f.readline(MAX_LINE + 1)
                    if len(line) > MAX_LINE:
                        raise ValueError("event_line_too_large; cursor retained")
                    if not line.endswith(b"\n"):
                        break
                    next_offset = f.tell()
                    try:
                        record = json.loads(line)
                        added += int(self._enqueue(db, source, record, time.time()))
                    except (ValueError, TypeError, KeyError):
                        # Do not persist raw event bodies or URLs in diagnostics.
                        db.execute("INSERT OR IGNORE INTO ingest_errors VALUES (?,?,?,?)", (source_id, offset, "invalid_or_conflicting_event", time.time()))
                    offset = next_offset
                db.execute("UPDATE cursors SET device=?,inode=?,offset=?,anchor=? WHERE source_id=?", (info.st_dev, info.st_ino, offset, anchor(f, offset), source_id))
                return added

    def claim(self, *, now: float | None = None) -> dict[str, Any] | None:
        now = time.time() if now is None else now
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE outbox SET status='retry',lease_token=NULL,lease_until=NULL WHERE status='inflight' AND lease_until<=?", (now,))
            if db.execute("SELECT 1 FROM settings WHERE key='delivery_paused'").fetchone():
                return None
            db.execute("""UPDATE outbox SET status='cancelled',error_code='mapping_changed'
                WHERE status IN ('pending','retry') AND NOT EXISTS
                (SELECT 1 FROM sources s WHERE s.source_id=outbox.source_id
                 AND s.enabled=1 AND s.user_id=outbox.user_id AND s.revision=outbox.revision)""")
            row = db.execute("""SELECT o.* FROM outbox o
                WHERE o.status IN ('pending','retry') AND o.next_at<=?
                  AND NOT EXISTS (SELECT 1 FROM outbox earlier WHERE earlier.source_id=o.source_id
                    AND earlier.seq<o.seq AND earlier.status IN
                    ('pending','retry','inflight','blocked_auth','blocked_contract'))
                ORDER BY o.seq LIMIT 1""", (now,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            item["lease_token"] = uuid.uuid4().hex
            item["attempts"] += 1
            db.execute("UPDATE outbox SET status='inflight',attempts=?,lease_token=?,lease_until=? WHERE seq=?", (item["attempts"], item["lease_token"], now + 60, item["seq"]))
            return item

    def finish(self, item: dict[str, Any], result: DeliveryResult, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("""UPDATE outbox SET status=?,http_status=?,error_code=?,
                next_at=?,sent_at=?,lease_until=NULL,lease_token=NULL
                WHERE seq=? AND status='inflight' AND lease_token=?""",
                (result.state, result.http_status, result.reason, now + result.retry_after,
                 now if result.state == "delivered" else None, item["seq"], item["lease_token"])).rowcount
            if changed and result.state in {"blocked_auth", "blocked_contract"}:
                db.execute("INSERT OR REPLACE INTO settings VALUES ('delivery_paused',?)", (result.reason,))
            return bool(changed)

    def resume(self) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM settings WHERE key='delivery_paused'")
            db.execute("UPDATE outbox SET status='retry',next_at=? WHERE status IN ('blocked_auth','blocked_contract')", (time.time(),))

    def summary(self) -> dict[str, Any]:
        with self.connection() as db:
            counts = {r[0]: r[1] for r in db.execute("SELECT status,count(*) FROM outbox GROUP BY status")}
            paused = db.execute("SELECT value FROM settings WHERE key='delivery_paused'").fetchone()
            oldest = db.execute("SELECT min(created_at) FROM outbox WHERE status IN (?,?,?,?,?)", ACTIVE).fetchone()[0]
            return {"counts": counts, "paused": paused[0] if paused else None,
                    "oldest_pending_age_seconds": round(time.time() - oldest, 1) if oldest else 0,
                    "ingest_errors": db.execute("SELECT count(*) FROM ingest_errors").fetchone()[0]}


@dataclass(frozen=True)
class DeliveryResult:
    state: str
    http_status: int | None = None
    reason: str = ""
    retry_after: float = 0


def retry_delay(attempts: int, header: str | None = None) -> float:
    if header:
        try:
            delay = float(header)
            if math.isfinite(delay):
                return min(86400, max(1, delay))
        except ValueError:
            try:
                value = parsedate_to_datetime(header)
                return min(86400, max(1, value.timestamp() - time.time()))
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60, 2 ** min(attempts - 1, 6)) * random.uniform(0.8, 1.2)


class EventClient:
    def __init__(self, endpoint: str, api_key: str, *, allow_local_http: bool = False) -> None:
        p = urlsplit(endpoint)
        local_http = allow_local_http and p.scheme == "http" and p.hostname in {"127.0.0.1", "localhost", "::1"}
        if not (p.scheme == "https" or local_http) or not p.hostname or p.username or p.password or p.fragment or any(c.isspace() for c in endpoint):
            raise ValueError("endpoint must be an HTTPS URL without credentials or fragment")
        if not api_key or any(ord(c) < 33 for c in api_key):
            raise ValueError("missing or invalid API key")
        self.url = p
        self.api_key = api_key

    def send(self, item: dict[str, Any]) -> DeliveryResult:
        try:
            stored = json.loads(item["payload"])
            user_id = backend_user_id(stored["user_id"])
        except (ValueError, TypeError, KeyError):
            return DeliveryResult("blocked_contract", None, "invalid_user_id")
        p = self.url
        connection: http.client.HTTPConnection
        if p.scheme == "https":
            connection = http.client.HTTPSConnection(p.hostname, p.port or 443, timeout=5, context=ssl.create_default_context())
        else:
            connection = http.client.HTTPConnection(p.hostname, p.port or 80, timeout=5)
        try:
            path = (p.path or "/") + (("?" + p.query) if p.query else "")
            # Existing outbox rows keep their immutable internal field names.
            # Translate only at the HTTP boundary, preserving identity on retry.
            payload = {"eventId": stored["event_id"], "userId": user_id, "eventType": stored["event_type"]}
            request_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            connection.request("POST", path, body=request_body, headers={"X-API-Key": self.api_key, "Content-Type": "application/json", "Accept": "application/json"})
            response = connection.getresponse()
            status = response.status
            body = response.read(65537)
            if status in (200, 201) and len(body) <= 65536:
                try:
                    decoded = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    decoded = None
                if isinstance(decoded, dict) and "code" in decoded:
                    # This backend also returns business failures with HTTP 200.
                    # A transport success alone must never acknowledge an event.
                    code = decoded["code"]
                    if type(code) is not int:
                        return DeliveryResult("blocked_contract", status, "invalid_business_code")
                    if code not in (200, 201):
                        reason = "backend_code_" + str(code)
                        if code in (401, 403):
                            return DeliveryResult("blocked_auth", status, reason)
                        if code in (408, 429) or 500 <= code < 600:
                            return DeliveryResult("retry", status, reason, retry_delay(item["attempts"], response.getheader("Retry-After")))
                        if code in (409, 422):
                            return DeliveryResult("dead", status, reason)
                        return DeliveryResult("blocked_contract", status, reason)
                    # Accept the documented acknowledgement within an envelope,
                    # but still require success=true and the matching event ID.
                    if isinstance(decoded.get("data"), dict):
                        if ("eventId" in decoded and decoded["eventId"] != item["event_id"]) or ("success" in decoded and decoded["success"] is not True):
                            return DeliveryResult("blocked_contract", status, "response_contract")
                        decoded = decoded["data"]
                if isinstance(decoded, dict) and decoded.get("success") is True and decoded.get("eventId") == item["event_id"]:
                    return DeliveryResult("delivered", status)
                return DeliveryResult("blocked_contract", status, "response_contract")
            if status in (401, 403):
                return DeliveryResult("blocked_auth", status, "authentication_failed")
            if status in (404, 405) or 300 <= status < 400 or 200 <= status < 300:
                return DeliveryResult("blocked_contract", status, "unexpected_endpoint_response")
            if status in (408, 429) or 500 <= status < 600:
                return DeliveryResult("retry", status, "temporary_http_failure", retry_delay(item["attempts"], response.getheader("Retry-After")))
            return DeliveryResult("dead", status, "http_" + str(status))
        except (OSError, http.client.HTTPException):
            return DeliveryResult("retry", None, "network_failure", retry_delay(item["attempts"]))
        finally:
            connection.close()


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **values}, ensure_ascii=False), flush=True)


def serve(store: EventStore, client: EventClient) -> None:
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with store.path.with_suffix(".sender.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("a sender already owns this database") from None
        last_status = 0.0
        failures: dict[str, float] = {}
        while not stop.is_set():
            for source in store.sources():
                if not source["enabled"]:
                    continue
                try:
                    store.ingest(source["source_id"])
                    failures.pop(source["source_id"], None)
                except (ValueError, OSError, sqlite3.Error, BufferError) as exc:
                    key = source["source_id"]
                    if time.monotonic() - failures.get(key, -1000) >= 60:
                        emit("ingest_failed", source_id=key, error_type=type(exc).__name__)
                        failures[key] = time.monotonic()
            for _ in range(10):
                if stop.is_set():
                    break
                item = store.claim()
                if item is None:
                    break
                result = client.send(item)
                store.finish(item, result)
                emit("delivery", event_id=item["event_id"], state=result.state, http_status=result.http_status, reason=result.reason)
                if stop.wait(0.2):
                    break
            if time.monotonic() - last_status >= 30:
                emit("summary", **store.summary())
                last_status = time.monotonic()
            stop.wait(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("EATING_BEHAVIOR_EVENTS_DB", DEFAULT_DB))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("migrate-users", help="Read explicit legacy-ID to numeric-ID/username mapping JSON from stdin")
    bind = commands.add_parser("bind")
    bind.add_argument("--source-id", required=True)
    bind.add_argument("--user-id", type=int, required=True)
    bind.add_argument("--username", required=True)
    url = bind.add_mutually_exclusive_group(required=True)
    url.add_argument("--rtmp-url")
    url.add_argument("--rtmp-url-stdin", action="store_true")
    bind.add_argument("--event-file", required=True)
    disable = commands.add_parser("disable")
    disable.add_argument("--source-id", required=True)
    commands.add_parser("list")
    commands.add_parser("status")
    commands.add_parser("resume")
    commands.add_parser("serve")
    args = parser.parse_args()
    try:
        if args.command == "migrate-users":
            print(json.dumps(EventStore.migrate_users(args.db, json.load(sys.stdin)), ensure_ascii=False))
            return 0
        store = EventStore(args.db, create=args.command == "init")
        if args.command == "bind":
            store.bind(args.source_id, args.user_id, sys.stdin.readline().strip() if args.rtmp_url_stdin else args.rtmp_url, args.event_file, username=args.username)
        elif args.command == "disable":
            store.disable(args.source_id)
        elif args.command == "list":
            print(json.dumps(store.sources(), ensure_ascii=False))
        elif args.command == "status":
            print(json.dumps(store.summary(), ensure_ascii=False))
        elif args.command == "resume":
            store.resume()
        elif args.command == "serve":
            endpoint = os.environ.get("EATING_BEHAVIOR_EVENTS_URL", DEFAULT_ENDPOINT)
            client = EventClient(endpoint, os.environ.get("EATING_BEHAVIOR_API_KEY", ""))
            store.endpoint(endpoint)
            serve(store, client)
        return 0
    except (ValueError, OSError, sqlite3.Error, RuntimeError) as exc:
        # Do not print raw network exceptions, URLs, credentials or SQL values.
        emit("command_failed", error_type=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
