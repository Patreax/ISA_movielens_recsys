import ast
import gzip
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm.auto import tqdm

from project2.config import INTERIM_DATA_DIR, STEAM_DATA_DIR

CACHE_DIR = INTERIM_DATA_DIR / "steam"
DEFAULT_CHUNK_SIZE = 250_000


def _cache_path(filepath: Path) -> Path:
    return CACHE_DIR / filepath.name.replace(".json.gz", ".parquet")


def _iter_jsonl_gz(
    filepath: Path,
    max_rows: int | None = None,
    allow_literal_fallback: bool = False,
) -> Iterator[dict]:
    """Yield parsed records from a line-oriented .json.gz file."""
    skipped = 0
    with gzip.open(filepath, "rb") as f:
        for i, line in enumerate(tqdm(f, desc=filepath.name, unit=" lines", leave=False)):
            if max_rows is not None and i >= max_rows:
                break

            line = line.strip()
            if not line:
                continue

            try:
                yield orjson.loads(line)
                continue
            except orjson.JSONDecodeError:
                if not allow_literal_fallback:
                    skipped += 1
                    continue

            try:
                yield ast.literal_eval(line.decode("utf-8"))
            except (ValueError, SyntaxError, UnicodeDecodeError):
                skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} malformed lines while parsing {filepath.name}")


def _serialize_nested(value):
    if isinstance(value, (list, dict)):
        return orjson.dumps(value).decode("utf-8")
    return value


def _iter_user_reviews(
    filepath: Path,
    max_rows: int | None,
    allow_literal_fallback: bool,
) -> Iterator[dict]:
    for entry in _iter_jsonl_gz(filepath, max_rows, allow_literal_fallback):
        user_id = entry.get("user_id")
        for review in entry.get("reviews", []):
            yield {
                "user_id": user_id,
                "item_id": str(review.get("item_id")),
                "recommend": review.get("recommend"),
                "review": review.get("review", ""),
                "posted": review.get("posted", ""),
                "helpful": review.get("helpful", ""),
                "funny": review.get("funny", ""),
            }


def _iter_user_items(
    filepath: Path,
    max_rows: int | None,
    allow_literal_fallback: bool,
) -> Iterator[dict]:
    for entry in _iter_jsonl_gz(filepath, max_rows, allow_literal_fallback):
        user_id = entry.get("user_id")
        for item in entry.get("items", []):
            yield {
                "user_id": user_id,
                "item_id": str(item.get("item_id")),
                "item_name": item.get("item_name"),
                "playtime_forever": item.get("playtime_forever"),
                "playtime_2weeks": item.get("playtime_2weeks"),
            }


def _iter_games(
    filepath: Path,
    max_rows: int | None,
    allow_literal_fallback: bool,
) -> Iterator[dict]:
    for entry in _iter_jsonl_gz(filepath, max_rows, allow_literal_fallback):
        if "id" in entry:
            entry["id"] = str(entry["id"])
        yield entry


def _iter_bundles(
    filepath: Path,
    max_rows: int | None,
    allow_literal_fallback: bool,
) -> Iterator[dict]:
    for entry in _iter_jsonl_gz(filepath, max_rows, allow_literal_fallback):
        yield {key: _serialize_nested(value) for key, value in entry.items()}


def _iter_steam_reviews(
    filepath: Path,
    max_rows: int | None,
    allow_literal_fallback: bool,
) -> Iterator[dict]:
    for entry in _iter_jsonl_gz(filepath, max_rows, allow_literal_fallback):
        if "product_id" in entry:
            entry["product_id"] = str(entry["product_id"])
        yield entry


def _iter_chunks(records: Iterable[dict], chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[list[dict]]:
    chunk: list[dict] = []
    for record in records:
        chunk.append(record)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _records_to_dataframe(records: Iterable[dict], chunk_size: int = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
    frames = [pd.DataFrame.from_records(chunk) for chunk in _iter_chunks(records, chunk_size)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_parquet_cache(
    cache: Path,
    records: Iterable[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = [pd.DataFrame.from_records(chunk) for chunk in _iter_chunks(records, chunk_size)]
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    # Columns with mixed scalar types (e.g. price: 4.99 vs "Free To Play") break parquet
    # schema inference — normalize to string. Skip list/dict columns (pyarrow handles them).
    for col in df.select_dtypes("object").columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if non_null.map(lambda x: isinstance(x, (list, dict))).mean() > 0.5:
            continue  # keep list/dict columns as-is
        if df[col].map(type).nunique() > 1:
            df[col] = df[col].astype(str)
    df.to_parquet(cache, index=False)
    return len(df)


def _load_or_cache(
    filepath: Path,
    record_builder: Callable[[Path, int | None, bool], Iterable[dict]],
    *,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    allow_literal_fallback: bool = True,
) -> pd.DataFrame:
    """Return cached Parquet if up-to-date, otherwise parse the source and cache it."""
    cache = _cache_path(filepath)
    cache_is_fresh = cache.exists() and cache.stat().st_mtime >= filepath.stat().st_mtime

    if max_rows is None and cache_is_fresh:
        logger.info(f"Loading from cache: {cache.name}")
        return pd.read_parquet(cache, columns=columns)

    if max_rows is None:
        logger.info(f"Building cache: {cache.name}")
        row_count = _write_parquet_cache(
            cache,
            record_builder(filepath, None, allow_literal_fallback),
        )
        logger.info(f"Cached {row_count:,} rows to {cache}")
        if not cache.exists():
            raise RuntimeError(
                f"Cache file was not created for {filepath.name}. "
                "All lines may have failed to parse — check the file format."
            )
        return pd.read_parquet(cache, columns=columns)

    df = _records_to_dataframe(
        record_builder(filepath, max_rows, allow_literal_fallback),
    )
    if columns is not None and not df.empty:
        keep = [column for column in columns if column in df.columns]
        df = df.loc[:, keep]
    return df


def load_user_reviews(
    data_dir: Path | None = None,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    allow_literal_fallback: bool = True,
) -> pd.DataFrame:
    """Load australian_user_reviews and flatten nested reviews into one row per review."""
    data_dir = data_dir or STEAM_DATA_DIR
    return _load_or_cache(
        data_dir / "australian_user_reviews.json.gz",
        _iter_user_reviews,
        max_rows=max_rows,
        columns=columns,
        allow_literal_fallback=allow_literal_fallback,
    )


def load_user_items(
    data_dir: Path | None = None,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    allow_literal_fallback: bool = True,
) -> pd.DataFrame:
    """Load australian_users_items and flatten nested items into one row per (user, item)."""
    data_dir = data_dir or STEAM_DATA_DIR
    return _load_or_cache(
        data_dir / "australian_users_items.json.gz",
        _iter_user_items,
        max_rows=max_rows,
        columns=columns,
        allow_literal_fallback=allow_literal_fallback,
    )


def load_games(
    data_dir: Path | None = None,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    allow_literal_fallback: bool = True,
) -> pd.DataFrame:
    """Load steam_games metadata."""
    data_dir = data_dir or STEAM_DATA_DIR
    return _load_or_cache(
        data_dir / "steam_games.json.gz",
        _iter_games,
        max_rows=max_rows,
        columns=columns,
        allow_literal_fallback=allow_literal_fallback,
    )


def load_bundles(
    data_dir: Path | None = None,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    allow_literal_fallback: bool = True,
) -> pd.DataFrame:
    """Load bundle_data."""
    data_dir = data_dir or STEAM_DATA_DIR
    return _load_or_cache(
        data_dir / "bundle_data.json.gz",
        _iter_bundles,
        max_rows=max_rows,
        columns=columns,
        allow_literal_fallback=allow_literal_fallback,
    )


def load_steam_reviews(
    data_dir: Path | None = None,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    allow_literal_fallback: bool = True,
) -> pd.DataFrame:
    """Load steam_reviews (aggregated review data with hours played)."""
    data_dir = data_dir or STEAM_DATA_DIR
    return _load_or_cache(
        data_dir / "steam_reviews.json.gz",
        _iter_steam_reviews,
        max_rows=max_rows,
        columns=columns,
        allow_literal_fallback=allow_literal_fallback,
    )


def load_all_datasets(
    data_dir: Path | None = None,
    max_rows: int | None = None,
# ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all five Steam datasets and return them as a tuple
    """
    data_dir = data_dir or STEAM_DATA_DIR
    df_reviews   = load_user_reviews(data_dir, max_rows=max_rows)
    df_items     = load_user_items(data_dir, max_rows=max_rows)
    df_games     = load_games(data_dir, max_rows=max_rows)
    # df_bundles   = load_bundles(data_dir, max_rows=max_rows)
    # df_steam_rev = load_steam_reviews(data_dir, max_rows=max_rows)
    df_games["price_numeric"] = parse_price(df_games["price"])
    logger.info("All datasets loaded.")
    # return df_reviews, df_items, df_games, df_bundles, df_steam_rev
    return df_reviews, df_items, df_games


def parse_price(series: pd.Series) -> pd.Series:
    """Convert a price column like '$4.99' or 4.99 to float, NaN if missing."""
    text = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    free_mask = text.str.lower().isin(["free", "free to play", "none", "nan", ""])
    return pd.to_numeric(text.mask(free_mask, "0"), errors="coerce")
