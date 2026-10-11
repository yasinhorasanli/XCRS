"""Dependencies shared by the API's routers (tests override `get_session`)."""

from collections.abc import Iterator

from sqlalchemy.orm import Session

from xcrs.db.session import new_session


def get_session() -> Iterator[Session]:
    """One session per request. Use it as `Depends(get_session, scope="function")`: the session closes, and its
    connection goes back to the pool, as soon as the endpoint function returns. With the default "request" scope
    the connection stays checked out until the response is sent, and for synchronous endpoints FastAPI validates
    the response in a worker thread first. Under overload, requests holding all pooled connections then queue for
    a worker thread behind requests that hold threads while they wait for a connection, and the API starves for
    minutes (ADR-0050, docs/incidents/2026-10-11-api-wedged-under-overload.md)."""
    with new_session() as session:
        yield session
