"""Dependencies shared by the API's routers (tests override `get_session`)."""

from collections.abc import Iterator

from sqlalchemy.orm import Session

from xcrs.db.session import new_session


def get_session() -> Iterator[Session]:
    with new_session() as session:
        yield session
