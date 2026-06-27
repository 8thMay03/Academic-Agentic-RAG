from app.models.paper import Paper


class MetadataStore:
    async def list_papers(self) -> list[Paper]:
        # TODO: Persist metadata to JSON or SQLite.
        return []

