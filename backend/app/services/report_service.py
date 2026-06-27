class ReportService:
    async def generate_survey(self, title: str, summaries: list[str], comparison: str) -> str:
        # TODO: Generate final survey report with LLM.
        return f"# {title}\n\n## Summaries\n\n" + "\n\n".join(summaries) + f"\n\n## Comparison\n\n{comparison}"

