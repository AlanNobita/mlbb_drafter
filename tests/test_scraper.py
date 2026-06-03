import pytest
from server.data.scraper import LiquipediaScraper


class TestLiquipediaScraper:
    def test_init(self, tmp_path):
        scraper = LiquipediaScraper(output_dir=str(tmp_path))
        assert scraper.output_dir.exists()
        assert scraper.RATE_LIMIT_DELAY == 2.0
