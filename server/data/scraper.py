from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List
from dataclasses import asdict

import aiohttp
from bs4 import BeautifulSoup

from server.data.loader import HeroMetadata


class LiquipediaScraper:
    BASE_URL = "https://liquipedia.net/mobilelegends/"
    HEADERS = {"User-Agent": "MLBBDrafter/1.0 (Educational Project)"}
    RATE_LIMIT_DELAY = 2.0

    def __init__(self, output_dir: str = "server/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0.0
        self.session: Optional[aiohttp.ClientSession] = None

    async def _rate_limit(self) -> None:
        current = time.time()
        elapsed = current - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    async def _fetch(self, url: str) -> str:
        await self._rate_limit()
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.HEADERS)
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def scrape_hero_list(self) -> List[str]:
        url = f"{self.BASE_URL}Heroes"
        html = await self._fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        hero_links = []
        for link in soup.select("a[href*='/mobilelegends/']"):
            href = link.get("href", "")
            if "/mobilelegends/" in href and not any(
                x in href for x in ["Heroes", "Special", "File:"]
            ):
                name = href.split("/")[-1]
                if name not in hero_links and len(name) > 1:
                    hero_links.append(name)
        return hero_links

    async def scrape_hero_details(self, hero_name: str) -> Optional[HeroMetadata]:
        url = f"{self.BASE_URL}{hero_name}"
        try:
            html = await self._fetch(url)
        except Exception as e:
            print(f"Failed to scrape {hero_name}: {e}")
            return None

        soup = BeautifulSoup(html, "html.parser")
        infobox = soup.select_one(".fo-nttax-infobox-wrapper")
        if not infobox:
            return None

        data = {"name": hero_name, "role": "Unknown", "roles": [], "lanes": []}

        role_elem = infobox.find(string=lambda t: t and "Role" in t)
        if role_elem and role_elem.parent:
            val = role_elem.parent.find_next_sibling(class_="infobox-cell-2")
            if val:
                data["role"] = val.get_text(strip=True)
                data["roles"] = [r.strip() for r in data["role"].split("/")]

        return HeroMetadata(
            id=hash(hero_name) % 10000,
            name=hero_name,
            real_name=hero_name,
            role=data["role"],
            roles=data["roles"],
            lanes=data["lanes"],
            win_rate=50.0,
            pick_rate=5.0,
            tier="B",
        )

    async def scrape_all_heroes(self, max_heroes: int = 50) -> List[HeroMetadata]:
        hero_names = await self.scrape_hero_list()
        print(f"Found {len(hero_names)} heroes")
        heroes = []
        for i, name in enumerate(hero_names[:max_heroes]):
            print(f"[{i+1}/{min(len(hero_names), max_heroes)}] {name}...")
            hero = await self.scrape_hero_details(name)
            if hero:
                heroes.append(hero)
        return heroes

    async def save_heroes(
        self, heroes: List[HeroMetadata], filename: str = "hero_meta.json"
    ) -> None:
        output = self.output_dir / filename
        with open(output, "w") as f:
            json.dump([asdict(h) for h in heroes], f, indent=2)
        print(f"Saved {len(heroes)} heroes to {output}")

    async def close(self) -> None:
        if self.session:
            await self.session.close()


async def main():
    scraper = LiquipediaScraper()
    try:
        heroes = await scraper.scrape_all_heroes(max_heroes=10)
        await scraper.save_heroes(heroes)
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
