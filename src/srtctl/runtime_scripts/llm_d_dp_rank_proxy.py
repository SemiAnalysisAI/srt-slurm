# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One HTTP listener per SGLang attention-DP rank, so llm-d's EPP can route to ranks.

An SGLang server with attention DP serves every rank on one port and dispatches
requests to ranks itself, each rank with its own radix cache, so the EPP's prefix
affinity stops at the server. Each listener here forwards to the server with
``X-Data-Parallel-Rank`` (SGLang's external DP routing) and serves ``/metrics``
narrowed to its rank's ``dp_rank`` samples, so the EPP's load scorers see that rank.
Ranks a node does not export (a multi-node server's non-leader ranks) get the
server's metrics unfiltered. Standalone; runs in the SGLang image (aiohttp).

Usage: llm_d_dp_rank_proxy.py PORT=http://HOST:PORT#RANK [...]
"""

from __future__ import annotations

import argparse
import asyncio

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

RANK_HEADER = "X-Data-Parallel-Rank"
HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def rank_metrics(text: str, rank: int) -> str:
    label = f'dp_rank="{rank}"'
    lines = text.splitlines()
    if not any(label in line for line in lines if not line.startswith("#")):
        return text
    return "\n".join(line for line in lines if line.startswith("#") or "dp_rank=" not in line or label in line) + "\n"


def rank_app(session: ClientSession, upstream: str, rank: int) -> web.Application:
    async def handle(request: web.Request) -> web.StreamResponse:
        url = upstream + request.rel_url.raw_path_qs
        if request.method == "GET" and request.path == "/metrics":
            # Uncompressed: the session leaves bodies encoded so the proxied routes pass through as-is.
            async with session.get(url, headers={"Accept-Encoding": "identity"}) as upstream_response:
                return web.Response(text=rank_metrics(await upstream_response.text(), rank), content_type="text/plain")
        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_HEADERS}
        headers[RANK_HEADER] = str(rank)
        async with session.request(
            request.method, url, headers=headers, data=await request.read()
        ) as upstream_response:
            response = web.StreamResponse(
                status=upstream_response.status,
                headers={k: v for k, v in upstream_response.headers.items() if k.lower() not in HOP_HEADERS},
            )
            await response.prepare(request)
            async for chunk in upstream_response.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response

    app = web.Application(client_max_size=1 << 30)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


async def serve(routes: list[str]) -> None:
    session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0), auto_decompress=False)
    for route in routes:
        port, target = route.split("=", 1)
        upstream, rank = target.rsplit("#", 1)
        runner = web.AppRunner(rank_app(session, upstream.rstrip("/"), int(rank)), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", int(port)).start()
        print(f"dp-rank proxy :{port} -> {upstream} rank {rank}", flush=True)
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("routes", nargs="+", metavar="PORT=URL#RANK")
    asyncio.run(serve(parser.parse_args().routes))


if __name__ == "__main__":
    main()
