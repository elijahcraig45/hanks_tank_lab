"""The Caddy site fronts everything the lab serves. Two mistakes here are silent, so they are pinned."""
import re
from pathlib import Path

CADDY = (Path(__file__).resolve().parents[1] / "host" / "Caddyfile.mllab").read_text()


def test_the_site_matches_any_host_but_binds_loopback_only():
    # `tailscale serve` forwards the tailnet hostname as Host. A site addressed to 127.0.0.1 does not match it and Caddy answers an empty 200: a blank page.
    assert re.search(r"^http://:8899 \{", CADDY, re.M)
    assert not re.search(r"^http://127\.0\.0\.1:8899", CADDY, re.M)
    assert re.search(r"^\s+bind 127\.0\.0\.1$", CADDY, re.M)


def test_everything_but_healthz_goes_through_the_login_check_first():
    assert CADDY.index("handle /healthz") < CADDY.index("forward_auth")
    after = CADDY[CADDY.index("forward_auth"):]
    for route in ("@app path /app /app/*", "handle /app-api/*", "handle /api/*", "reverse_proxy 127.0.0.1:8898"):
        assert route in after, route


def test_the_app_routes_come_before_the_catch_all_that_goes_to_jupyter():
    assert CADDY.index("handle /app-api/*") < CADDY.rindex("reverse_proxy 127.0.0.1:8898")
    assert CADDY.index("@app path /app /app/*") < CADDY.rindex("reverse_proxy 127.0.0.1:8898")


def test_handle_takes_one_matcher_and_a_named_matcher_carries_several_paths():
    # `handle /a /b {` is a parse error ("wrong argument count"): Caddy refused to start and the whole lab went dark. Several paths need a named matcher.
    for line in CADDY.splitlines():
        m = re.match(r"^\s*handle(?:_path)?\s+(.*?)\s*\{\s*$", line)
        if m:
            assert len(m.group(1).split()) <= 1, f"handle with several arguments: {line.strip()!r}"
    assert "@app path /app /app/*" in CADDY and "handle @app {" in CADDY


def test_nothing_is_published_publicly():
    assert "funnel" not in CADDY.lower().replace("never publish with funnel", "")
    assert "0.0.0.0" not in CADDY


def test_the_root_redirects_to_the_app_with_an_explicit_matcher():
    # `redir /app/ 302` reads /app/ as a path matcher and 302 as the destination: it never fires (and would send /app/ to "302"). A single
    # destination needs the `*` matcher in front of it.
    assert re.search(r"^\s+redir \* /app/$", CADDY, re.M)
    assert not re.search(r"redir /app/ \d+", CADDY)
