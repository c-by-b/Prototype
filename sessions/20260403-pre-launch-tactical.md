# Session: 2026-04-03 (afternoon) — Pre-Launch Tactical

## What We Did

Four tactical items to prepare the Prototype for public access, plus server hardening.

### Task 1: Compliance Summary Per Round in UI

The compliance step enriches the action_summary each round, but the UI never displayed it. Added:
- `renderCompliance()` function in `templates/index.html` — shows the enriched action summary (open by default) after the Cognitive Twin Proposal section, before Evidence
- Revision tracking table with color-coded status badges (Fully Addressed → green/approve palette, Partially Addressed → amber/revise, Not Addressed → red/veto)
- Compliance timing in the round timings line

### Task 2: Security Hardening

- **SECRET_KEY**: Random per startup via `secrets.token_hex(32)`
- **HSTS header**: `Strict-Transport-Security: max-age=31536000`
- **Permissions-Policy header**: `camera=(), microphone=(), geolocation=()`
- **Contract ID**: Changed from predictable timestamp to UUID4 (non-enumerable)
- **requirements.txt**: Pinned all dependencies from venv

Decision: No authentication layer. Rate limiting + URL secrecy is sufficient for this deployment.

### Task 3: Tailscale Funnel

Configured `tailscale funnel --bg 5050` for persistent public access at `https://c-by-b-mini.tail31d611.ts.net/`. The funnel config is stored in tailscaled's state and persists across reboots.

### Task 4: c-by-b.ai Website Update

Replaced the Namecheap Website Builder generated page with a clean static `index.html`:
- Two-column layout: 2/3 intro text + architecture diagram, 1/3 resources
- Fonts: Recursive for headers/logo, Inter for body text
- Prototype launch button (prominent, top of resources column)
- Links: Zenodo paper, Dev Notes blog, GitHub, PoC Streamlit app
- PoC demo video embedded
- Preserved: SEO meta tags, Open Graph, Plausible analytics, contact email, privacy notice

Deployment: uploaded `index.html` to `public_html/` on StellarPlus, copied `gallery/` assets from `ncsitebuilder/gallery/`, updated `.htaccess` DirectoryIndex to prioritize `index.html`.

### Server Hardening: Boot Persistence

Discovered that `nathanmeyer`'s LaunchAgents don't load on reboot because `macmini` is the auto-login user. Fixed:
- Moved Flask Prototype plist to `/Library/LaunchDaemons/com.cbyb.prototype.plist` with `UserName: nathanmeyer` so it runs as the correct user at boot
- Added `/opt/homebrew/bin` to plist PATH
- Fixed permissions on `logs/` and `results/` directories (group-writable for staff)
- Caddy was already a system LaunchDaemon at `/Library/LaunchDaemons/local.caddy.plist`
- Verified full boot chain survives reboot: Tailscale → Funnel → Caddy → Flask all come up

Power management confirmed: `sleep 0`, `autorestart 1`, `womp 1`.

## Design Decisions Made

| Decision | Value | Rationale |
|----------|-------|-----------|
| Authentication | None | Rate limiting + URL secrecy sufficient for prototype |
| Contract ID format | UUID4 | Prevents enumeration of /contract/<id> endpoint |
| Website hosting | Namecheap StellarPlus (existing) | Simpler than self-hosting on Mini alongside Prototype |
| Website body font | Inter | Recursive too playful for body text; reserved for brand/headers |
| Website layout | 2/3 + 1/3 columns | Resources sidebar always visible |
| Flask auto-start | LaunchDaemon with UserName | Runs at boot regardless of login, correct user permissions |

## Outstanding Items

1. **SSH keepalive** — Nathan's SSH sessions drop overnight. Likely client-side timeout, not Mini sleeping. Fix: add `ServerAliveInterval 60` to laptop's `~/.ssh/config`.
2. **SMB mount drops** — Finder mounts to Mini are fragile over network hiccups. Separate from the server uptime issue.

## Files Created or Modified

| File | Change |
|------|--------|
| `templates/index.html` | Added `renderCompliance()` function, compliance timing in round timings |
| `static/style.css` | Added compliance summary, revision tracking table, status badge styles |
| `cbyb/app.py` | Added SECRET_KEY, HSTS, Permissions-Policy headers |
| `cbyb/coordinator/contract.py` | Changed contract_id to UUID4 |
| `requirements.txt` | **New** — pinned dependencies |
| `docs/com.cbyb.prototype.plist` | Added UserName key, updated PATH |
| `/Library/LaunchDaemons/com.cbyb.prototype.plist` | **New** (system) — Flask auto-start |
| `/Users/nathanmeyer/Dev/C-by-B-code/Website/index.html` | **New** — c-by-b.ai landing page |

## Test Count

118 tests passing, 0 failed (Prototype). Boot chain verified via reboot.
