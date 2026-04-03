# Exposing the Safety Socket via Tailscale Funnel

Tailscale Funnel makes a local service reachable from the public internet through Tailscale's network, with automatic HTTPS. No port forwarding, no DNS configuration, no certificate management.

## Prerequisites

1. **Tailscale installed on the Mac Mini** and logged into your tailnet.
   Verify: `tailscale status` should show the machine as connected.

2. **Funnel enabled in your tailnet ACL policy.** In the Tailscale admin console (https://login.tailscale.com/admin/acls), the ACL must include a `nodeAttrs` entry allowing funnel:
   ```json
   "nodeAttrs": [
     {
       "target": ["autogroup:member"],
       "attr": ["funnel"]
     }
   ]
   ```
   If you use the default ACL, funnel may already be enabled. Check under Access Controls.

3. **Flask server running** on port 5050 (`make serve` or the launchd agent).

## Quick Start

```bash
# Expose port 5050 via Funnel (HTTPS, public internet)
tailscale funnel 5050
```

That's it. Tailscale will:
- Provision a Let's Encrypt TLS certificate automatically
- Make the server available at `https://<hostname>.<tailnet>.ts.net`
- Proxy HTTPS traffic to your local HTTP server on port 5050

The hostname is your machine's Tailscale name (visible in `tailscale status`).

## Verify

From any device (phone, laptop, different network):
```bash
curl https://<hostname>.<tailnet>.ts.net/health
# Should return: {"status": "ok", "services": [...]}
```

Or open the URL in a browser — you should see the Safety Socket input form.

## Run Funnel in the Background

By default `tailscale funnel` runs in the foreground. To persist it:

```bash
# Run as a background process
tailscale funnel --bg 5050

# Check status
tailscale funnel status

# Stop
tailscale funnel --bg off
```

Alternatively, `tailscale serve` with `--set-path` can configure persistent forwarding that survives reboots (managed by Tailscale's own daemon).

## Tailnet-Only Alternative

If you don't want public internet access — only devices on your tailnet:

```bash
# Expose to tailnet only (not the public internet)
tailscale serve 5050
```

Same HTTPS certificate, same URL, but only reachable from machines signed into your tailnet. Good for internal testing before going public.

## Security Considerations

When using Funnel (public internet):

- **Anyone with the URL can access the prototype.** There is no authentication. The URL is not guessable (it's your tailnet hostname), but if shared, anyone can use it.
- **Rate limiting** is in place (10 requests/minute, configurable in `config.yaml`).
- **GPU queue** limits concurrent evaluations to 3 queued requests.
- **Prompt length** is capped at 5,000 characters.
- **Security headers** (CSP, X-Frame-Options, nosniff) are set.
- **No secrets leak** — error messages to the client are generic; exception details are logged server-side only.

If you need authentication before public exposure, options include:
1. Use `tailscale serve` (tailnet-only) instead of `funnel`
2. Add HTTP Basic Auth via a reverse proxy (nginx/caddy) in front of Flask
3. Add a simple password gate in the Flask app itself

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `funnel` command not found | Update Tailscale: `brew upgrade tailscale` or download latest |
| "Funnel not available" error | Enable funnel in ACL policy (see Prerequisites step 2) |
| Certificate errors | Wait 30-60 seconds for Let's Encrypt provisioning |
| Connection refused | Verify Flask is running: `curl http://localhost:5050/health` |
| Timeout on first request | Normal — first request loads the model (~30s). Subsequent requests are faster. |
