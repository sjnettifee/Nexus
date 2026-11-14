# Who can use your Nexus server: access, API keys, and building your own apps

Nexus is meant to be used from more than one place. You might chat in a browser on another computer, run the code hardener against servers on two machines, or build your own app on top of it. This page explains what's allowed by default, how to add an API key, and how to connect your own app.

---

## The short version

- **On one computer, nothing to set up.** The launcher, the web chat, the code hardener and your own apps all just work.
- **On a home network, also nothing to set up.** Set the launcher's *Who can connect* to *Anyone on my network*. Other computers can then open the web chat and run the hardener or your apps against it.
- **Internet websites can't use your models.** Your browser can reach your Nexus server, so a website you visit could try to send it requests. Nexus refuses them.
- **Optional API key.** Want other computers to prove they're allowed? Set a key. The computer running Nexus never needs it.

---

## What's allowed

The rules below are about **web pages**, the only thing a browser limits. **Programs** (Python scripts, curl, the code hardener, another Nexus) don't run in a browser, so these rules don't apply to them. Only the API key does, when one is set and the program is on another computer.

| Request comes from | Default |
|---|---|
| Any program, on any computer | Allowed (needs the key from other computers, if one is set) |
| Nexus's own web chat, on this computer or another one | Allowed |
| Your own web app served from this computer (e.g. `http://localhost:5173`) | Allowed |
| Your own web app served from your local network: private addresses (192.168.x.x, 10.x.x.x, 172.16-31.x.x), Tailscale (100.64-127.x.x), plain computer names (`http://desktop:8000`), `.local` / `.lan` / `.home.arpa` names | Allowed |
| A page on the internet (`https://some-site.com`), whether it uses `fetch`, a form or a hidden image | **Refused** |
| A website that points its own name at your computer to look like your server ("DNS rebinding") | **Refused** |

**Why the last two matter.** Before this, any website open in your browser while Nexus was running could quietly chat with your model, read your conversations and change its memory.

---

## Settings

### In the launcher

**Network** card (one setting for all models):

- **Who can connect:**
  - *Only this computer*: the server listens on 127.0.0.1 only.
  - *Anyone on my network*: it listens on all addresses.
- **API key for other computers:** empty means no key. **Make a key** creates a random one, **Copy key** copies it, **No key** removes it.
- **Web pages that may use it:**
  - *This computer and my network* (default).
  - *This computer only*.
  - *Any website* (not recommended; this is how Nexus behaved before).
- **Also allow web pages from:** extra addresses, comma-separated, for example `http://192.168.1.20:5173, https://myapp.example.com`.

Changes apply when a model is started or restarted. The key is handed to the server through an environment variable, so it doesn't appear in the process list.

### Server flags (starting the server yourself)

| Flag | Meaning |
|---|---|
| `--host 0.0.0.0` | Listen on the network (default `127.0.0.1`: this computer only). |
| `--api-key <key>` | Require this key from other computers. Or set the environment variable `NEXUS_API_KEY`. |
| `--cors lan` | Default. Web pages from this computer and the local network. |
| `--cors local` | Web pages from this computer only. |
| `--cors any` | Every website. Old behaviour, not recommended. |
| `--allow-origin <url>` | Also allow this page address. Repeat it, or comma-separate several. `--allow-origin '*'` is the same as `--cors any`. |
| `--enable-web-tools` | Turn on `/api/search`, `/api/search_detail`, `/api/search_detail_by_url` (fetching web pages for a model). Off by default. |

---

## The API key

When a key is set:

- Requests from **other computers** must include it, in any one of three ways:
  - `Authorization: Bearer <key>`
  - `X-API-Key: <key>`
  - `?key=<key>` in the URL, for links and downloads
- Requests from **the computer running Nexus** never need it.
- **The web chat** asks for the key the first time it's opened from another computer, and remembers it in that browser. To change it, clear the site data for that page, or run `localStorage.removeItem('nexus_api_key')` in the browser console.
- **The code hardener** has an *API key* box for each server in its Models card. Fill it in for servers on other computers that have a key.

A wrong or missing key gets `401` with `{"ok":false,"need_key":true,"error":"..."}`.

The key travels as plain text over plain HTTP. That's fine on a network you control. Across the internet, put Nexus behind HTTPS (a reverse proxy, or Tailscale) rather than exposing the port directly.

---

## Building your own app

### Easiest: a program

Any language works, because programs aren't limited by browser rules.

```bash
# one-off answer, streamed (Server-Sent Events)
curl -N http://192.168.1.10:8080/api/chat \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer nx-yourkey' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

```python
import json, urllib.request
req = urllib.request.Request(
    "http://192.168.1.10:8080/api/chat",
    data=json.dumps({"messages": [{"role": "user", "content": "Hello"}]}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer nx-yourkey"})
with urllib.request.urlopen(req) as r:
    for line in r:                                   # read line by line
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            ev = json.loads(line[6:])
            print(ev.get("token", ""), end="", flush=True)
```

### A web page

A page served from this computer or your local network can call Nexus directly with no setup:

```js
const r = await fetch('http://192.168.1.10:8080/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer nx-yourkey'},  // only if a key is set
  body: JSON.stringify({messages: [{role: 'user', content: 'Hello'}]})
});
const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
for (;;) {
  const {value, done} = await reader.read(); if (done) break;
  buf += dec.decode(value, {stream: true});
  let i; while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
      const ev = JSON.parse(line.slice(6));
      if (ev.token) document.body.append(ev.token);
    }
  }
}
```

A page hosted on the internet needs its address added under *Also allow web pages from*, or `--allow-origin https://your.app`.

### Events in the chat stream

| Event | Meaning |
|---|---|
| `{"conv_id":"..."}` | Conversation id. Send it back as `conv_id` to continue the same conversation with `/api/chat`. |
| `{"progress":{"pct":34.2,"tokens":98304,"tps":104.2,"eta":580}}` | The prompt is being read. `tps` is 0 while still measuring, `eta` is -1 when unknown. |
| `{"token":"...","gen":N,"context_tokens":M}` | Text. `gen` is the real number of tokens generated so far, `context_tokens` the tokens in memory. |
| `{"final":true,"gen":N,"context_tokens":M}` | Exact totals for the reply. |
| `{"notice":"..."}` | Something worth showing the user, such as the memory safety limit. |
| `[DONE]` | End of the reply. |

### Endpoints you'll use most

| Endpoint | Use |
|---|---|
| `POST /api/chat` | Messages in, streamed reply out. With `conv_id`, a continuation only reads what's new. |
| `POST /api/session/extend_turn` then `POST /api/continue` | Add a user message to what the model already remembers, then stream the reply. This is what the web chat does after the first message, and it also works after loading a saved session. |
| `GET /api/progress` | While a prompt is being read (for example during `extend_turn`): `{"active":true,"pct":..,"tokens":..,"tps":..,"eta":..}`, otherwise `{"active":false}`. |
| `POST /api/stop` | Stop the reply being written. |
| `GET /api/info` | Model name, context length, model 2 details. |
| `GET /api/context` | Tokens in memory, limit, remaining. |
| `POST /api/session/save`, `/api/session/load` | Save or restore the whole conversation memory. |
| `/api/kv/*` | Forget, archive and restore parts of the conversation. See [Conversation memory](CONVERSATION_MEMORY.md). |

Add `"model":"2"` to use the second model when the server runs two.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| The server log shows `[access] refused POST /api/chat from 192.168.1.20: web pages from https://… may not use this server` | Your page is served from an internet address. Add it under *Also allow web pages from* (or `--allow-origin`). |
| `refused: this server was addressed as "nexus.example.com"…` | You reached the server through a domain name, and Nexus can't tell that apart from DNS rebinding. Add that name with `--allow-origin`. Plain IPs, computer names and `.local` names always work. |
| `401` / *needs its API key* | Another computer is calling without the key, or with the wrong one. |
| Works on this computer but not from another one | Set *Who can connect* to *Anyone on my network*, and check the firewall allows the port. |

The `[access] refused …` lines in the server log always give the reason.
