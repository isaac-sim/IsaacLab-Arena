# Local Mermaid Preview Setup

## Install live-server

```bash
npm install -g live-server
```

---

## Run live server (with npm bin in PATH)

```bash
export PATH="$(npm config get prefix)/bin:$PATH"
live-server
```

This starts a local server with auto-refresh for your Mermaid diagrams.
