# Local Mermaid Preview Setup

## Install node deps

Install live-server for viewing in browser and the mermaid tool for exporting.

```bash
npm install -g live-server
npm install -g @mermaid-js/mermaid-cli
```

---

## Run live server (with npm bin in PATH)

```bash
export PATH="$(npm config get prefix)/bin:$PATH"
live-server
```

This starts a local server with auto-refresh for your Mermaid diagrams.

## Export to an image

This exports at four times the standard resolution.

```bash
mmdc -i diagram.mmd -o diagram.png -s 4 -w 2000
```
