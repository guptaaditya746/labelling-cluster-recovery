# Infrastructure

Argilla and its stores (Elasticsearch, PostgreSQL, Redis) run in Docker on **Omen**.
Scripts that push candidates and export gold run on the **Slurm node**. Browsers
(you, your professor) open Argilla on the **laptop**. `omen-tunnels` connects all three.

## 1. Start Argilla on Omen

```sh
mkdir -p ~/stacks/argilla && cd ~/stacks/argilla
cp /path/to/labelling/infra/argilla/compose.yaml .
cp /path/to/labelling/infra/argilla/.env.example .env   # set passwords and the API key
docker compose up -d
docker compose logs -f argilla                           # wait for "Uvicorn running"
```

Or add the same `compose.yaml` and `.env` as a stack in Dockge.
Needs about 3-4 GB RAM and a few GB of disk. Most start-up problems come
from Elasticsearch; `docker compose logs elasticsearch` shows them.

## 2. Add the tunnel

The harness copy of the tunnel helper (`rambodambo/scripts/omen-tunnels`) already lists the
service. Install it on **both** the laptop and the compute node
(`install -m 755 scripts/omen-tunnels ~/.local/bin/omen-tunnels` from the harness checkout), or add
this line to `SERVICES` in an older copy:

```python
    "argilla": (6900, 26900, 46900, 36900),
```

Restart the tunnels (laptop first):

```sh
omen-tunnels laptop stop  && omen-tunnels laptop start all     # on the laptop
omen-tunnels compute stop && omen-tunnels compute start all    # on the compute node
```

- Laptop browser: <http://127.0.0.1:26900>
- Slurm node (scripts): `ARGILLA_API_URL=http://127.0.0.1:36900`

## 3. Accounts

Log in as the owner from `.env`, then create an annotator for your professor
(Argilla UI → Users, role `annotator`, add to workspace `thesis`), or from Python:

```python
import argilla as rg
client = rg.Argilla(api_url="http://127.0.0.1:36900", api_key="...")
rg.User(username="professor", password="...", role="annotator", client=client).create()
client.workspaces("thesis").add_user("professor")
```

The tunnel listens on the laptop's localhost only. Your professor reviews either on
your laptop (in person or screen share) or through their own SSH tunnel to Omen.
