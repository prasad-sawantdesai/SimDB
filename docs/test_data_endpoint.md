# Testing the `/data` endpoint locally

This guide walks through running the new `GET /v1.2/simulation/<uuid>/data` endpoint
against a real IMAS HDF5 file on your local machine using the SimDB CLI.

All commands and config formats are taken directly from the
[maintenance guide](maintenance_guide.md) and [developer guide](developer_guide.md).

---

## Prerequisites

Install SimDB with server dependencies:

```bash
git clone https://github.com/iterorganization/SimDB.git
cd SimDB
pip install -e .[server]
```

This makes both `simdb` and `simdb_server` available on your PATH.

---

## Step 1 — Create the server config file

The server reads `app.cfg` from the same directory as the CLI config file.
Find that directory:

```bash
dirname "$(simdb config path)"
# e.g. /home/$USER/.config/simdb
```

Create `app.cfg` there (the server config — **different from** `simdb.cfg` which is the CLI config):

```bash
SERVER_CFG="$(dirname "$(simdb config path)")/app.cfg"
mkdir -p "$(dirname "$SERVER_CFG")"
mkdir -p ./simdb-local/simulations

cat > "$SERVER_CFG" << 'EOF'
[flask]
flask_env = development
debug = True
testing = True
secret_key = CHANGE_ME

[server]
upload_folder = ./simdb-local/simulations
ssl_enabled = False
admin_password = admin

[database]
type = sqlite
file = ./simdb-local/simdb.db

[authentication]
type = None

[development]
disable_checksum = True
EOF

chmod 600 "$SERVER_CFG"
```

> **Note:** `disable_checksum = True` skips checksum computation and IMAS file
> opening during ingest — useful for local testing. Remove it for a full ingest.

---

## Step 2 — Initialise the database

Run the Alembic migration to create the SQLite database schema (only needed once):

```bash
DATABASE_URL=sqlite:///simdb-local/simdb.db alembic upgrade head
```

---

## Step 3 — Start the server

In a dedicated terminal:

```bash
simdb_server
# → Serving on http://0.0.0.0:5000
```

Verify it is running:

```bash
curl -s http://localhost:5000/ | python -m json.tool
```

---

## Step 4 — Write a real IMAS HDF5 file

```python
import imas
import numpy as np

DB_PATH = "./simdb-local/test_core_profiles"

with imas.DBEntry(f"imas:hdf5?path={DB_PATH}", "w") as dbe:
    factory = imas.IDSFactory()
    ids = factory.core_profiles()

    ids.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    ids.ids_properties.comment = "local test"

    ids.time = np.array([1.0, 2.0, 3.0])
    ids.global_quantities.ip = np.array([1e6, 2e6, 3e6])

    ids.profiles_1d.resize(3)
    for i in range(3):
        ids.profiles_1d[i].grid.rho_tor_norm = np.linspace(0, 1, 5)
        ids.profiles_1d[i].ion.resize(i + 1)
        for j in range(i + 1):
            ids.profiles_1d[i].ion[j].element.resize(1)
            ids.profiles_1d[i].ion[j].element[0].a = float(j + 1)
            ids.profiles_1d[i].ion[j].density = np.array([10.0, 20.0, 30.0, 40.0, 50.0]) + j

    dbe.put(ids)

    # --- summary IDS ---
    summary = factory.summary()
    summary.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    summary.ids_properties.comment = "local test summary"

    summary.time = np.array([1.0, 2.0, 3.0])
    summary.global_quantities.ip.value = np.array([1e6, 2e6, 3e6])       # A
    summary.global_quantities.b0.value = np.array([5.3, 5.3, 5.3])       # T
    summary.global_quantities.r0.value = 6.2                              # m (scalar)
    summary.global_quantities.energy_mhd.value = np.array([1e7, 2e7, 3e7])  # J
    summary.global_quantities.v_loop.value = np.array([0.1, 0.2, 0.3])   # V

    dbe.put(summary)

print(f"Written: {DB_PATH}")
```

---

## Step 5 — Create a manifest and ingest locally

Create `manifest.yaml` (see [user guide](user_guide.md#local-simulation-management) for full schema):

```yaml
manifest_version: 2
alias: test-data-endpoint
inputs:
  - uri: imas:hdf5?path=./simdb-local/test_core_profiles
outputs:
  - uri: imas:hdf5?path=./simdb-local/test_core_profiles
metadata:
  - description: local test for /data endpoint
```

Ingest it into your local SimDB database:

```bash
simdb simulation ingest manifest.yaml
# ALIAS: test-data-endpoint
# UUID:  <uuid>
```

Save the UUID printed:

```bash
SIM_UUID=<uuid>
echo $SIM_UUID
```

---

## Step 6 — Configure a local remote and push

```bash
# Register localhost as a remote named "local"
simdb remote config new local http://localhost:5000/
simdb remote config set-default local

# Push the simulation to the running server
simdb simulation push $SIM_UUID
```

Confirm it arrived on the server:

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID" | python -m json.tool
```

---

## Step 7 — Query the `/data` endpoint

### Time array (1-D ndarray)

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/time" \
  | python -m json.tool
```

Expected `value`:

```json
{
    "_type": "numpy.ndarray",
    "dtype": "float64",
    "bytes": "<base64>"
}
```

### Nested structure field

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/global_quantities/ip" \
  | python -m json.tool
```

### Array-of-structures index

```bash
# profiles_1d[0] → grid → rho_tor_norm
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/profiles_1d/0/grid/rho_tor_norm" \
  | python -m json.tool
```

### Doubly-nested AoS

```bash
# profiles_1d[2] → ion[1] → density
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/profiles_1d/2/ion/1/density" \
  | python -m json.tool
```

### Ion atomic mass (scalar FLT_0D)

```bash
# profiles_1d[2] → ion[0] → element[0] → a
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/profiles_1d/2/ion/0/element/0/a" \
  | python -m json.tool
# → {"value": 1.0, ...}
```

### String scalar

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/ids_properties/comment" \
  | python -m json.tool
# → {"value": "local test", ...}
```

### With explicit occurrence (default is 0)

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/time&occurrence=0" \
  | python -m json.tool
```

### Pin to a specific output file

```bash
FILE_UUID=$(curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID" \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['outputs'][0]['uuid'])")

curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/time&file_uuid=$FILE_UUID" \
  | python -m json.tool
```

---

## Step 8 — Verify error cases

### Missing `path` → 400

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data"
# → {"error": "Query parameter 'path' is required"}
```

### Path stops at a structure, not a leaf → 400

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/profiles_1d"
# → {"error": "path does not point to a scalar/array leaf ..."}
```

### Non-existent field → 400

```bash
curl -s "http://localhost:5000/v1.2/simulation/$SIM_UUID/data?path=core_profiles/does_not_exist"
# → {"error": "Invalid IDS path ..."}
```

### Unknown simulation UUID → 404

```bash
curl -s "http://localhost:5000/v1.2/simulation/00000000000000000000000000000000/data?path=core_profiles/time"
# → {"error": "Simulation ... not found."}
```

---

## Step 9 — Swagger UI

All endpoints are browsable interactively at:

```
http://localhost:5000/v1.2/docs
```

The `/simulation/{sim_id}/data` resource appears under the **data** namespace.

---

## Decoding the numpy array response

The `value` field for array data uses SimDB's `CustomEncoder` format.
Decode it on the client side:

```python
import base64, numpy as np, requests

r = requests.get(
    "http://localhost:5000/v1.2/simulation/{SIM_UUID}/data",
    params={"path": "core_profiles/time"},
)
v = r.json()["value"]
array = np.frombuffer(base64.b64decode(v["bytes"]), dtype=v["dtype"])
print(array)  # [1. 2. 3.]
```
