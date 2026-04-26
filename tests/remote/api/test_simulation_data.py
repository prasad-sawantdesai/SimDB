"""Unit tests for simulation_data.py helpers."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from simdb.remote.apis.v1_2.simulation_data import (
    _to_python,
    _walk_nonempty,
    _scan_imas_file,
)


# ── Test 1: _to_python scalar and array conversions ──────────────────────── #

class TestToPython:
    def test_numpy_integer_converts_to_int(self):
        assert _to_python(np.int32(42)) == 42
        assert isinstance(_to_python(np.int32(42)), int)

    def test_numpy_float_converts_to_float(self):
        result = _to_python(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_nan_inf_become_none(self):
        assert _to_python(np.float64(float("nan"))) is None
        assert _to_python(np.float64(float("inf"))) is None
        assert _to_python(np.float64(float("-inf"))) is None

    def test_numpy_array_becomes_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_python(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_array_nan_inf_become_none(self):
        arr = np.array([1.0, float("nan"), float("inf"), 2.0])
        result = _to_python(arr)
        assert result == [1.0, None, None, 2.0]

    def test_numpy_2d_array_converts_to_nested_list(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _to_python(arr)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_plain_python_passthrough(self):
        assert _to_python("hello") == "hello"
        assert _to_python(99) == 99
        assert _to_python(None) is None

    def test_numpy_bool_converts_to_bool(self):
        assert _to_python(np.bool_(True)) is True
        assert isinstance(_to_python(np.bool_(True)), bool)


# ── Test 2: _walk_nonempty statistics for 1-D array ──────────────────────── #

class TestWalkNonemptyStats:
    """Verify that _walk_nonempty attaches correct statistics to field dicts."""

    def _make_primitive(self, value, units="A", dtype="FLT_1D", doc="test field"):
        """Build a minimal IDSPrimitive-like mock."""
        mock = MagicMock()
        mock.value = value
        mock.data_type = dtype
        mock.shape = list(np.asarray(value).shape) if isinstance(value, np.ndarray) else []
        mock.metadata.units = units
        mock.metadata.documentation = doc
        mock._path.__str__ = lambda s: "leaf"
        # iter_nonempty_ yields the mock itself
        mock.iter_nonempty_ = MagicMock(return_value=[mock])

        # Make isinstance checks work
        from imas.ids_primitive import IDSPrimitive
        mock.__class__ = IDSPrimitive
        return mock

    def test_1d_array_stats_populated(self):
        val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prim = self._make_primitive(val)
        out = []
        _walk_nonempty(prim, "", out)
        assert len(out) == 1
        f = out[0]
        assert f["n_elements"] == 5
        assert f["n_nan"] == 0
        assert f["min"] == pytest.approx(1.0)
        assert f["max"] == pytest.approx(5.0)
        assert f["mean"] == pytest.approx(3.0)
        assert f["is_monotonic"] is True

    def test_1d_non_monotonic_array(self):
        val = np.array([1.0, 3.0, 2.0])
        prim = self._make_primitive(val)
        out = []
        _walk_nonempty(prim, "", out)
        assert out[0]["is_monotonic"] is False

    def test_nan_counted_correctly(self):
        val = np.array([1.0, float("nan"), 3.0])
        prim = self._make_primitive(val)
        out = []
        _walk_nonempty(prim, "", out)
        f = out[0]
        assert f["n_nan"] == 1
        assert f["min"] == pytest.approx(1.0)
        assert f["max"] == pytest.approx(3.0)

    def test_scalar_stores_value(self):
        prim = self._make_primitive(np.float64(42.0), dtype="FLT_0D")
        out = []
        _walk_nonempty(prim, "", out)
        assert out[0]["value"] == pytest.approx(42.0)

    def test_string_stores_value(self):
        prim = self._make_primitive("ITER tokamak", dtype="STR_0D")
        out = []
        _walk_nonempty(prim, "", out)
        assert out[0]["value"] == "ITER tokamak"

    def test_units_cleaned(self):
        prim = self._make_primitive(np.float64(1.0), units="?")
        out = []
        _walk_nonempty(prim, "", out)
        assert out[0]["units"] == ""


# ── Test 3: _scan_imas_file returns correct structure ────────────────────── #

class TestScanImasFile:
    """Verify _scan_imas_file output structure without a real HDF5 file."""

    def _make_imas_file(self):
        from simdb.uri import URI
        f = MagicMock()
        f.uri = URI(scheme="imas", path=None, query={"backend": "mdsplus", "path": "/fake"})
        return f

    def test_empty_ids_returns_empty_dict(self):
        """If every IDS has no occurrences, result must be {}."""
        imas_file = self._make_imas_file()

        mock_entry = MagicMock()
        mock_entry.list_all_occurrences.return_value = []

        mock_factory = MagicMock()
        mock_factory.ids_names.return_value = ["core_profiles", "equilibrium"]

        with (
            patch("simdb.remote.apis.v1_2.simulation_data.open_imas", return_value=mock_entry),
            patch("simdb.remote.apis.v1_2.simulation_data.__import__") as _,
        ):
            import importlib, types
            fake_imas = types.ModuleType("imas")
            fake_imas.IDSFactory = lambda: mock_factory
            with patch.dict("sys.modules", {"imas": fake_imas}):
                result = _scan_imas_file(imas_file)

        assert result == {}
        mock_entry.close.assert_called_once()

    def test_ids_filter_scans_only_one_ids(self):
        """ids_filter must restrict scanning to that IDS only."""
        imas_file = self._make_imas_file()

        mock_entry = MagicMock()
        mock_entry.list_all_occurrences.return_value = []

        mock_factory = MagicMock()
        mock_factory.ids_names.return_value = ["core_profiles", "equilibrium", "summary"]

        import types
        fake_imas = types.ModuleType("imas")
        fake_imas.IDSFactory = lambda: mock_factory

        with (
            patch("simdb.remote.apis.v1_2.simulation_data.open_imas", return_value=mock_entry),
            patch.dict("sys.modules", {"imas": fake_imas}),
        ):
            _scan_imas_file(imas_file, ids_filter="equilibrium")

        # list_all_occurrences should be called exactly once (for "equilibrium" only)
        assert mock_entry.list_all_occurrences.call_count == 1
        mock_entry.list_all_occurrences.assert_called_with("equilibrium")


# ── Test 4: /simulation/<sim_id>/data — missing path query param ─────────── #

class TestSimulationDataEndpoint:
    """HTTP-level tests for the /data endpoint using the Flask test client."""

    @pytest.fixture
    def client(self, tmp_path):
        import base64
        from simdb.config import Config
        from simdb.remote.app import create_app
        from simdb.cli.manifest import Manifest
        from simdb.database.models import Simulation as SimModel

        config = Config()
        config.load()
        db_fd, db_file = __import__("tempfile").mkstemp()
        config.set_option("database.type", "sqlite")
        config.set_option("database.file", db_file)
        config.set_option("server.admin_password", "test")
        config.set_option("server.upload_folder", str(tmp_path))
        config.set_option("authentication.type", "None")
        config.set_option("server.copy_files", False)

        app = create_app(config=config, testing=True, debug=True)
        app.testing = True

        sim = SimModel(Manifest())
        with app.test_client() as c:
            app.db.insert_simulation(sim)
            app.db.session.commit()
            app.db.session.close()
            self._sim_id = str(sim.uuid)
            yield c

        __import__("os").close(db_fd)
        Path(db_file).unlink(missing_ok=True)

    _headers = {"Authorization": "Basic " + __import__("base64").b64encode(b"admin:test").decode()}

    def test_data_missing_path_returns_400(self, client):
        rv = client.get(
            f"/v1.2/simulation/{self._sim_id}/data",
            headers=self._headers,
        )
        assert rv.status_code == 400
        assert "path" in rv.json["error"].lower()

    def test_data_no_imas_outputs_returns_404(self, client):
        rv = client.get(
            f"/v1.2/simulation/{self._sim_id}/data?path=core_profiles/global_quantities/ip",
            headers=self._headers,
        )
        assert rv.status_code == 404


# ── Test 5: /simulation/<sim_id>/fields — cache hit ──────────────────────── #

class TestFieldsCacheHit:
    """Verify the in-memory cache is returned without a second IMAS scan."""

    def test_second_call_returns_cached_true(self, tmp_path):
        import base64
        from simdb.config import Config
        from simdb.remote.app import create_app
        from simdb.cli.manifest import Manifest, DataObject
        from simdb.database.models import Simulation as SimModel
        from simdb.uri import URI
        from simdb.remote.apis.v1_2 import simulation_data as sd

        headers = {"Authorization": "Basic " + base64.b64encode(b"admin:test").decode()}

        config = Config()
        config.load()
        _, db_file = __import__("tempfile").mkstemp()
        config.set_option("database.type", "sqlite")
        config.set_option("database.file", db_file)
        config.set_option("server.admin_password", "test")
        config.set_option("server.upload_folder", str(tmp_path))
        config.set_option("authentication.type", "None")
        config.set_option("server.copy_files", False)

        app = create_app(config=config, testing=True, debug=True)
        app.testing = True

        sim = SimModel(Manifest())
        file_uuid = __import__("uuid").uuid4()

        # Inject a fake IMAS output into the simulation
        fake_output = MagicMock()
        fake_output.type = DataObject.Type.IMAS
        fake_output.uuid = file_uuid
        fake_output.uri = URI(scheme="imas", path=None, query={"path": "/fake"})

        with app.test_client() as client:
            app.db.insert_simulation(sim)
            app.db.session.commit()

            # Pre-populate the cache
            cache_key = (str(file_uuid), None, None)
            sd._fields_cache[cache_key] = {"core_profiles": {0: [{"path": "time"}]}}

            sim_obj = app.db.get_simulation(str(sim.uuid))
            sim_obj.outputs = [fake_output]

            with patch.object(app.db, "get_simulation", return_value=sim_obj):
                rv = client.get(
                    f"/v1.2/simulation/{sim.uuid}/fields",
                    headers=headers,
                )

        assert rv.status_code == 200
        data = rv.json
        assert data.get("cached") is True
        assert "core_profiles" in data["ids"]
