import logging
import os
from pathlib import Path
from typing import Optional, Type, cast

from alembic.config import Config as AlembicConfig
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from flask import Flask, jsonify, request
from flask.json import JSONDecoder, JSONEncoder
from flask_compress import Compress
from flask_cors import CORS

from simdb.config import Config
from simdb.database.models import Base
from simdb.json import CustomDecoder, CustomEncoder

from .apis import blueprints
from .core.auth._authenticator import Authenticator
from .core.cache import cache
from .core.typing import SimDBApp

compress = Compress()


# Path to alembic.ini, located at the project root (two levels above this file's
# package: src/simdb/remote/ -> src/simdb/ -> src/ -> project root)
_ALEMBIC_INI = Path("alembic.ini")


def check_migrations(app: "SimDBApp") -> None:
    """Check that the database is up-to-date with the latest Alembic migration.

    Logs a warning if the database is behind the head revision, and raises a
    :class:`RuntimeError` if the database has not been initialised at all (i.e.
    the ``alembic_version`` table is absent).
    """
    alembic_cfg = AlembicConfig(str(_ALEMBIC_INI))
    script = ScriptDirectory.from_config(alembic_cfg)
    head_revision = script.get_current_head()

    engine = app.db.engine
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_revision = context.get_current_revision()

    if current_revision is None:
        raise RuntimeError(
            "The database has not been initialised. "
            f"Run 'DATABASE_URL={engine.url} alembic upgrade head' before starting the "
            "server. "
        )

    if current_revision != head_revision:
        raise RuntimeError(
            f"Database schema is out of date: current revision is {current_revision}, "
            f"but the latest revision is {head_revision}. "
            f"Run 'DATABASE_URL={engine.url} alembic upgrade head' to apply pending "
            "migrations. "
        )
    else:
        app.logger.info(
            "Database schema is up to date (revision %s).", current_revision
        )


def run_migrations(app: "SimDBApp") -> None:
    """Run the database migrations."""
    config = AlembicConfig(_ALEMBIC_INI)
    config.set_main_option("script_location", "alembic")
    script = ScriptDirectory.from_config(config)

    def upgrade(rev, context):
        return script._upgrade_revs("head", rev)

    engine = app.db.engine
    with engine.connect() as conn:
        context = MigrationContext.configure(
            conn, opts={"target_metadata": Base.metadata, "fn": upgrade}
        )

        with context.begin_transaction(), Operations.context(context):
            context.run_migrations()


def create_app(
    config: Optional[Config] = None, testing=False, debug=False, profile=False
):
    if config is None:
        config_file = os.environ.get("SIMDB_CONFIG_FILE", default="app.cfg")
        config = Config(config_file)
        config.load()
    flask_options = {k.upper(): v for (k, v) in config.get_section("flask", {}).items()}

    app = cast(SimDBApp, Flask(__name__))
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.config["TESTING"] = testing
    app.config["DEBUG"] = debug
    app.config["RESTX_INCLUDE_ALL_MODELS"] = True
    app.config["PROFILE"] = profile
    app.json_encoder = cast(Type[JSONEncoder], CustomEncoder)
    app.json_decoder = cast(Type[JSONDecoder], CustomDecoder)
    app.config.from_mapping(flask_options)
    app.simdb_config = config
    cache.init_app(app)
    compress.init_app(app)

    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_logger.handlers)
    app.logger.setLevel(gunicorn_logger.level)

    @app.route("/")
    def index():
        endpoints = []
        for ver in blueprints:
            endpoints.append(f"{request.url}{ver}")
        authentication_types = config.get_string_option("authentication.type").split(
            ","
        )
        authenticators = [
            Authenticator.get(auth_type) for auth_type in authentication_types
        ]
        return jsonify(
            {
                "endpoints": endpoints,
                "authentication": authenticators[0].Name,
                "authenticators": [auth.Name for auth in authenticators],
            }
        )

    for version, blueprint in blueprints.items():
        app.register_blueprint(blueprint, url_prefix=f"/{version}")

    # Preload Qdrant client + embedding model in background which will download/load model.
    import threading as _threading
    from simdb.remote.apis.v1_2 import qdrant_store as _qdrant_store
    _threading.Thread(target=_qdrant_store._init_qdrant, daemon=True, name="qdrant-preload").start()

    if not _ALEMBIC_INI.exists():
        raise RuntimeError(f"Alembic configuration not found at {_ALEMBIC_INI}.")

    if testing:
        run_migrations(app)

    try:
        check_migrations(app)
    except Exception as exc:
        app.logger.error("Migration check failed: %s", exc)
        raise
    return app
