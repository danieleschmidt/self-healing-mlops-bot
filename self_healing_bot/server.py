"""Production server entry point with graceful shutdown and configuration."""

import asyncio
import signal
import sys
import os
import logging
from typing import Optional
from pathlib import Path

import uvicorn
from uvicorn.config import LOGGING_CONFIG
import structlog

from .core.config import config
from .monitoring.logging import setup_logging, get_logger
from .monitoring.metrics import prometheus_metrics


logger = get_logger(__name__)


class GracefulServer:
    """Server with graceful shutdown capabilities."""
    
    def __init__(self):
        self.server: Optional[uvicorn.Server] = None
        self.should_exit = False
        self.force_exit = False
        
    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        if sys.platform == "win32":
            # Windows signal handling
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        else:
            # Unix signal handling
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._handle_signal, sig)
    
    def _handle_signal(self, sig=None):
        """Handle shutdown signals."""
        if sig:
            logger.info(f"Received signal {sig}, initiating graceful shutdown")
        else:
            logger.info("Received shutdown signal, initiating graceful shutdown")
        
        if self.should_exit:
            logger.warning("Force shutdown initiated")
            self.force_exit = True
        else:
            self.should_exit = True
        
        if self.server:
            self.server.should_exit = True
    
    async def serve(self):
        """Start the server with graceful shutdown."""
        # Setup logging
        setup_logging()
        logger.info("Initializing Self-Healing MLOps Bot Server")
        
        # Validate configuration
        try:
            self._validate_configuration()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return 1
        
        # Create uvicorn configuration
        uvicorn_config = self._create_uvicorn_config()
        
        # Create server
        self.server = uvicorn.Server(uvicorn_config)
        
        # Install signal handlers
        self.install_signal_handlers()
        
        try:
            # Start server
            logger.info(
                f"Starting server on {config.host}:{config.port}",
                environment=config.environment,
                debug=config.debug
            )
            await self.server.serve()
            
        except Exception as e:
            logger.exception(f"Server error: {e}")
            return 1
        finally:
            # Cleanup
            await self._cleanup()
        
        return 0
    
    def _validate_configuration(self):
        """Validate server configuration."""
        errors = []
        
        # Check GitHub configuration
        if not config.github_app_id:
            errors.append("GITHUB_APP_ID is required")
        
        if not config.github_webhook_secret and config.environment != "development":
            errors.append("GITHUB_WEBHOOK_SECRET is required for production")
        
        # Check private key file
        private_key_path = Path(config.github_private_key_path)
        if not private_key_path.exists():
            errors.append(f"GitHub private key file not found: {private_key_path}")
        
        # Check port availability (basic check)
        if not (1 <= config.port <= 65535):
            errors.append(f"Invalid port number: {config.port}")
        
        # Check log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {config.log_level}")
        
        if errors:
            raise ValueError("Configuration validation failed:\\n" + "\\n".join(errors))
        
        logger.info("Configuration validation passed")
    
    def _create_uvicorn_config(self) -> uvicorn.Config:
        """Create uvicorn server configuration."""
        # Configure logging
        log_config = LOGGING_CONFIG.copy()
        
        # Update log format for production
        if config.environment != "development":
            log_config["formatters"]["default"]["fmt"] = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            log_config["formatters"]["access"]["fmt"] = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Create uvicorn config
        uvicorn_config = uvicorn.Config(
            app="self_healing_bot.web.app:app",
            host=config.host,
            port=config.port,
            reload=config.debug and config.environment == "development",
            log_level=config.log_level.lower(),
            log_config=log_config,
            access_log=True,
            server_header=False,  # Don't expose server version
            date_header=True,
            use_colors=config.environment == "development",
            loop="asyncio",
            # Production optimizations
            workers=1,  # Single worker for now, can be increased
            backlog=2048,
            timeout_keep_alive=5,
            timeout_graceful_shutdown=30,
            limit_concurrency=1000,
            limit_max_requests=10000,
        )
        
        # Additional production settings
        if config.environment == "production":
            uvicorn_config.ssl_keyfile = os.getenv("SSL_KEYFILE")
            uvicorn_config.ssl_certfile = os.getenv("SSL_CERTFILE")
            uvicorn_config.ssl_ca_certs = os.getenv("SSL_CA_CERTS")
        
        return uvicorn_config
    
    async def _cleanup(self):
        """Cleanup resources during shutdown."""
        logger.info("Starting cleanup process")
        
        try:
            # Import here to avoid circular imports
            from .web.app import app_state
            
            # Shutdown webhook handler
            if app_state.get("webhook_handler"):
                await app_state["webhook_handler"].shutdown()
            
            # Shutdown bot (if it has cleanup methods)
            if app_state.get("bot") and hasattr(app_state["bot"], "shutdown"):
                await app_state["bot"].shutdown()
            
            # Flush metrics
            if prometheus_metrics:
                logger.info("Flushing metrics")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.exception(f"Error during cleanup: {e}")


def create_app():
    """Create FastAPI application (for WSGI servers like Gunicorn)."""
    from .web.app import app
    return app


async def main():
    """Main server entry point."""
    server = GracefulServer()
    return await server.serve()


def run_server():
    """Synchronous server entry point."""
    try:
        if sys.version_info >= (3, 11):
            # Python 3.11+ has asyncio.Runner
            with asyncio.Runner() as runner:
                exit_code = runner.run(main())
        else:
            # Python < 3.11 compatibility
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            exit_code = asyncio.run(main())
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal server error: {e}")
        sys.exit(1)


def run_production_server():
    """Production server entry point with Gunicorn-compatible settings."""
    import multiprocessing
    
    # Production configuration
    workers = os.getenv("WORKERS", str(multiprocessing.cpu_count() * 2 + 1))
    worker_class = "uvicorn.workers.UvicornWorker"
    
    # Gunicorn configuration
    gunicorn_config = {
        "bind": f"{config.host}:{config.port}",
        "workers": int(workers),
        "worker_class": worker_class,
        "worker_connections": 1000,
        "max_requests": 10000,
        "max_requests_jitter": 1000,
        "timeout": 30,
        "keepalive": 2,
        "preload_app": True,
        "reload": config.debug,
        "log_level": config.log_level.lower(),
        "access_log_format": (
            "%(t)s %(h)s '%(r)s' %(s)s %(b)s '%(f)s' '%(a)s' %(D)s"
        ),
    }
    
    logger.info(
        f"Starting production server with {workers} workers",
        config=gunicorn_config
    )
    
    try:
        from gunicorn.app.base import BaseApplication
        
        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)
            
            def load(self):
                return self.application
        
        # Create and run Gunicorn application
        app = create_app()
        StandaloneApplication(app, gunicorn_config).run()
        
    except ImportError:
        logger.warning(
            "Gunicorn not available, falling back to single-worker uvicorn"
        )
        run_server()


if __name__ == "__main__":
    # Determine which server to run based on environment
    if config.environment == "production" and os.getenv("USE_GUNICORN", "false").lower() == "true":
        run_production_server()
    else:
        run_server()