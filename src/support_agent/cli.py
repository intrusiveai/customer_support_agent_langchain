import argparse
import logging
import sys

from pydantic import ValidationError

from .config import Settings
from .results import Rejected, json_text
from .schema import SchemaError
from .session import SupportApplication, invoke_support


def main():
    parser = argparse.ArgumentParser(
        description="Intentionally vulnerable support simulation agent"
    )
    parser.add_argument(
        "--customer-id", required=True, help="Trusted identity selected by the local operator"
    )
    parser.add_argument("--message", help="Run one turn; omit for an interactive conversation")
    args = parser.parse_args()
    try:
        settings = Settings()
        try:
            handler = (
                logging.FileHandler(settings.log_file, encoding="utf-8")
                if settings.log_file is not None
                else logging.StreamHandler()
            )
        except OSError:
            print("Configuration error: cannot open SUPPORT_LOG_FILE for appending.", file=sys.stderr)
            raise SystemExit(1) from None
        handler.setLevel(settings.log_level)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[handler],
            force=True,
        )
        logging.getLogger("support_agent").setLevel(settings.log_level)
        with SupportApplication(settings) as app:
            for notice in app.notices:
                print(notice, file=sys.stderr)
            session = app.start_session(args.customer_id)
            if args.message is not None:
                print(json_text(invoke_support(app, session, args.message)))
                return
            print(f"Session {session.session_id}")
            while True:
                try:
                    message = input("> ")
                except EOFError:
                    break
                if message.strip() in {"/quit", "/exit"}:
                    break
                if message:
                    print(json_text(invoke_support(app, session, message)))
    except ValidationError as exc:
        fields = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        )
        print(f"Configuration error: {fields}", file=sys.stderr)
        raise SystemExit(1) from None
    except (Rejected, SchemaError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        # Do not dump provider/connection exceptions, which may contain credentials.
        print(
            f"{type(exc).__name__}: application failed; check configuration and application logs.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
