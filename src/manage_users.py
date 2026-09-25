import argparse
import getpass
import sys

import auth
import quotas
import setup_app_db
from db import get_connection


def _password(args) -> str:
    if args.password:
        return args.password
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return first


def _limit(value):
    """CLI value -> stored limit. 'default' resets to the role default."""
    if value is None:
        return "unchanged"
    return None if value.lower() == "default" else int(value)


def main():
    parser = argparse.ArgumentParser(description="Administer CrossScan users (run from src/).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup-db", help="create the app tables")

    create = sub.add_parser("create-user", help="create a user")
    create.add_argument("username")
    create.add_argument("--role", choices=auth.ROLES, default="user")
    create.add_argument("--password", help="omit to be prompted (recommended)")

    sub.add_parser("list-users", help="list users with today's usage")

    limits = sub.add_parser("set-limits", help="set per-user daily limits ('default' = role default)")
    limits.add_argument("username")
    limits.add_argument("--requests", help="daily requests, or 'default'")
    limits.add_argument("--tokens", help="daily tokens, or 'default'")

    for name in ("deactivate", "activate"):
        sub.add_parser(name, help=f"{name} a user").add_argument("username")

    reset = sub.add_parser("reset-password", help="set a new password (signs the user out everywhere)")
    reset.add_argument("username")
    reset.add_argument("--password")

    args = parser.parse_args()

    if args.command == "setup-db":
        setup_app_db.setup_app_db()
        return

    conn = get_connection()
    try:
        if args.command == "create-user":
            user = auth.create_user(conn, args.username, _password(args), args.role)
            print(f"Created {user['role']} '{user['username']}'.")

        elif args.command == "list-users":
            for user in auth.list_users(conn):
                usage = quotas.remaining(conn, user)
                request_limit = "unlimited" if usage["requests_limit"] is None else usage["requests_limit"]
                token_limit = "unlimited" if usage["tokens_limit"] is None else usage["tokens_limit"]
                print(
                    f"{user['username']:<20} {user['role']:<6} {'active' if user['is_active'] else 'DISABLED':<8} "
                    f"requests {usage['requests_used']}/{request_limit}  tokens {usage['tokens_used']}/{token_limit}"
                )

        elif args.command == "set-limits":
            fields = {}
            for column, raw in (("daily_request_limit", args.requests), ("daily_token_limit", args.tokens)):
                value = _limit(raw)
                if value != "unchanged":
                    fields[column] = value
            print("Updated." if auth.update_user(conn, args.username, **fields) else "No such user / nothing to change.")

        elif args.command in ("deactivate", "activate"):
            done = auth.update_user(conn, args.username, is_active=(args.command == "activate"))
            print("Done." if done else "No such user.")

        elif args.command == "reset-password":
            print("Password updated." if auth.set_password(conn, args.username, _password(args)) else "No such user.")
    except auth.AuthError as e:
        sys.exit(str(e))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
